import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
import gc
from io import BytesIO
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile
from ..core.queue import RequestQueue
from ..models.mflux import ImageGenerationModel
from ..utils.errors import create_error_response
from ..schemas.openai import ImageSize, ImageGenerationRequest, ImageGenerationResponse, ImageEditRequest, ImageEditResponse, ImageData


class MLXFluxHandler:
    """
    Handler class for making image generation requests to the underlying MLX Flux model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1, quantize: Optional[int] = None, 
                 config_name: str = "flux-schnell", lora_paths: Optional[List[str]] = None, 
                 lora_scales: Optional[List[float]] = None):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory or model name for Flux.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
            quantize (Optional[int]): Quantization level for the model. Must be 4, 8, or 16 if provided.
            config_name (str): Model config name (flux-schnell, flux-dev, etc.).
            lora_paths (List[str]): List of LoRA adapter paths.
            lora_scales (List[float]): List of LoRA scales.
        """
        self.model_path = model_path
        self.quantize = quantize
        self.config_name = config_name
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        
        self.model = ImageGenerationModel(
            model_path=model_path, 
            quantize=quantize,
            config_name=config_name,
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )
        self.model_created = int(time.time())  # Store creation time when model is loaded
        
        # Initialize request queue for image generation tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXFluxHandler with model path: {model_path}, config name: {config_name}")
        if lora_paths:
            logger.info(f"Using LoRA adapters: {lora_paths} with scales: {lora_scales}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local"
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []

    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None):
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {
                "max_concurrency": 1,
                "timeout": 300,
                "queue_size": 100
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency", 1),
            timeout=queue_config.get("timeout", 300),
            queue_size=queue_config.get("queue_size", 100)
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXFluxHandler and started request queue")
        logger.info(f"Queue configuration: {queue_config}")

    def _parse_image_size(self, size: ImageSize) -> tuple[int, int]:
        """
        Parse image size string to width, height tuple.
        
        Parameters
        ----------
        size : ImageSize
            Image size enum value (e.g., "1024x1024").
            
        Returns
        -------
        tuple[int, int]
            Width and height as integers.
        """
        width, height = map(int, size.value.split('x'))
        return width, height
    
    def _build_generation_request_data(
        self, 
        request: ImageGenerationRequest, 
        width: int, 
        height: int
    ) -> dict[str, Any]:
        """
        Build request data dictionary for image generation.
        
        Parameters
        ----------
        request : ImageGenerationRequest
            The image generation request.
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
            
        Returns
        -------
        dict[str, Any]
            Request data for the model.
        """
        return {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "seed": request.seed,
            "guidance": request.guidance_scale,
            "width": width,
            "height": height
        }
    
    def _build_edit_request_data(self, image_edit_request: ImageEditRequest, temp_file_paths: list[str]) -> dict[str, Any]:
        """
        Build request data dictionary for image editing.
        
        Parameters
        ----------
        image_edit_request : ImageEditRequest
            The image editing request.
        temp_file_paths : list[str]
            List of temporary file paths.
            
        Returns
        -------
        dict[str, Any]
            Request data for the model.
        """
        return {
            "image_path": temp_file_paths[0],
            "prompt": image_edit_request.prompt,
            "negative_prompt": image_edit_request.negative_prompt,
            "steps": image_edit_request.steps,
            "seed": image_edit_request.seed,
            "guidance": image_edit_request.guidance_scale,
            "image_paths": temp_file_paths
        }
    
    def _create_image_response(self, image_result: Image.Image) -> ImageGenerationResponse:
        """
        Create image generation response from PIL Image.
        
        Parameters
        ----------
        image_result : Image.Image
            The generated PIL Image.
            
        Returns
        -------
        ImageGenerationResponse
            Response containing base64 encoded image data.
        """
        image_data_b64 = self._image_to_base64(image_result)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=image_data_b64)]
        )

    def _create_edit_response(self, image_result: Image.Image) -> ImageEditResponse:
        """
        Create image editing response from PIL Image.
        
        Parameters
        ----------
        image_result : Image.Image
            The edited PIL Image.
        """
        image_data_b64 = self._image_to_base64(image_result)
        return ImageEditResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=image_data_b64)]
        )
    
    def _handle_queue_full_error(self, request_id: str) -> None:
        """
        Handle queue capacity errors.
        
        Parameters
        ----------
        request_id : str
            The request ID for logging.
            
        Raises
        ------
        HTTPException
            429 error with rate limit details.
        """
        logger.error(f"Queue at capacity for request {request_id}")
        content = create_error_response(
            "Too many requests. Service is at capacity.", 
            "rate_limit_exceeded", 
            HTTPStatus.TOO_MANY_REQUESTS
        )
        raise HTTPException(status_code=429, detail=content)
    
    def _handle_generation_error(self, request_id: str, error: Exception) -> None:
        """
        Handle general generation errors.
        
        Parameters
        ----------
        request_id : str
            The request ID for logging.
        error : Exception
            The exception that occurred.
            
        Raises
        ------
        HTTPException
            500 error with error details.
        """
        logger.error(f"Error in image generation for request {request_id}: {str(error)}")
        content = create_error_response(
            f"Failed to generate image: {str(error)}", 
            "server_error", 
            HTTPStatus.INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=500, detail=content)

    def _handle_edit_error(self, request_id: str, error: Exception) -> None:
        """
        Handle general editing errors.
        
        Parameters
        ----------
        request_id : str
            The request ID for logging.
        error : Exception
            The exception that occurred.
            
        Raises
        ------
        HTTPException
            500 error with error details.
        """
        logger.error(f"Error in image editing for request {request_id}: {str(error)}")
        content = create_error_response(
            f"Failed to edit image: {str(error)}", 
            "server_error", 
            HTTPStatus.INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=500, detail=content)
    
    def _validate_image_file(self, image: UploadFile, idx: int) -> None:
        """
        Validate image file type and size.
        
        Parameters
        ----------
        image : UploadFile
            The uploaded image file to validate.
        idx : int
            Index of the image (for error messages).
            
        Raises
        ------
        HTTPException
            If validation fails.
        """
        if not image.content_type or image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Image {idx + 1} must be a PNG, JPEG, or JPG file"
            )
        
        if hasattr(image, 'size') and image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Image {idx + 1} file size must be less than 10MB"
            )
    
    async def _upload_to_temp_file(self, image: UploadFile, idx: int, request_id: str) -> str:
        """
        Read, process, and save uploaded image to a temporary file.
        
        Parameters
        ----------
        image : UploadFile
            The uploaded image file.
        idx : int
            Index of the image (for error messages and file naming).
        request_id : str
            Request ID for file naming.
            
        Returns
        -------
        str
            Path to the temporary file.
            
        Raises
        ------
        HTTPException
            If image processing or file creation fails.
        """
        # Read and validate image data
        image_data = await image.read()
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail=f"Empty image file received for image {idx + 1}"
            )
        
        # Load and process image
        try:
            input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            input_image = ImageOps.exif_transpose(input_image)
        except Exception as img_error:
            logger.error(f"Failed to process image {idx + 1}: {str(img_error)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted image file for image {idx + 1}"
            )
        
        # Create and save to temporary file
        try:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".png",
                prefix=f"edit_{request_id}_{idx + 1}_"
            )
            temp_file_path = temp_file.name
            input_image.save(temp_file_path, format="PNG")
            temp_file.close()
            return temp_file_path
        except Exception as temp_error:
            logger.error(f"Failed to create temporary file for image {idx + 1}: {str(temp_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process image {idx + 1} for editing"
            )
    
    def _cleanup_temp_files(self, temp_file_paths: list[str]) -> None:
        """Clean up temporary files and force garbage collection."""
        for temp_file_path in temp_file_paths:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {str(cleanup_error)}")
        gc.collect()

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image based on the request parameters.
        
        Parameters
        ----------
        request : ImageGenerationRequest
            Request object containing the generation parameters.
        
        Returns
        -------
        ImageGenerationResponse
            Response containing the generated image data.
            
        Raises
        ------
        HTTPException
            For queue capacity issues or processing failures.
        """
        request_id = f"image-{uuid.uuid4()}"
        
        try:
            # Parse image dimensions
            width, height = 1024, 1024
            if request.size:
                width, height = self._parse_image_size(request.size)
            
            # Build and submit request
            request_data = self._build_generation_request_data(request, width, height)
            image_result = await self.request_queue.submit(request_id, request_data)
            
            # Create and return response
            return self._create_image_response(image_result)
            
        except asyncio.QueueFull:
            self._handle_queue_full_error(request_id)
            
        except Exception as e:
            self._handle_generation_error(request_id, e)

    async def edit_image(self, image_edit_request: ImageEditRequest) -> ImageEditResponse:
        """
        Edit an image or multiple images based on the request parameters.
        
        Parameters
        ----------
        image_edit_request : ImageEditRequest
            Request parameters for image editing.
            
        Returns
        -------
        ImageEditResponse
            Response containing the edited image data.
            
        Raises
        ------
        HTTPException
            For validation errors, queue capacity issues, or processing failures.
        """
        # Normalize and validate inputs
        images: list[UploadFile] = (
            image_edit_request.image 
            if isinstance(image_edit_request.image, list) 
            else [image_edit_request.image]
        )
        
        if not images:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required for image editing"
            )
                
        for idx, image in enumerate(images):
            self._validate_image_file(image, idx)

        request_id = f"image-edit-{uuid.uuid4()}"
        temp_file_paths: list[str] = []
        
        try:
            # Process all images to temporary files
            for idx, image in enumerate(images):
                temp_file_path = await self._upload_to_temp_file(image, idx, request_id)
                temp_file_paths.append(temp_file_path)

            # Submit request to queue
            request_data = self._build_edit_request_data(image_edit_request, temp_file_paths)
            
            image_result = await self.request_queue.submit(request_id, request_data)
            
            return self._create_edit_response(image_result)

        except asyncio.QueueFull:
            self._handle_queue_full_error(request_id)
            
        except HTTPException:
            raise
            
        except Exception as e:
            self._handle_edit_error(request_id, e)
            
        finally:
            self._cleanup_temp_files(temp_file_paths)
        
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object.
            
        Returns:
            str: Base64 encoded image string.
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')

    async def _process_request(self, request_data: Dict[str, Any]) -> Image.Image:
        """
        Process an image generation request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            Image.Image: The generated PIL Image.
        """
        try:
            # Extract request parameters
            prompt = request_data.get("prompt")
            negative_prompt = request_data.get("negative_prompt")
            steps = request_data.get("steps")
            seed = request_data.get("seed")
            width = request_data.get("width")
            height = request_data.get("height")
            image_path = request_data.get("image_path")  # For image editing
            guidance = request_data.get("guidance")
            image_paths = request_data.get("image_paths", [])
    
            # Prepare model parameters
            model_params = {
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance": guidance,
                "image_paths": image_paths,
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                model_params["negative_prompt"] = negative_prompt
            
            # Add image path for image editing if provided
            if image_path:
                model_params["image_path"] = image_path
                logger.info(f"Processing image edit with prompt: {prompt[:50]}... and image: {image_path}")
            else:
                logger.info(f"Generating image with prompt: {prompt[:50]}...")
            
            # Log all model parameters
            logger.info(f"Model inference configurations:")
            logger.info(f"  - Prompt: {prompt[:100]}...")
            logger.info(f"  - Negative prompt: {negative_prompt}")
            logger.info(f"  - Steps: {steps}")
            logger.info(f"  - Seed: {seed}")
            logger.info(f"  - Width: {width}")
            logger.info(f"  - Height: {height}")
            logger.info(f"  - Guidance scale: {guidance}")
            logger.info(f"  - Image path: {image_path}")
            logger.info(f"  - Model params: {model_params}")
            
            # Generate image
            image = self.model(
                prompt=prompt,
                seed=seed,
                **model_params
            )
            
            # Force garbage collection after model inference
            gc.collect()
            return image
            
        except Exception as e:
            logger.error(f"Error processing image generation request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics.
        
        Returns:
            Dict containing queue statistics.
        """
        if not hasattr(self, 'request_queue') or self.request_queue is None:
            return {"error": "Request queue not initialized"}
        
        stats = await self.request_queue.get_stats()
        return {
            "queue_size": stats.get("queue_size", 0),
            "active_requests": stats.get("active_requests", 0),
            "completed_requests": stats.get("completed_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "average_processing_time": stats.get("average_processing_time", 0)
        }

    async def cleanup(self):
        """
        Clean up resources and shut down the request queue.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        self._cleaned = True
        
        try:
            logger.info("Cleaning up MLXFluxHandler resources")
            if hasattr(self, 'request_queue') and self.request_queue:
                await self.request_queue.stop()
                logger.info("Request queue stopped successfully")
        except Exception as e:
            logger.error(f"Error during MLXFluxHandler cleanup: {str(e)}")
        
        # Force garbage collection
        gc.collect()
        logger.info("MLXFluxHandler cleanup completed")

    def __del__(self):
        """
        Destructor to ensure cleanup on object deletion.
        Note: Async cleanup cannot be reliably performed in __del__.
        Please use 'await cleanup()' explicitly.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        # Set flag to prevent multiple cleanup attempts
        self._cleaned = True

if __name__ == "__main__":
    handler = MLXFluxHandler(model_path="qwen-image", config_name="qwen-image")
    print(handler.model.get_model_info("qwen-image"))