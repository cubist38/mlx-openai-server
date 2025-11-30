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
from typing import Any, Dict, List, Optional

from PIL import Image
from loguru import logger
from fastapi import HTTPException
from ..core.queue import RequestQueue
from ..models.mflux import ImageGenerationModel
from ..utils.errors import create_error_response
from ..schemas.openai import ImageSize, ImageGenerationRequest, ImageGenerationResponse, ImageEditRequest, ImageEditResponse, ImageData


class MLXFluxHandler:
    """
    Handler class for making image generation requests to the underlying MLX Flux model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1, quantize: int = 8, 
                 config_name: str = "flux-schnell", lora_paths: Optional[List[str]] = None, 
                 lora_scales: Optional[List[float]] = None):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory or model name for Flux.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
            quantize (int): Quantization level for the model.
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

    def _parse_image_size(self, size: ImageSize):
        """Parse image size string to width, height tuple"""
        width, height = map(int, size.value.split('x'))
        return width, height

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image based on the request parameters.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ImageGenerationRequest object containing the generation parameters.
        
        Returns:
            ImageGenerationResponse: Response containing the generated image data.
        """
        request_id = f"image-{uuid.uuid4()}"
        
        try:
            width, height = 1024, 1024
            if request.size:
                width, height = self._parse_image_size(request.size)
            # Prepare request data
            request_data = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,
                "seed": request.seed,
                "guidance": request.guidance_scale,
                "width": width,
                "height": height
            }
            
            # Submit to the request queue
            image_result = await self.request_queue.submit(request_id, request_data)
            
            # Convert PIL Image to base64
            image_data = self._image_to_base64(image_result)
            
            # Create response
            response = ImageGenerationResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=image_data)]
            )
            
            return response
            
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.", 
                "rate_limit_exceeded", 
                HTTPStatus.TOO_MANY_REQUESTS
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in image generation for request {request_id}: {str(e)}")
            content = create_error_response(
                f"Failed to generate image: {str(e)}", 
                "server_error", 
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            raise HTTPException(status_code=500, detail=content)

    async def edit_image(self, image_edit_request: ImageEditRequest) -> ImageEditResponse:
        """
        Edit an image based on the request parameters.
        
        Args:
            image_edit_request: Request parameters for image editing
            
        Returns:
            ImageEditResponse: Response containing the edited image data
            
        Raises:
            HTTPException: For validation errors, queue capacity issues, or processing failures
        """
        image = image_edit_request.image
        # Validate image file type and size
        if not image.content_type or image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail="Image must be a PNG, JPEG, or JPG file"
            )
        
        # Check file size (limit to 10MB)
        if hasattr(image, 'size') and image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file size must be less than 10MB"
            )
        
        # Validate request parameters
        if not image_edit_request.prompt or not image_edit_request.prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        request_id = f"image-edit-{uuid.uuid4()}"
        temp_file_path = None
        
        try:
            # Read and validate image data
            image_data = await image.read()
            if not image_data:
                raise HTTPException(
                    status_code=400,
                    detail="Empty image file received"
                )
            
            # Load and process image using proper utility function
            try:
                input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as img_error:
                logger.error(f"Failed to process image: {str(img_error)}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or corrupted image file"
                )

            width, height = input_image.size
            if image_edit_request.size is not None:
                width, height = self._parse_image_size(image_edit_request.size)

            # Create temporary file with proper cleanup handling
            try:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=".png",
                    prefix=f"edit_{request_id}_"
                )
                temp_file_path = temp_file.name
                input_image.save(temp_file_path, format="PNG")
                temp_file.close()
            except Exception as temp_error:
                logger.error(f"Failed to create temporary file: {str(temp_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process image for editing"
                )

            # Prepare request data with all necessary parameters
            request_data = {
                "image_path": temp_file_path,
                "prompt": image_edit_request.prompt.strip(),
                "steps": image_edit_request.steps,
                "seed": image_edit_request.seed,                
                "negative_prompt": image_edit_request.negative_prompt,
                "width": width,
                "height": height,
                "guidance": image_edit_request.guidance_scale,
            }
            
            # Submit to the request queue
            image_result = await self.request_queue.submit(request_id, request_data)
            
            # resize image to original size
            image_result = image_result.resize((width, height))

            # Convert PIL Image to base64
            image_data_b64 = self._image_to_base64(image_result)
            
            # Create response
            response = ImageEditResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=image_data_b64)]
            )
            
            logger.info(f"Successfully processed image edit request {request_id}")
            return response

        except asyncio.QueueFull:
            logger.error(f"Queue at capacity for image edit request {request_id}")
            content = create_error_response(
                "Too many requests. Service is at capacity.", 
                "rate_limit_exceeded", 
                HTTPStatus.TOO_MANY_REQUESTS
            )
            raise HTTPException(status_code=429, detail=content)
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in image edit for request {request_id}: {str(e)}")
            content = create_error_response(
                f"Failed to edit image: {str(e)}", 
                "server_error", 
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            raise HTTPException(status_code=500, detail=content)
            
        finally:
            # Ensure cleanup of temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {str(cleanup_error)}")
            
            # Force garbage collection to free memory
            gc.collect()
        
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
            prompt = request_data.get("prompt", "")
            negative_prompt = request_data.get("negative_prompt")
            steps = request_data.get("steps", 4)
            seed = request_data.get("seed", 42)
            width = request_data.get("width", 1024)
            height = request_data.get("height", 1024)
            image_path = request_data.get("image_path")  # For image editing
            guidance = request_data.get("guidance", 2.5)
    
            # Prepare model parameters
            model_params = {
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance": guidance,
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