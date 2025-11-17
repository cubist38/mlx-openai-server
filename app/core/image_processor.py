import gc
import asyncio
from PIL import Image
from loguru import logger
from io import BytesIO
from typing import List
from .base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    """Image processor for handling image files with caching, validation, and processing."""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        super().__init__(max_workers, cache_size)
        Image.MAX_IMAGE_PIXELS = 100000000  # Limit to 100 megapixels

    def _get_media_format(self, media_url: str, data: bytes = None) -> str:
        """Determine image format from URL or data."""
        # For images, we always save as JPEG for consistency
        return "jpg"

    def _validate_media_data(self, data: bytes) -> bool:
        """Basic validation of image data."""
        if len(data) < 100:  # Too small to be a valid image file
            return False
        
        # Check for common image file signatures
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'BM',  # BMP
            b'II*\x00',  # TIFF (little endian)
            b'MM\x00*',  # TIFF (big endian)
            b'RIFF',  # WebP (part of RIFF)
        ]
        
        for sig in image_signatures:
            if data.startswith(sig):
                return True
        
        # Additional check for WebP
        if data.startswith(b'RIFF') and b'WEBP' in data[:20]:
            return True
        
        return False

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 30  # Standard timeout for images

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 100 * 1024 * 1024  # 100 MB limit for images

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "image"

    def _resize_image_keep_aspect_ratio(self, image: Image.Image, max_size: int = 448) -> Image.Image:
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image to {new_width}x{new_height} from {width}x{height}")

        return image

    def _prepare_image_for_saving(self, image: Image.Image) -> Image.Image:
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image, mask=image.split()[1])
            return background
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process image data and save to cached path."""
        image = None
        resize = kwargs.get("resize", True)
        try:
            with Image.open(BytesIO(data), mode='r') as image:
                if resize:
                    image = self._resize_image_keep_aspect_ratio(image)
                image = self._prepare_image_for_saving(image)
                image.save(cached_path, 'PNG', quality=100, optimize=True)
            
            self._cleanup_old_files()
            return cached_path
        finally:
            # Ensure image object is closed to free memory
            if image:
                try:
                    image.close()
                except:
                    pass

    async def process_image_url(self, image_url: str, resize: bool = True) -> str:
        """Process a single image URL and return path to cached file."""
        return await self._process_single_media(image_url, resize=resize)

    async def process_image_urls(self, image_urls: List[str], resize: bool = True) -> List[str]:
        """Process multiple image URLs and return paths to cached files."""
        tasks = [self.process_image_url(url, resize=resize) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results