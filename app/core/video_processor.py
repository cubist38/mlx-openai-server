import os
import gc
import asyncio
from loguru import logger
from typing import List
from .base_processor import BaseProcessor


class VideoProcessor(BaseProcessor):
    """Video processor for handling video files with caching, validation, and processing."""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        super().__init__(max_workers, cache_size)
        # Supported video formats
        self._supported_formats = {'.mp4', '.avi', '.mov'}

    def _get_media_format(self, media_url: str, data: bytes = None) -> str:
        """Determine video format from URL or data."""
        if media_url.startswith("data:"):
            # Extract format from data URL
            mime_type = media_url.split(";")[0].split(":")[1]
            if "mp4" in mime_type:
                return "mp4"
            elif "quicktime" in mime_type or "mov" in mime_type:
                return "mov"
            elif "x-msvideo" in mime_type or "avi" in mime_type:
                return "avi"
        else:
            # Extract format from file extension
            ext = os.path.splitext(media_url.lower())[1]
            if ext in self._supported_formats:
                return ext[1:]  # Remove the dot
        
        # Default to mp4 if format cannot be determined
        return "mp4"

    def _validate_media_data(self, data: bytes) -> bool:
        """Basic validation of video data."""
        if len(data) < 100:  # Too small to be a valid video file
            return False
        
        # Check for common video file signatures
        video_signatures = [
            # MP4/M4V/MOV (ISO Base Media File Format)
            (b'\x00\x00\x00\x14ftypisom', 0),  # MP4
            (b'\x00\x00\x00\x18ftyp', 0),       # MP4/MOV
            (b'\x00\x00\x00\x1cftyp', 0),       # MP4/MOV
            (b'\x00\x00\x00\x20ftyp', 0),       # MP4/MOV
            (b'ftyp', 4),                        # MP4/MOV (ftyp at offset 4)
            
            # AVI
            (b'RIFF', 0),  # AVI (also check for 'AVI ' at offset 8)
            
            # WebM/MKV (Matroska)
            (b'\x1a\x45\xdf\xa3', 0),  # Matroska/WebM
            
            # FLV
            (b'FLV\x01', 0),  # Flash Video
            
            # MPEG
            (b'\x00\x00\x01\xba', 0),  # MPEG PS
            (b'\x00\x00\x01\xb3', 0),  # MPEG PS
            
            # QuickTime
            (b'moov', 0),  # QuickTime
            (b'mdat', 0),  # QuickTime
        ]
        
        for sig, offset in video_signatures:
            if len(data) > offset + len(sig):
                if data[offset:offset+len(sig)] == sig:
                    # Additional validation for AVI
                    if sig == b'RIFF' and len(data) > 12:
                        if data[8:12] == b'AVI ':
                            return True
                    elif sig == b'RIFF':
                        continue  # Not AVI, might be WAV
                    else:
                        return True
        
        # Check for ftyp box anywhere in first 32 bytes (MP4/MOV)
        if b'ftyp' in data[:32]:
            return True
        
        # Allow unknown formats to pass through for flexibility
        return True

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 120  # Longer timeout for video files (2 minutes)

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 1024 * 1024 * 1024  # 1 GB limit for videos

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process video data and save to cached path."""
        try:
            with open(cached_path, 'wb') as f:
                f.write(data)
            
            logger.info(f"Saved video to {cached_path} ({len(data)} bytes)")
            self._cleanup_old_files()
            return cached_path
        except Exception as e:
            logger.error(f"Failed to save video data: {str(e)}")
            raise

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "video"

    async def process_video_url(self, video_url: str) -> str:
        """
        Process a single video URL and return path to cached file.
        
        Supports:
        - HTTP/HTTPS URLs (downloads video)
        - Local file paths (copies to cache)
        - Data URLs (base64 encoded videos)
        
        Args:
            video_url: URL, file path, or data URL of the video
            
        Returns:
            Path to the cached video file in temp directory
        """
        return await self._process_single_media(video_url)

    async def process_video_urls(self, video_urls: List[str]) -> List[str]:
        """
        Process multiple video URLs and return paths to cached files.
        
        Args:
            video_urls: List of URLs, file paths, or data URLs of videos
            
        Returns:
            List of paths to cached video files
        """
        tasks = [self.process_video_url(url) for url in video_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results
