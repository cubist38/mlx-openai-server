"""Audio processing utilities for MLX OpenAI server."""

import asyncio
import gc
from pathlib import Path

from .base_processor import BaseProcessor


class AudioProcessor(BaseProcessor):
    """Audio processor for handling audio files with caching and validation."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000) -> None:
        super().__init__(max_workers, cache_size)
        # Supported audio formats
        self._supported_formats = {".mp3", ".wav"}

    def _get_media_format(self, media_url: str, data: bytes | None = None) -> str:
        """Determine audio format from URL or data."""
        if media_url.startswith("data:"):
            # Extract format from data URL
            mime_type = media_url.split(";")[0].split(":")[1]
            if "mp3" in mime_type or "mpeg" in mime_type:
                return "mp3"
            if "wav" in mime_type:
                return "wav"
            if "m4a" in mime_type or "mp4" in mime_type:
                return "m4a"
            if "ogg" in mime_type:
                return "ogg"
            if "flac" in mime_type:
                return "flac"
            if "aac" in mime_type:
                return "aac"
        else:
            # Extract format from file extension
            ext = Path(media_url.lower()).suffix
            if ext in self._supported_formats:
                return ext[1:]  # Remove the dot

        # Default to mp3 if format cannot be determined
        return "mp3"

    def _validate_media_data(self, data: bytes) -> bool:
        """Validate basic audio data."""
        if len(data) < 100:  # Too small to be a valid audio file
            return False

        # Check for common audio file signatures
        audio_signatures = [
            b"ID3",  # MP3 with ID3 tag
            b"\xff\xfb",  # MP3 frame header
            b"\xff\xf3",  # MP3 frame header
            b"\xff\xf2",  # MP3 frame header
            b"RIFF",  # WAV/AVI
            b"OggS",  # OGG
            b"fLaC",  # FLAC
            b"\x00\x00\x00\x20ftypM4A",  # M4A
        ]

        for sig in audio_signatures:
            if data.startswith(sig):
                return True

        # Check for WAV format (RIFF header might be at different position)
        if b"WAVE" in data[:50]:
            return True

        return True  # Allow unknown formats to pass through

    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests."""
        return 60  # Longer timeout for audio files

    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        return 500 * 1024 * 1024  # 500 MB limit for audio

    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> str:
        """Process audio data and save to cached path."""
        with Path(cached_path).open("wb") as f:
            f.write(data)
        self._cleanup_old_files()
        return cached_path

    def _get_media_type_name(self) -> str:
        """Get media type name for logging."""
        return "audio"

    async def process_audio_url(self, audio_url: str) -> str:
        """Process a single audio URL and return path to cached file."""
        return await self._process_single_media(audio_url)

    async def process_audio_urls(self, audio_urls: list[str]) -> list[str | Exception]:
        """Process multiple audio URLs and return paths to cached files."""
        tasks = [self.process_audio_url(url) for url in audio_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results
