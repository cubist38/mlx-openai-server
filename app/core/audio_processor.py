"""Audio processing utilities for MLX OpenAI server."""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base_processor import BaseProcessor


class AudioProcessor(BaseProcessor):
    """Audio processor for handling audio files with caching and validation."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000) -> None:
        """
        Initialize the AudioProcessor.

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker threads for processing, by default 4.
        cache_size : int, optional
            Maximum number of cached files to keep, by default 1000.
        """
        super().__init__(max_workers, cache_size)
        # Supported audio formats
        self._supported_formats = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

    def _get_media_format(self, media_url: str, _data: bytes | None = None) -> str:
        """
        Determine audio format from URL or data.

        Parameters
        ----------
        media_url : str
            The URL or data URL of the audio file.
        _data : bytes or None, optional
            Audio data bytes, not used in this implementation.

        Returns
        -------
        str
            The audio format (e.g., 'mp3', 'wav').
        """
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
            parsed = urlparse(media_url)
            if parsed.scheme:
                # It's a URL, get the path part
                path = parsed.path
            else:
                path = media_url
            ext = Path(path.lower()).suffix
            if ext in self._supported_formats:
                return ext[1:]  # Remove the dot

        # Default to mp3 if format cannot be determined
        return "mp3"

    def _validate_media_data(self, data: bytes) -> bool:
        """
        Validate basic audio data.

        Parameters
        ----------
        data : bytes
            The audio data to validate.

        Returns
        -------
        bool
            True if the data appears to be valid audio, False otherwise.
        """
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
        """
        Get timeout for HTTP requests.

        Returns
        -------
        int
            Timeout in seconds for audio file downloads.
        """
        return 60  # Longer timeout for audio files

    def _get_max_file_size(self) -> int:
        """
        Get maximum file size in bytes.

        Returns
        -------
        int
            Maximum allowed file size for audio files in bytes.
        """
        return 500 * 1024 * 1024  # 500 MB limit for audio

    def _process_media_data(self, data: bytes, cached_path: str, **_kwargs: Any) -> str:
        """
        Process audio data and save to cached path.

        Parameters
        ----------
        data : bytes
            The audio data to process.
        cached_path : str
            Path where the processed audio should be saved.
        **_kwargs : Any
            Additional keyword arguments (unused).

        Returns
        -------
        str
            The path to the cached audio file.
        """
        with Path(cached_path).open("wb") as f:
            f.write(data)
        self._cleanup_old_files()
        return cached_path

    def _get_media_type_name(self) -> str:
        """
        Get media type name for logging.

        Returns
        -------
        str
            The string 'audio' for logging purposes.
        """
        return "audio"

    async def process_audio_url(self, audio_url: str) -> str:
        """
        Process a single audio URL and return path to cached file.

        Parameters
        ----------
        audio_url : str
            The URL of the audio file to process.

        Returns
        -------
        str
            Path to the cached audio file.
        """
        return await self._process_single_media(audio_url)

    async def process_audio_urls(self, audio_urls: list[str]) -> list[str | BaseException]:
        """
        Process multiple audio URLs and return a list containing either file path strings or BaseException instances for failed items.

        Parameters
        ----------
        audio_urls : list[str]
            List of audio URLs to process.

        Returns
        -------
        list[str | BaseException]
            List where each element is either a path to a cached audio file (str) or a BaseException for failed processing.
        """
        tasks = [self.process_audio_url(url) for url in audio_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results
