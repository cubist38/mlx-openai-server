"""Base processor classes for media handling with caching and session management."""

from abc import ABC, abstractmethod
import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import os
import tempfile
import time
from typing import Any

import aiohttp
from loguru import logger


class BaseProcessor(ABC):
    """Base class for media processors with common caching and session management."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        # Use tempfile for macOS-efficient temporary file handling
        self.temp_dir = tempfile.TemporaryDirectory()
        self._session: aiohttp.ClientSession | None = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_size = cache_size
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        # Replace lru_cache with manual cache for better control
        self._hash_cache: dict[str, str] = {}
        self._cache_access_times: dict[str, float] = {}

    def _get_media_hash(self, media_url: str) -> str:
        """Get hash for media URL with manual caching that can be cleared."""
        # Check if already cached
        if media_url in self._hash_cache:
            self._cache_access_times[media_url] = time.time()
            return self._hash_cache[media_url]

        # Generate hash
        if media_url.startswith("data:"):
            _, encoded = media_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            data = media_url.encode("utf-8")

        hash_value = hashlib.md5(data).hexdigest()

        # Add to cache with size management
        if len(self._hash_cache) >= self._cache_size:
            self._evict_oldest_cache_entries()

        self._hash_cache[media_url] = hash_value
        self._cache_access_times[media_url] = time.time()
        return hash_value

    def _evict_oldest_cache_entries(self):
        """Remove oldest 20% of cache entries to make room."""
        if not self._cache_access_times:
            return

        # Sort by access time and remove oldest 20%
        sorted_items = sorted(self._cache_access_times.items(), key=lambda x: x[1])
        to_remove = len(sorted_items) // 5  # Remove 20%

        for url, _ in sorted_items[:to_remove]:
            self._hash_cache.pop(url, None)
            self._cache_access_times.pop(url, None)

        # Force garbage collection after cache eviction
        gc.collect()

    @abstractmethod
    def _get_media_format(self, media_url: str, data: bytes | None = None) -> str:
        """Determine media format from URL or data. Must be implemented by subclasses."""

    @abstractmethod
    def _validate_media_data(self, data: bytes) -> bool:
        """Validate media data. Must be implemented by subclasses."""

    @abstractmethod
    def _get_timeout(self) -> int:
        """Get timeout for HTTP requests. Must be implemented by subclasses."""

    @abstractmethod
    def _get_max_file_size(self) -> int:
        """Get maximum file size in bytes. Must be implemented by subclasses."""

    @abstractmethod
    def _process_media_data(self, data: bytes, cached_path: str, **kwargs) -> dict[str, Any]:
        """Process media data and save to cached path. Must be implemented by subclasses."""

    @abstractmethod
    def _get_media_type_name(self) -> str:
        """Get media type name for logging. Must be implemented by subclasses."""

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._get_timeout()),
                headers={"User-Agent": "mlx-server-OAI-compat/1.0"},
            )
        return self._session

    def _cleanup_old_files(self):
        """Clean up old temporary files and cache entries periodically.

        This method removes temporary files older than the cleanup interval
        and evicts cache entries if the cache is near capacity. It runs
        periodically based on the _cleanup_interval setting.
        """
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                for file in os.listdir(self.temp_dir.name):
                    file_path = os.path.join(self.temp_dir.name, file)
                    if os.path.getmtime(file_path) < current_time - self._cleanup_interval:
                        os.remove(file_path)
                self._last_cleanup = current_time
                # Also clean up cache periodically
                if len(self._hash_cache) > self._cache_size * 0.8:
                    self._evict_oldest_cache_entries()
                gc.collect()  # Force garbage collection after cleanup
            except Exception as e:
                logger.warning(f"Failed to clean up old {self._get_media_type_name()} files: {e!s}")

    async def _process_single_media(self, media_url: str, **kwargs) -> str:
        """Process a single media item from URL or local path.

        This method handles downloading, validating, and processing media
        from various sources (local files, HTTP URLs, base64 data URIs).
        It uses caching to avoid reprocessing the same media.

        Parameters
        ----------
        media_url : str
            The URL or path to the media. Can be a local file path,
            HTTP/HTTPS URL, or base64 data URI.
        **kwargs
            Additional keyword arguments passed to the processing method.

        Returns
        -------
        str
            Path to the processed and cached media file.

        Raises
        ------
        ValueError
            If the media data is invalid or processing fails.
        """
        try:
            media_hash = self._get_media_hash(media_url)
            media_format = self._get_media_format(media_url)
            cached_path = os.path.join(self.temp_dir.name, f"{media_hash}.{media_format}")

            if os.path.exists(cached_path):
                logger.debug(f"Using cached {self._get_media_type_name()}: {cached_path}")
                return cached_path

            if os.path.exists(media_url):
                # Copy local file to cache
                with open(media_url, "rb") as f:
                    data = f.read()

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                return self._process_media_data(data, cached_path, **kwargs)

            if media_url.startswith("data:"):
                _, encoded = media_url.split(",", 1)
                estimated_size = len(encoded) * 3 / 4
                if estimated_size > self._get_max_file_size():
                    raise ValueError(
                        f"Base64-encoded {self._get_media_type_name()} exceeds size limit"
                    )
                data = base64.b64decode(encoded)

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                return self._process_media_data(data, cached_path, **kwargs)
            session = await self._get_session()
            async with session.get(media_url) as response:
                response.raise_for_status()
                data = await response.read()

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                return self._process_media_data(data, cached_path, **kwargs)

        except Exception as e:
            logger.error(f"Failed to process {self._get_media_type_name()}: {e!s}")
            raise ValueError(f"Failed to process {self._get_media_type_name()}: {e!s}") from e
        finally:
            gc.collect()

    def clear_cache(self):
        """Manually clear the hash cache to free memory."""
        self._hash_cache.clear()
        self._cache_access_times.clear()
        gc.collect()

    async def cleanup(self):
        """Clean up resources used by the processor.

        This method closes the HTTP session, shuts down the thread executor,
        and cleans up the temporary directory. It ensures proper resource
        cleanup and prevents resource leaks.
        """
        if hasattr(self, "_cleaned") and self._cleaned:
            return
        self._cleaned = True
        try:
            # Clear caches before cleanup
            self.clear_cache()

            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning(f"Exception closing aiohttp session: {e!s}")
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Exception shutting down executor: {e!s}")
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Exception cleaning up temp directory: {e!s}")

    async def __aenter__(self):
        """Enter the async context manager.

        Returns
        -------
        BaseProcessor
            The processor instance.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the async context manager and clean up resources.

        Parameters
        ----------
        exc_type
            The exception type, if any.
        exc
            The exception instance, if any.
        tb
            The traceback, if any.
        """
        await self.cleanup()

    def __del__(self):
        """Destructor - warns about proper cleanup.

        Async cleanup cannot be reliably performed in __del__.
        Please use 'async with Processor()' or call 'await cleanup()' explicitly.
        """
