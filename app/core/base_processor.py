"""Base processor classes for media processing with caching and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
from pathlib import Path
import tempfile
import time
from types import TracebackType
from typing import Any, Self

import aiofiles
import aiohttp
from loguru import logger


class BaseProcessor(ABC):
    """Base class for media processors with common caching and session management."""

    def __init__(self, max_workers: int = 4, cache_size: int = 1000) -> None:
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
        self._cleaned: bool = False

    def _get_media_hash(self, media_url: str) -> str:
        """
        Get hash for media URL with manual caching that can be cleared.

        Parameters
        ----------
        media_url : str
            URL or data URI for the media.

        Returns
        -------
        str
            MD5 hash string for the provided media.
        """
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

    def _evict_oldest_cache_entries(self) -> None:
        """
        Remove oldest 20% of cache entries to make room.

        Returns
        -------
        None
        """
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
        """
        Determine media format from URL or data.

        Parameters
        ----------
        media_url : str
            Media URL or path.
        data : bytes | None, optional
            Optional raw bytes of media for format detection.

        Returns
        -------
        str
            File format or extension (without leading dot).

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """

    @abstractmethod
    def _validate_media_data(self, data: bytes) -> bool:
        """
        Validate media data. Must be implemented by subclasses.

        Parameters
        ----------
        data : bytes
            Raw media bytes to validate.

        Returns
        -------
        bool
            True if data is valid for this media type, False otherwise.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """

    @abstractmethod
    def _get_timeout(self) -> int:
        """
        Get timeout for HTTP requests. Must be implemented by subclasses.

        Returns
        -------
        int
            Timeout in seconds.
        """

    @abstractmethod
    def _get_max_file_size(self) -> int:
        """
        Get maximum file size in bytes. Must be implemented by subclasses.

        Returns
        -------
        int
            Maximum allowed file size in bytes.
        """

    @abstractmethod
    def _process_media_data(self, data: bytes, cached_path: str, **kwargs: Any) -> str:
        """
        Process media data and save to cached path and return the cached file path.

        Parameters
        ----------
        data : bytes
            Raw media bytes to process.
        cached_path : str
            Path where processed file should be saved.
        **kwargs : Any
            Additional processing options.

        Returns
        -------
        str
            Path to the saved cached file.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """

    @abstractmethod
    def _get_media_type_name(self) -> str:
        """
        Get media type name for logging. Must be implemented by subclasses.

        Returns
        -------
        str
            Human-readable media type name (e.g., 'image', 'video').
        """

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._get_timeout()),
                headers={"User-Agent": "mlx-server-OAI-compat/1.0"},
            )
        return self._session

    def _cleanup_old_files(self) -> None:
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                temp_dir_path = Path(self.temp_dir.name)
                for file_path in temp_dir_path.iterdir():
                    if file_path.stat().st_mtime < current_time - self._cleanup_interval:
                        file_path.unlink()
                self._last_cleanup = current_time
                # Also clean up cache periodically
                if len(self._hash_cache) > self._cache_size * 0.8:
                    self._evict_oldest_cache_entries()
                gc.collect()  # Force garbage collection after cleanup
            except Exception as e:
                logger.warning(
                    f"Failed to clean up old {self._get_media_type_name()} files. {type(e).__name__}: {e}"
                )

    async def _process_single_media(self, media_url: str, **kwargs: Any) -> str:
        try:
            media_hash = self._get_media_hash(media_url)
            media_format = self._get_media_format(media_url)
            cached_path = str(Path(self.temp_dir.name) / f"{media_hash}.{media_format}")

            if Path(cached_path).exists():
                logger.debug(f"Using cached {self._get_media_type_name()}: {cached_path}")
                return cached_path

            if Path(media_url).exists():
                # Check file size before opening
                file_size = Path(media_url).stat().st_size
                if file_size > self._get_max_file_size():
                    raise ValueError(
                        f"Local {self._get_media_type_name()} file exceeds size limit: {file_size} > {self._get_max_file_size()}"
                    )
                # Copy local file to cache
                async with aiofiles.open(media_url, "rb") as f:
                    data = await f.read()

                # Validate size after reading (in case file changed)
                if len(data) > self._get_max_file_size():
                    raise ValueError(
                        f"Read {self._get_media_type_name()} data exceeds size limit: {len(data)} > {self._get_max_file_size()}"
                    )

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
                # Check Content-Length if available
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        size = int(content_length)
                        if size > self._get_max_file_size():
                            raise ValueError(
                                f"HTTP {self._get_media_type_name()} Content-Length exceeds size limit: {size} > {self._get_max_file_size()}"
                            )
                    except ValueError:
                        logger.warning(f"Invalid Content-Length header: {content_length}")
                data = await response.read()

                # Validate size after reading
                if len(data) > self._get_max_file_size():
                    raise ValueError(
                        f"Downloaded {self._get_media_type_name()} data exceeds size limit: {len(data)} > {self._get_max_file_size()}"
                    )

                if not self._validate_media_data(data):
                    raise ValueError(f"Invalid {self._get_media_type_name()} file format")

                return self._process_media_data(data, cached_path, **kwargs)

        except Exception as e:
            logger.error(f"Failed to process {self._get_media_type_name()} {type(e).__name__}: {e}")
            raise ValueError(f"Failed to process {self._get_media_type_name()}: {e}") from e
        finally:
            gc.collect()

    def clear_cache(self) -> None:
        """
        Manually clear the hash cache to free memory.

        Returns
        -------
        None
        """
        self._hash_cache.clear()
        self._cache_access_times.clear()
        gc.collect()

    async def cleanup(self) -> None:
        """
        Clean up resources and caches.

        Notes
        -----
        This closes the aiohttp session (if open), shuts down the executor, and
        removes temporary files.

        Returns
        -------
        None
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
            logger.warning(f"Exception closing aiohttp session. {type(e).__name__}: {e}")
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Exception shutting down executor. {type(e).__name__}: {e}")
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Exception cleaning up temp directory. {type(e).__name__}: {e}")

    async def __aenter__(self) -> Self:
        """
        Enter async context manager.

        Returns
        -------
        Self
            The processor instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        Exit async context manager and cleanup.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type if one was raised inside the context manager.
        exc : BaseException | None
            Exception instance if one was raised inside the context manager.
        tb : TracebackType | None
            Traceback for the exception if present.

        Returns
        -------
        None
        """
        await self.cleanup()
