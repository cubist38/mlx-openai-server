"""Handler subprocess wrapper for MLX model isolation.

Spawns each model handler in a dedicated subprocess using the ``spawn``
start method to prevent MLX Metal/GPU semaphore leaks.

References
----------
- https://github.com/ml-explore/mlx/issues/2457
- https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

Architecture
------------
::

    Main Process (FastAPI)              Child Process (Handler)
    ┌──────────────────────┐           ┌──────────────────────┐
    │  HandlerProcessProxy │           │  _handler_worker()   │
    │  ├─ request_queue ───┼──────────>│  ├─ handler          │
    │  ├─ response_queue <─┼──────────<│  ├─ model            │
    │  ├─ Process          │           │  └─ inference_worker  │
    │  │                   │           │                      │
    │  ├─ generate_*()     │           │                      │
    │  ├─ get_models()     │           │                      │
    │  └─ cleanup()        │           │                      │
    └──────────────────────┘           └──────────────────────┘

Each handler subprocess owns its MLX model exclusively, ensuring that
the Metal runtime is never shared across process boundaries (which causes
the ``resource_tracker`` semaphore leak warning on macOS).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, AsyncGenerator

from loguru import logger

# ---------------------------------------------------------------------------
# IPC protocol constants
# ---------------------------------------------------------------------------

_SHUTDOWN = "__SHUTDOWN__"
_STREAM_END = "__STREAM_END__"
_CANCEL = "__CANCEL__"


# ---------------------------------------------------------------------------
# Child process entry point
# ---------------------------------------------------------------------------


def _handler_worker(
    model_cfg_dict: dict[str, Any],
    queue_config: dict[str, Any],
    request_queue: mp.Queue,  # type: ignore[type-arg]
    response_queue: mp.Queue,  # type: ignore[type-arg]
    control_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Entry point for the spawned handler subprocess.

    Creates a handler from the serialized config, initializes it with
    the inference worker, then enters a blocking request loop that
    dispatches method calls received from the parent process.

    The child process ignores ``SIGINT`` so that ``Ctrl+C`` is handled
    exclusively by the parent process.  The parent orchestrates an
    orderly shutdown by sending a ``_SHUTDOWN`` message through the
    request queue, which allows the handler to clean up resources
    (e.g. GPU memory, temp files) before the process exits.

    Parameters
    ----------
    model_cfg_dict : dict[str, Any]
        Serialized ``ModelEntryConfig`` fields (plain dict for pickling).
    queue_config : dict[str, Any]
        Configuration for the handler's ``InferenceWorker``
        (``queue_size``, ``timeout``).
    request_queue : mp.Queue
        Queue for receiving requests from the main process.
    response_queue : mp.Queue
        Queue for sending responses back to the main process.
    control_queue : mp.Queue
        Queue for cancel signals from the main process (client disconnect).
    """
    # ------------------------------------------------------------------
    # Ignore SIGINT — the parent manages our lifecycle via _SHUTDOWN.
    # Without this, Ctrl+C sends SIGINT to every process in the group,
    # killing children before the parent can perform an orderly shutdown.
    # ------------------------------------------------------------------
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import asyncio
    import gc

    import mlx.core as mx
    from loguru import logger

    from app.config import ModelEntryConfig
    from app.server import create_handler_from_config

    # Remember the parent PID so the request loop can detect if the
    # parent dies unexpectedly (e.g. SIGKILL).  Because we use the
    # ``spawn`` start method, ``os.getppid()`` returns the PID of the
    # process that called ``Process.start()``.
    _parent_pid = os.getppid()

    # Request IDs cancelled by the parent (e.g. client disconnect).
    # A dedicated thread drains control_queue and adds ids here so the
    # request loop can stop forwarding streaming chunks.
    _cancelled_ids: set[str] = set()
    _cancelled_lock = threading.Lock()

    def _control_reader() -> None:
        while True:
            try:
                msg = control_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                break
            req_id = msg.get("id", "") if isinstance(msg, dict) else msg
            if req_id:
                with _cancelled_lock:
                    _cancelled_ids.add(req_id)

    _control_thread = threading.Thread(
        target=_control_reader, daemon=True, name="control-reader"
    )
    _control_thread.start()

    async def _main() -> None:
        model_cfg = ModelEntryConfig(**model_cfg_dict)
        model_id = model_cfg.model_id

        # ------------------------------------------------------------------
        # Handler creation & initialization
        # ------------------------------------------------------------------
        try:
            handler = create_handler_from_config(model_cfg)
            await handler.initialize(queue_config)
            response_queue.put({"type": "ready", "success": True})
            logger.info(f"Handler process ready for model '{model_id}'")
        except Exception as exc:
            response_queue.put(
                {
                    "type": "ready",
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return

        # ------------------------------------------------------------------
        # Request loop
        # ------------------------------------------------------------------
        while True:
            try:
                request = request_queue.get(timeout=1.0)
            except queue.Empty:
                # Detect orphaned child: if the parent died the queue will
                # never receive a _SHUTDOWN, so we exit proactively.
                try:
                    os.kill(_parent_pid, 0)
                except (ProcessLookupError, PermissionError):
                    logger.warning(
                        f"Parent process (pid={_parent_pid}) exited; "
                        f"handler subprocess for '{model_id}' shutting down"
                    )
                    break
                continue
            except Exception:
                break

            req_id: str = request.get("id", "")
            method_name: str = request.get("method", "")

            # Shutdown signal
            if method_name == _SHUTDOWN:
                try:
                    await handler.cleanup()
                except Exception as exc:
                    logger.error(
                        f"Error during handler cleanup in subprocess: {exc}"
                    )
                response_queue.put(
                    {"id": req_id, "type": "shutdown_complete"}
                )
                break

            args: tuple[Any, ...] = request.get("args", ())
            kwargs: dict[str, Any] = request.get("kwargs", {})
            is_stream: bool = request.get("stream", False)

            try:
                method = getattr(handler, method_name)

                if is_stream:
                    cancelled_early = False
                    async for chunk in method(*args, **kwargs):
                        with _cancelled_lock:
                            if req_id in _cancelled_ids:
                                cancelled_early = True
                                break
                        response_queue.put(
                            {"id": req_id, "type": "chunk", "value": chunk}
                        )
                    response_queue.put({"id": req_id, "type": _STREAM_END})
                    with _cancelled_lock:
                        _cancelled_ids.discard(req_id)
                else:
                    result = await method(*args, **kwargs)
                    response_queue.put(
                        {"id": req_id, "type": "result", "value": result}
                    )
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    f"Error handling request {req_id} "
                    f"(method={method_name}): {exc}\n{tb}"
                )
                response_queue.put(
                    {
                        "id": req_id,
                        "type": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "status_code": getattr(exc, "status_code", 500),
                        "detail": getattr(exc, "detail", None),
                    }
                )

            gc.collect()
            mx.clear_cache()

        # Final cleanup
        gc.collect()
        logger.info(f"Handler subprocess for model '{model_id}' exiting")

    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Main-process proxy
# ---------------------------------------------------------------------------


class HandlerProcessProxy:
    """Proxy that forwards handler method calls to a spawned subprocess.

    Exposes the same public interface as the concrete handler classes
    (``MLXLMHandler``, ``MLXVLMHandler``, etc.) so it can be used as a
    drop-in replacement in the ``ModelRegistry`` and API endpoints.

    A dedicated reader thread continuously drains the response queue and
    routes responses to the appropriate in-flight caller via per-request
    ``asyncio.Queue`` instances.

    Attributes
    ----------
    model_path : str
        Path to the model (used for display / API responses).
    model_id : str
        Unique model identifier in the registry.
    handler_type : str
        Handler type string (``"lm"``, ``"multimodal"``, ``"embeddings"``,
        ``"image"``, ``"whisper"``).
    model_created : int
        Unix timestamp when the handler process was started.
    """

    # Maps model_type config values to handler_type strings
    _MODEL_TYPE_TO_HANDLER_TYPE: dict[str, str] = {
        "lm": "lm",
        "multimodal": "multimodal",
        "embeddings": "embeddings",
        "image-generation": "image",
        "image-edit": "image",
        "whisper": "whisper",
    }

    def __init__(
        self,
        model_cfg_dict: dict[str, Any],
        model_type: str,
        model_path: str,
        model_id: str,
    ) -> None:
        """Initialize the handler process proxy.

        Parameters
        ----------
        model_cfg_dict : dict[str, Any]
            Serialized ``ModelEntryConfig`` fields.
        model_type : str
            Model type from config (``"lm"``, ``"multimodal"``, etc.).
        model_path : str
            Path to the model.
        model_id : str
            Unique identifier for the model.
        """
        self.model_path = model_path
        self.model_id = model_id
        self.handler_type = self._MODEL_TYPE_TO_HANDLER_TYPE.get(
            model_type, model_type
        )
        self.model_created: int = 0

        self._model_cfg_dict = model_cfg_dict

        # Use the ``spawn`` start method for clean Metal runtime isolation.
        self._ctx = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        self._response_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        self._control_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        self._process: mp.Process | None = None

        # Response routing: maps request IDs → per-caller asyncio queues.
        self._pending: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # RPC timeouts and streaming backpressure (set in start() from queue_config).
        self._rpc_timeout: float = 600.0
        self._stream_queue_size: int = 64

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, queue_config: dict[str, Any]) -> None:
        """Spawn the handler subprocess and wait for it to become ready.

        Parameters
        ----------
        queue_config : dict[str, Any]
            Configuration forwarded to the handler's
            ``InferenceWorker`` (``queue_size``, ``timeout``).

        Raises
        ------
        RuntimeError
            If the child process fails to initialize within 300 s.
        """
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._rpc_timeout = float(
            queue_config.get("timeout", 300)
        )
        self._stream_queue_size = int(
            queue_config.get("stream_queue_size", 64)
        )

        # Start the response reader thread.
        self._reader_thread = threading.Thread(
            target=self._response_reader,
            daemon=True,
            name=f"proxy-reader-{self.model_id}",
        )
        self._reader_thread.start()

        # Spawn the child process.
        self._process = self._ctx.Process(
            target=_handler_worker,
            args=(
                self._model_cfg_dict,
                queue_config,
                self._request_queue,
                self._response_queue,
                self._control_queue,
            ),
            name=f"handler-{self.model_id}",
        )
        self._process.start()
        logger.info(
            f"Spawned handler process for '{self.model_id}' "
            f"(pid={self._process.pid})"
        )

        # Wait for the ready signal.
        ready_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending["__ready__"] = ready_queue

        try:
            response = await asyncio.wait_for(ready_queue.get(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Handler process for '{self.model_id}' "
                "did not become ready within 300 s"
            )
        finally:
            self._pending.pop("__ready__", None)

        if not response.get("success"):
            error_msg = response.get("error", "unknown error")
            raise RuntimeError(
                f"Handler process for '{self.model_id}' "
                f"failed to initialize: {error_msg}"
            )

        self.model_created = int(time.time())
        logger.info(f"Handler process for '{self.model_id}' is ready")

    def _response_reader(self) -> None:
        """Dedicated thread that reads from the response queue.

        Routes each response to the appropriate pending caller's
        ``asyncio.Queue`` via ``loop.call_soon_threadsafe``.
        """
        while self._running:
            try:
                response = self._response_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                if self._running:
                    logger.error(
                        "Error reading from handler response queue",
                        exc_info=True,
                    )
                break

            # Special case: ready signal during start().
            if response.get("type") == "ready":
                pending = self._pending.get("__ready__")
                if pending and self._loop:
                    self._loop.call_soon_threadsafe(
                        pending.put_nowait, response
                    )
                continue

            req_id = response.get("id", "")
            pending = self._pending.get(req_id)
            if pending and self._loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        pending.put(response), self._loop
                    )
                    future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        f"Timeout delivering stream chunk for {req_id}"
                    )
                except Exception:
                    if self._running:
                        logger.debug(
                            f"Failed to deliver chunk for {req_id}",
                            exc_info=True,
                        )

    # ------------------------------------------------------------------
    # Generic RPC helpers
    # ------------------------------------------------------------------

    async def _call(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Send a non-streaming RPC call to the child and return the result.

        Parameters
        ----------
        method_name : str
            Name of the handler method to invoke.
        *args : Any
            Positional arguments forwarded to the method.
        **kwargs : Any
            Keyword arguments forwarded to the method.

        Returns
        -------
        Any
            The return value from the remote handler method.

        Raises
        ------
        fastapi.HTTPException
            When the child reports an error.
        """
        req_id = str(uuid.uuid4())
        result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending[req_id] = result_queue

        try:
            await asyncio.to_thread(
                self._request_queue.put,
                {
                    "id": req_id,
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                    "stream": False,
                },
            )

            response = await asyncio.wait_for(
                result_queue.get(), timeout=self._rpc_timeout
            )

            if response["type"] == "error":
                self._raise_remote_error(response)

            return response["value"]
        finally:
            self._pending.pop(req_id, None)

    async def _call_stream(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming RPC call and yield chunks from the child.

        Parameters
        ----------
        method_name : str
            Name of the handler method that returns an async generator.
        *args : Any
            Positional arguments forwarded to the method.
        **kwargs : Any
            Keyword arguments forwarded to the method.

        Yields
        ------
        Any
            Chunks produced by the remote handler method.

        Raises
        ------
        fastapi.HTTPException
            When the child reports an error.
        """
        req_id = str(uuid.uuid4())
        result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._stream_queue_size
        )
        self._pending[req_id] = result_queue

        try:
            await asyncio.to_thread(
                self._request_queue.put,
                {
                    "id": req_id,
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                    "stream": True,
                },
            )

            while True:
                response = await asyncio.wait_for(
                    result_queue.get(), timeout=self._rpc_timeout
                )

                if response["type"] == _STREAM_END:
                    break
                if response["type"] == "error":
                    self._raise_remote_error(response)

                yield response["value"]
        finally:
            self._pending.pop(req_id, None)
            # Signal child to stop forwarding chunks (e.g. client disconnect).
            try:
                self._control_queue.put({"id": req_id, "method": _CANCEL})
            except (BrokenPipeError, EOFError, OSError):
                pass

    @staticmethod
    def _raise_remote_error(response: dict[str, Any]) -> None:
        """Reconstruct and raise an error received from the child process.

        Parameters
        ----------
        response : dict[str, Any]
            Error response dict from the child process.

        Raises
        ------
        fastapi.HTTPException
            Always raised with status code and detail from the response.
        """
        from fastapi import HTTPException

        status_code = response.get("status_code", 500)
        detail = response.get("detail") or response.get(
            "message", "Unknown error in handler subprocess"
        )
        raise HTTPException(status_code=status_code, detail=detail)

    # ------------------------------------------------------------------
    # File pre-processing helpers (for non-picklable UploadFile args)
    # ------------------------------------------------------------------

    @staticmethod
    async def _save_upload_file(
        upload_file: Any, suffix: str = ".bin"
    ) -> str:
        """Save an ``UploadFile`` to a temporary file and return the path.

        Parameters
        ----------
        upload_file : UploadFile
            FastAPI upload file object.
        suffix : str
            File extension for the temporary file.

        Returns
        -------
        str
            Filesystem path to the saved temporary file.
        """
        content = await upload_file.read()
        filename = getattr(upload_file, "filename", None)
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(content)
            return tmp.name

    # ------------------------------------------------------------------
    # Public handler interface (forwarded to child process)
    # ------------------------------------------------------------------

    async def initialize(
        self, queue_config: dict[str, Any] | None = None
    ) -> None:
        """No-op — initialization is handled by ``start()``.

        Parameters
        ----------
        queue_config : dict[str, Any] | None
            Ignored. Kept for interface compatibility.
        """

    async def get_models(self) -> list[dict[str, Any]]:
        """Get list of available models from the subprocess handler.

        Returns
        -------
        list[dict[str, Any]]
            Model metadata list.
        """
        return await self._call("get_models")

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get inference worker statistics from the subprocess handler.

        Returns
        -------
        dict[str, Any]
            Worker and queue statistics.
        """
        return await self._call("get_queue_stats")

    # -- LM handler methods --

    async def generate_text_stream(
        self, request: Any
    ) -> AsyncGenerator[Any, None]:
        """Forward a streaming text generation request to the subprocess.

        Parameters
        ----------
        request : ChatCompletionRequest
            The chat completion request.

        Yields
        ------
        Any
            Text generation chunks (str, dict, or usage info).
        """
        async for chunk in self._call_stream(
            "generate_text_stream", request
        ):
            yield chunk

    async def generate_text_response(self, request: Any) -> dict[str, Any]:
        """Forward a non-streaming text generation request to the subprocess.

        Parameters
        ----------
        request : ChatCompletionRequest
            The chat completion request.

        Returns
        -------
        dict[str, Any]
            Response dict with ``"response"`` and ``"usage"`` keys.
        """
        return await self._call("generate_text_response", request)

    # -- VLM handler methods --

    async def generate_multimodal_stream(
        self, request: Any
    ) -> AsyncGenerator[Any, None]:
        """Forward a streaming multimodal generation request to the subprocess.

        Parameters
        ----------
        request : ChatCompletionRequest
            The multimodal chat completion request.

        Yields
        ------
        Any
            Multimodal generation chunks.
        """
        async for chunk in self._call_stream(
            "generate_multimodal_stream", request
        ):
            yield chunk

    async def generate_multimodal_response(
        self, request: Any
    ) -> dict[str, Any]:
        """Forward a non-streaming multimodal generation request to the subprocess.

        Parameters
        ----------
        request : ChatCompletionRequest
            The multimodal chat completion request.

        Returns
        -------
        dict[str, Any]
            Response dict with ``"response"`` and ``"usage"`` keys.
        """
        return await self._call("generate_multimodal_response", request)

    # -- Embeddings handler methods --

    async def generate_embeddings_response(self, request: Any) -> Any:
        """Forward an embeddings generation request to the subprocess.

        Parameters
        ----------
        request : EmbeddingRequest
            The embedding request.

        Returns
        -------
        Any
            Embeddings result (list of lists of floats).
        """
        return await self._call("generate_embeddings_response", request)

    # -- Image generation handler methods --

    async def generate_image(self, request: Any) -> Any:
        """Forward an image generation request to the subprocess.

        Parameters
        ----------
        request : ImageGenerationRequest
            The image generation request.

        Returns
        -------
        ImageGenerationResponse
            The generated image response.
        """
        return await self._call("generate_image", request)

    async def edit_image(self, request: Any) -> Any:
        """Forward an image editing request to the subprocess.

        For ``ImageEditRequest`` objects that contain ``UploadFile``
        fields (which are not picklable), this method pre-processes
        the files in the main process and sends file paths to the
        child process via the ``edit_image_from_paths`` handler method.

        Parameters
        ----------
        request : ImageEditRequest
            The image editing request.

        Returns
        -------
        ImageEditResponse
            The edited image response.
        """
        # ImageEditRequest.image contains UploadFile(s) — not picklable.
        # Save files locally and forward paths to the subprocess.
        images = (
            request.image
            if isinstance(request.image, list)
            else [request.image]
        )

        temp_paths: list[str] = []
        for img in images:
            path = await self._save_upload_file(img, suffix=".png")
            temp_paths.append(path)

        edit_data = {
            "image_paths": temp_paths,
            "prompt": request.prompt,
            "negative_prompt": getattr(request, "negative_prompt", None),
            "steps": getattr(request, "steps", None),
            "seed": getattr(request, "seed", None),
            "guidance_scale": getattr(request, "guidance_scale", None),
        }

        try:
            return await self._call("edit_image_from_paths", edit_data)
        finally:
            # Clean up temp files regardless of success/failure
            for path in temp_paths:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass

    # -- Whisper handler methods --

    async def prepare_transcription_request(
        self, request: Any
    ) -> dict[str, Any]:
        """Pre-process a transcription request by saving the uploaded file.

        Since ``UploadFile`` objects are not picklable, file I/O is
        performed in the main process and the resulting dict (with a
        filesystem path) is returned for subsequent calls.

        Parameters
        ----------
        request : TranscriptionRequest
            The transcription request containing an uploaded audio file.

        Returns
        -------
        dict[str, Any]
            Pre-processed request data with ``audio_path`` instead of
            a file object.
        """
        file = request.file
        audio_path = await self._save_upload_file(file, suffix=".wav")

        request_data: dict[str, Any] = {
            "audio_path": audio_path,
            "verbose": False,
        }
        if request.temperature is not None:
            request_data["temperature"] = request.temperature
        if request.language is not None:
            request_data["language"] = request.language
        if request.prompt is not None:
            request_data["initial_prompt"] = request.prompt

        return request_data

    async def generate_transcription_response(self, request: Any) -> Any:
        """Forward a transcription request to the subprocess.

        The uploaded file is saved locally first, then the
        pre-processed data is sent to the child process for inference.

        Parameters
        ----------
        request : TranscriptionRequest
            The transcription request.

        Returns
        -------
        TranscriptionResponse
            The transcription result.
        """
        request_data = await self.prepare_transcription_request(request)
        return await self._call(
            "transcribe_from_data", request_data
        )

    async def generate_transcription_stream_from_data(
        self, request_data: dict[str, Any], *args: Any, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Forward a streaming transcription request to the subprocess.

        Parameters
        ----------
        request_data : dict[str, Any]
            Pre-processed transcription request data (from
            ``prepare_transcription_request``).
        *args : Any
            Additional positional arguments forwarded to the handler.
        **kwargs : Any
            Additional keyword arguments forwarded to the handler.

        Yields
        ------
        str
            SSE-formatted transcription chunks.
        """
        async for chunk in self._call_stream(
            "transcribe_stream_from_data", request_data, *args, **kwargs
        ):
            yield chunk

    # -- Cleanup --

    async def cleanup(self) -> None:
        """Send shutdown signal to the child process and clean up.

        Waits for the child to acknowledge shutdown, then joins the
        process and reader thread.  If the child does not respond
        within 10 s it is forcefully terminated.

        Blocking ``Process.join`` calls are wrapped in
        ``asyncio.to_thread`` so that multiple proxies can be cleaned
        up concurrently via ``asyncio.gather`` without blocking the
        event loop.
        """
        if not self._process or not self._process.is_alive():
            self._running = False
            return

        # -- Phase 1: Request a graceful shutdown via the IPC queue. --
        req_id = str(uuid.uuid4())
        shutdown_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending[req_id] = shutdown_queue
        graceful = False

        try:
            # Attempt to enqueue the shutdown command.  The put itself
            # can fail if the child has already exited and the
            # underlying pipe is broken.
            try:
                await asyncio.to_thread(
                    self._request_queue.put,
                    {"id": req_id, "method": _SHUTDOWN},
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                logger.warning(
                    f"Could not send shutdown to '{self.model_id}': {exc}"
                )
            else:
                # Wait for the child to acknowledge the shutdown.
                try:
                    await asyncio.wait_for(
                        shutdown_queue.get(), timeout=10
                    )
                    graceful = True
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Handler process for '{self.model_id}' did not "
                        "acknowledge shutdown within 10 s; terminating"
                    )
        finally:
            self._pending.pop(req_id, None)

        self._running = False

        # -- Phase 2: Ensure the child process exits. --
        if not graceful and self._process.is_alive():
            self._process.terminate()

        try:
            await asyncio.to_thread(self._process.join, 5)
        except (OSError, ValueError):
            pass  # Process handle already closed / invalid.

        if self._process.is_alive():
            logger.warning(
                f"Force-killing handler process for '{self.model_id}'"
            )
            try:
                self._process.kill()
            except (OSError, ProcessLookupError):
                pass  # Already dead.
            try:
                await asyncio.to_thread(self._process.join, 3)
            except (OSError, ValueError):
                pass

        # -- Phase 3: Stop the response reader thread. --
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)

        logger.info(
            f"Handler process for '{self.model_id}' shut down successfully"
        )
