"""
mlx-audio model wrapper

The best audio processing library built on Apple's MLX framework, providing fast and efficient text-to-speech (TTS), speech-to-text (STT), and speech-to-speech (STS) on Apple Silicon.
"""

import os
import io
import time
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional

from mlx_audio.audio_io import write as audio_write
from mlx_audio.utils import load_audio as load_audio_from_file

from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_audio.stt.utils import load_model as load_stt_model


@dataclass
class STTResponse:
    """
    The output of :func:`stt`.

    Args:
        text (str): The text of the transcription.
        language (str): The language of the transcription.
        start (float): The start time of the transcription.
        end (float): The end time of the transcription.
        prompt_tokens (int): The number of tokens in the prompt.
        total_tokens (int): The total number of tokens in the transcription.
        prompt_tps (float): The prompt tokens-per-second.
        generation_tps (float): The generation tokens-per-second.
        total_time (float): The total time of the transcription.
    """

    text: str = None
    language: str = None
    start: float = None
    end: float = None
    prompt_tokens: int = None
    total_tokens: int = None
    prompt_tps: float = None
    generation_tps: float = None
    total_time: float = None
  

class MLX_Audio:
    """
    A wrapper class for MLX Audio that handles memory management to prevent leaks.
    """

    def __init__(self, model_path: str, task: str = "tts"):
        """
        Initialize the MLX_Audio model.
        """
        self.task = task
        if task == "tts":
            self.model = load_tts_model(model_path)
        elif task == "stt":
            self.model = load_stt_model(model_path)
        else:
            raise ValueError(f"Invalid task: {task}")

    def tts(
        self, 
        text: str,
        max_tokens: int = 1200,
        voice: str = "af_heart",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        gender: Optional[str] = "male",
        pitch: float = 1.0,
        lang_code: str = "en",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        verbose: bool = False,
        temperature: float = 0.7,
        stream: bool = False,
        streaming_interval: float = 2.0,
        response_format: str = "mp3",
        **kwargs
    ) -> Generator[bytes, None, None]:
        """
        Generate audio from text using the loaded TTS model.

        Parameters
        ----------
        text : str
            The input text to be converted to speech.
        voice : str
            The voice style to use (also used as speaker for Qwen3-TTS models).
        instruct : str, optional
            Instruction for emotion/style (CustomVoice) or voice description (VoiceDesign).
        speed : float
            Playback speed multiplier.
        gender : str, optional
            The gender of the voice.
        pitch : float
            The pitch of the voice.
        lang_code : str
            The language code.
        ref_audio : str, optional
            Path to reference audio to clone the voice from.
        ref_text : str, optional
            Caption for reference audio.
        temperature : float
            The temperature for the model.
        stream : bool
            If True, yield encoded audio chunks as they are generated; otherwise yield a single final chunk.
        streaming_interval : float
            Interval between streaming chunks in seconds.
        max_tokens : int
            Maximum number of tokens to generate.
        verbose : bool
            Whether to print verbose output.
        response_format : str
            Output audio format (e.g. "mp3", "wav").
        **kwargs
            Additional keyword arguments passed to the model's generate method.

        Yields
        ------
        bytes
            Encoded audio chunks (format determined by response_format).
        """
        audio_chunks = []
        sample_rate = None

        if ref_audio and isinstance(ref_audio, str):
            if not os.path.exists(ref_audio):
                raise ValueError(f"Reference audio file not found: {ref_audio}")

            # Determine if volume normalization is needed
            normalize = hasattr(self.model, "model_type") and self.model.model_type == "spark"

            ref_audio = load_audio_from_file(
                ref_audio, sample_rate=self.model.sample_rate, volume_normalize=normalize
            )

        stream_response = self.model.generate(
            text = text,
            voice = voice,
            speed = speed,
            instruct = instruct,
            lang_code = lang_code,
            ref_audio = ref_audio,
            ref_text = ref_text,
            gender = gender,
            pitch = pitch,
            temperature = temperature,
            max_tokens = max_tokens,
            verbose = verbose,
            stream = True,
            streaming_interval = streaming_interval,
            **kwargs,
        )

        if stream:
            for chunk in stream_response:
                buffer = io.BytesIO()
                audio_write(
                    buffer, chunk.audio, chunk.sample_rate, format=response_format
                )
                yield buffer.getvalue()

        else:
            for chunk in stream_response:
                audio_chunks.append(chunk.audio)
                if sample_rate is None:
                    sample_rate = chunk.sample_rate

            # Ensure numpy for concatenate and audio_write (chunk.audio may be mx.array)
            arrays = [np.asarray(a) for a in audio_chunks]
            concatenated_audio = np.concatenate(arrays)
            buffer = io.BytesIO()
            audio_write(buffer, concatenated_audio, sample_rate, format=response_format)
            yield buffer.getvalue()

    def stt(
        self,
        audio: str,
        verbose: bool = False,
        stream: bool = False,
        format: str = "json",
        **kwargs
    ) -> Generator[bytes, None, None] | STTResponse:
        """
        Transcribe audio file.

        Parameters
        ----------
        audio: str
            The path to the audio file to transcribe.
        verbose: bool
            Whether to print verbose output.
        stream: bool
            Whether to stream the transcription.
        format: str
            The format of the transcription.
        **kwargs
            Additional keyword arguments passed to the model's generate method.

        Yields
        ------
        bytes
            Encoded audio chunks when stream=True.

        Returns
        -------
        Generator[bytes, None, None] | STTResponse
            The raw stream when stream=True, else an STTResponse with transcription and stats.
        """
        stream_response = self.model.generate(
            audio=audio,
            verbose=verbose,
            stream=True,
            format=format,
            **kwargs
        )
        if stream:
            return stream_response

        start_time = time.time()
        text_parts: list[str] = []
        last_final = None

        for chunk in stream_response:
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.is_final:
                last_final = chunk

        total_time = time.time() - start_time
        if last_final is not None:
            prompt_tokens = last_final.prompt_tokens
            generation_tokens = last_final.generation_tokens
            total_tokens = prompt_tokens + generation_tokens
            prompt_tps = prompt_tokens / total_time if total_time > 0 else 0.0
            generation_tps = generation_tokens / total_time if total_time > 0 else 0.0
            language = last_final.language
        else:
            prompt_tokens = 0
            generation_tokens = 0
            total_tokens = 0
            prompt_tps = 0.0
            generation_tps = 0.0
            language = None

        return STTResponse(
            text="".join(text_parts),
            language=language,
            start=0,
            end=total_time,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            total_time=total_time,
        )
    
    def __call__(self, *args, **kwargs):
        if self.task == "tts":
            return self.tts(*args, **kwargs)
        elif self.task == "stt":
            return self.stt(*args, **kwargs)
        else:
            raise ValueError(f"Invalid task: {self.task}")


if __name__ == "__main__":
    model = MLX_Audio("mlx-community/Qwen3-ASR-0.6B-4bit", task = "stt")
    result = model(audio="examples/audios/podcast.wav", stream=True)
    for chunk in result:
        print(chunk)