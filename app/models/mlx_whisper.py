from functools import lru_cache

import librosa
from mlx_whisper.transcribe import transcribe
import numpy as np

SAMPLING_RATE = 16000
CHUNK_SIZE = 30


@lru_cache(maxsize=32)
def load_audio(fname):
    """Load and cache audio file. Cache size limited to 32 recent files."""
    a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
    return a


@lru_cache(maxsize=32)
def calculate_audio_duration(audio_path: str) -> int:
    """Calculate the duration of the audio file in seconds."""
    audio = load_audio(audio_path)
    return len(audio) / SAMPLING_RATE


class MLX_Whisper:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def _transcribe_generator(self, audio_path: str, **kwargs):
        """Stream transcription by processing audio in larger chunks."""
        # Load the audio file
        audio = load_audio(audio_path)
        duration = calculate_audio_duration(audio_path)

        beg = 0.0
        while beg < duration:
            # Calculate chunk boundaries
            chunk_end = min(beg + CHUNK_SIZE, duration)

            # Extract audio chunk
            beg_samples = int(beg * SAMPLING_RATE)
            end_samples = int(chunk_end * SAMPLING_RATE)
            audio_chunk = audio[beg_samples:end_samples]

            # Transcribe chunk
            result = transcribe(audio_chunk, path_or_hf_repo=self.model_path, **kwargs)

            # Add timing information
            result["chunk_start"] = beg
            result["chunk_end"] = chunk_end

            yield result

            beg += CHUNK_SIZE

    def __call__(self, audio_path: str, stream: bool = False, **kwargs):
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            stream: If True, yields chunks. If False, transcribes entire file at once.
            **kwargs: Additional arguments passed to transcribe()
        """
        if stream:
            return self._transcribe_generator(audio_path, **kwargs)
        return transcribe(audio_path, path_or_hf_repo=self.model_path, **kwargs)


if __name__ == "__main__":
    model = MLX_Whisper("mlx-community/whisper-tiny")
    # Non-streaming (fastest for most use cases)
    result = model("examples/audios/podcast.wav", stream=True)
    for chunk in result:
        print(f"[{chunk['chunk_start']:.1f}s - {chunk['chunk_end']:.1f}s]: {chunk['text']}")
