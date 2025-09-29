from sktdementia.constants import SAMPLING_RATE, MAX_AUDIO_DURATION_S
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from webrtcvad import Vad
import math


def load_audio(example):
    audio, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE, duration=MAX_AUDIO_DURATION_S)
    example['audio'] = np.array(audio)
    return example


def load_and_split_audio(example, max_chunk_duration=30):
    audio, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE)
    example['audio_chunks'] = []
    chunk_size = max_chunk_duration  # Maximum chunk duration in seconds
    # Split audio into chunks
    for start in range(0, len(audio), chunk_size * sr):
        chunk = audio[start:start + chunk_size * sr]
        example['audio_chunks'].append(np.array(chunk))
    return example


def load_audio_and_mask_chunked(example, max_chunk_duration=30):
    audio_array, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE)

    audio = AudioSegment.from_file(example['audio_path'], format="wav")
    mask_length = int(np.ceil(len(audio) / 10 / 3000) * 3000)  # Convert ms to frames (10ms per frame)
    attention_mask = np.ones(mask_length, dtype=int)
    silent_intervals = detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    for start_ms, end_ms in silent_intervals:
        start_frame = int(start_ms / 10)  # Convert ms to frame index
        end_frame = int(end_ms / 10)
        attention_mask[start_frame:end_frame] = 0  # Silence marked as 0

    example['audio_chunks'] = []
    example['silence_masks'] = []
    chunk_size_sample = max_chunk_duration * sr  # Chunk size in samples (e.g., 30s * 16000)
    for start_sample in range(0, len(audio_array), chunk_size_sample):
        chunk = audio_array[start_sample:start_sample + chunk_size_sample]
        start_frame = int(start_sample / sr * 100)
        end_frame = int((start_sample + chunk_size_sample) / sr * 100)
        mask = attention_mask[start_frame:end_frame]
        example['audio_chunks'].append(chunk)
        example['silence_masks'].append(mask)

    return example


def load_audio_and_mask(example):
    audio_array, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE, duration=MAX_AUDIO_DURATION_S)
    example['audio'] = np.array(audio_array)

    audio = AudioSegment.from_file(example['audio_path'], format="wav")

    mask_length = math.ceil((audio.duration_seconds * 1000) / 10)
    attention_mask = np.ones(mask_length, dtype=int) if mask_length > 3000 else np.ones(3000, dtype=int)
    # mask_length = int((MAX_AUDIO_DURATION_S * 1000) / FRAME_RATE_MS)
    # attention_mask = np.ones(mask_length, dtype=int)
    silence_threshold = audio.dBFS - 10
    silent_intervals = detect_silence(audio, min_silence_len=500, silence_thresh=silence_threshold)
    for start_ms, end_ms in silent_intervals:
        start_frame = int(start_ms / 10)  # Convert ms to frames
        end_frame = int(end_ms / 10)
        attention_mask[start_frame:end_frame] = 0  # Silence marked as 0
    example['silence_mask'] = attention_mask
    return example


def load_audio_and_mask_webrtc(example):
    audio_array, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE)
    example['audio'] = np.array(audio_array)

    # Load the audio file
    audio = AudioSegment.from_file(example['audio_path'], format="wav")

    mask_length = int((audio.duration_seconds * 1000) / 10)
    attention_mask = np.ones(mask_length, dtype=int) if mask_length > 3000 else np.ones(3000, dtype=int)
    # mask_length = int((MAX_AUDIO_DURATION_S * 1000) / FRAME_RATE_MS)
    # attention_mask = np.ones(mask_length, dtype=int)
    silent_intervals = detect_silence_webrtc(audio, min_silence_len=500, aggressiveness_level=1)
    for start_ms, end_ms in silent_intervals:
        start_frame = int(start_ms / 10)  # Convert ms to frames
        end_frame = int(end_ms / 10)
        attention_mask[start_frame:end_frame] = 0  # Silence marked as 0
    example['silence_mask'] = attention_mask
    return example


def detect_silence_webrtc(audio_segment, min_silence_len=500, aggressiveness_level=1):
    """
    Detect silence intervals in an audio segment using WebRTC VAD.
    """
    vad = Vad(aggressiveness_level)
    audio_data = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    frame_duration_ms = 10  # WebRTC VAD processes 10ms frames
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(audio_data) // frame_size

    silent_intervals = []
    silence_start = None

    for i in range(num_frames):
        start_ms = i * frame_duration_ms
        end_ms = start_ms + frame_duration_ms
        frame = audio_data[i * frame_size:(i + 1) * frame_size]

        if len(frame) < frame_size:
            continue

        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if not is_speech:
            if silence_start is None:
                silence_start = start_ms
        else:
            if silence_start is not None:
                if end_ms - silence_start >= min_silence_len:
                    silent_intervals.append((silence_start, end_ms))
                silence_start = None

    # Handle trailing silence
    if silence_start is not None:
        silent_intervals.append((silence_start, len(audio_data) * 1000 / sample_rate))

    return silent_intervals


def detect_silence_and_speech_webrtc(audio_segment, min_silence_len=500, aggressiveness_level=1):
    """
    Detect both silence and speech intervals in an audio segment using WebRTC VAD.
    """
    vad = Vad(aggressiveness_level)
    audio_data = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    frame_duration_ms = 10  # WebRTC VAD processes 10ms frames
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(audio_data) // frame_size

    silent_intervals = []
    speech_intervals = []
    silence_start = None
    speech_start = None

    for i in range(num_frames):
        start_ms = i * frame_duration_ms
        end_ms = start_ms + frame_duration_ms
        frame = audio_data[i * frame_size:(i + 1) * frame_size]

        if len(frame) < frame_size:
            continue

        is_speech = vad.is_speech(frame.tobytes(), sample_rate)

        if not is_speech:
            if silence_start is None:
                silence_start = start_ms
            if speech_start is not None:
                speech_intervals.append((speech_start, start_ms))
                speech_start = None
        else:
            if speech_start is None:
                speech_start = start_ms
            if silence_start is not None:
                if end_ms - silence_start >= min_silence_len:
                    silent_intervals.append((silence_start, end_ms))
                silence_start = None

    # Handle trailing intervals
    if silence_start is not None:
        silent_intervals.append((silence_start, len(audio_data) * 1000 / sample_rate))
    if speech_start is not None:
        speech_intervals.append((speech_start, len(audio_data) * 1000 / sample_rate))

    return speech_intervals, silent_intervals


def load_audio_and_mask_silence_chunked(example):
    """
    Load audio, detect silence intervals using WebRTC VAD, and chunk Librosa audio array into 30-second natural segments.
    """
    # Load audio using Librosa
    audio_array, sr = librosa.load(example['audio_path'], sr=SAMPLING_RATE, duration=MAX_AUDIO_DURATION_S)
    example['audio'] = np.array(audio_array)

    # Convert audio array to AudioSegment for silence detection
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sr,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )
    silent_intervals = detect_silence_webrtc(audio_segment, min_silence_len=500, aggressiveness_level=1)

    # Initialize variables for chunking
    max_chunk_duration_samples = 30 * sr  # 30 seconds in samples
    chunks = []
    chunk_start = 0
    silence_mask = np.ones(len(audio_array), dtype=int)

    # Iterate through silence intervals and determine chunk boundaries
    for start_ms, end_ms in silent_intervals:
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)

        if start_sample - chunk_start > max_chunk_duration_samples:
            # If the chunk exceeds the max duration, cut here
            chunk_end = min(chunk_start + max_chunk_duration_samples, start_sample)
            chunks.append(audio_array[chunk_start:chunk_end])
            chunk_start = chunk_end

        # Mark silence in the mask
        silence_mask[start_sample:end_sample] = 0

    # Add the final chunk if it exists
    if chunk_start < len(audio_array):
        chunks.append(audio_array[chunk_start:])

    # Save chunks and silence mask
    example['audio_chunks'] = chunks
    example['silence_mask'] = silence_mask
    return example
