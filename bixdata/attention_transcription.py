from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from pydub import AudioSegment
from pydub.silence import detect_silence

SAMPLE_RATE = 16000


def load_audio_and_mask(example):
    audio_array, sr = librosa.load(example['audio_path'], sr=SAMPLE_RATE)
    example['audio'] = np.array(audio_array)
    audio = AudioSegment.from_file(example['audio_path'], format="wav")
    mask_length = int(np.ceil(len(audio) / 10))
    attention_mask = np.ones(mask_length, dtype=int)
    silence_threshold = audio.dBFS - 10
    silent_intervals = detect_silence(audio, min_silence_len=500, silence_thresh=silence_threshold)
    for start_ms, end_ms in silent_intervals:
        start_frame = int(start_ms / 10)  # Convert ms to frames
        end_frame = int(end_ms / 10)
        attention_mask[start_frame:end_frame] = 0  # Silence marked as 0
    example['silence_mask'] = attention_mask
    return example


def feature_extraction_transcription(whisper_model, labels):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.to(device)
    model.eval()
    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_audio_and_mask)
    transcriptions = []
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio'],
                               return_tensors='pt',
                               # padding='max_length',
                               truncation=False,
                               return_attention_mask=True,
                               sampling_rate=SAMPLE_RATE,
                               return_token_timestamps=True,
                               do_normalize=True)
            attention_mask = inputs.attention_mask
            silence_mask = torch.tensor(example['silence_mask'], dtype=torch.int32).unsqueeze(dim=0)
            # insert silence mask into attention mask
            attention_mask[:, 0:silence_mask.size(-1)] = silence_mask
            # merge silence attention mask with padding attention mask
            # attention_mask = attention_mask * inputs.attention_mask
            input_features = inputs.input_features.to(device=device)
            attention_mask = attention_mask.to(device=device)
            transcription = model.generate(input_features,
                                           language="german",
                                           task="transcribe",
                                           attention_mask=attention_mask,
                                           return_timestamps=True).cpu()
            transcription = processor.batch_decode(transcription, skip_special_tokens=True, decode_with_timestamps=True)
            transcriptions = [*transcriptions, *transcription]
    return transcriptions
