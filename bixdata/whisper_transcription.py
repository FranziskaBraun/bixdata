import whisper_timestamped as whisperTS
import whisper
import os
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Audio


def load_audio(audio_dir, keywords):
    audio_dir = Path(audio_dir)
    audios = list([f for f in audio_dir.glob('**/*.wav') if any(ele in f.name for ele in keywords)])
    print(f'loaded {len(audios)} audio files')
    return audios


def load_audio_huggingface(audio_dir, keywords):
    audio_dir = Path(audio_dir)
    audios = list([str(f) for f in audio_dir.glob('**/*.wav') if any(ele in f.name for ele in keywords)])
    print(f'loaded {len(audios)} audio files')
    audio_dataset = Dataset.from_dict({"audio": audios}).cast_column("audio", Audio())
    return audio_dataset


def filter_disfl(segment):
    return segment["text"] != "[*]"


def whisper_transcribe(audio_dir, keywords, device, model_name="large-v3", log_dir="/tmp", language="de"):
    audios = load_audio(audio_dir, keywords)
    model = whisper.load_model(model_name)
    model = model.to(device)
    print(f'{model_name} model loaded')
    options = dict(language=language, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    transcribe_options = dict(task="transcribe", **options)

    for wav in tqdm(audios):
        out = model.transcribe(str(wav), **transcribe_options)
        text = out['text']
        print(wav)

        text_directory = os.path.join(Path(log_dir), f'whisper_{model_name}')
        os.makedirs(text_directory, exist_ok=True)
        text_file_path = os.path.join(text_directory, wav.stem + '.txt')

        with open(text_file_path, 'w') as f:
            f.write(text.strip())

        return text


def whisper_ts_transcribe(audio_dir, keywords, device, special_tokens=None, model_name="large-v3", log_dir="/tmp", language="de"):
    audios = load_audio(audio_dir, keywords)
    print(len(audios))
    print('loading model')
    model = whisperTS.load_model(model_name, device=device)
    print(f'{model_name} model loaded')
    # no_speech_threshold=0.8
    options = dict(language=language, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True, detect_disfluencies=True)
    transcribe_options = dict(task="transcribe", **options)

    for wav in tqdm(audios):
        out = whisperTS.transcribe(model, str(wav), **transcribe_options)
        print(f'processing{wav}')
        text = get_text_for_special_token(special_tokens, out)
        text_directory = os.path.join(Path(log_dir), f'{language}/whisper_ts_{model_name}_{special_tokens}')
        os.makedirs(text_directory, exist_ok=True)
        text_file_path = os.path.join(text_directory, wav.stem + '.txt')

        with open(text_file_path, 'w') as f:
            f.write(text)


def get_text_for_special_token(special_tokens, out):
    if special_tokens:
        words = []
        for segment in out["segments"]:
            words.extend(segment["words"])

        if special_tokens == "baseline1":
            # to get pauses only we need to filter out disfluencies before
            words = list(filter(filter_disfl, words))
            text = baseline1_pauses(words)
        elif special_tokens == "baseline2":
            # filter out disfluencies
            words = list(filter(filter_disfl, words))
            text = baseline2_pauses(words)
        elif special_tokens == "baseline3":
            # filter out disfluencies
            words = list(filter(filter_disfl, words))
            text = baseline3_pauses(words)
        elif special_tokens == "disfl":
            text = whisper_disfluencies(words)
        elif special_tokens == "pause_disfl":
            text = baseline3_pauses(words)
        elif special_tokens == "continuous":
            text = continuous_pauses(words)
    else:
        text = out["text"]
    return text.strip()


def continuous_pauses(words):
    text = ""
    for i in range(0, len(words) - 1):
        word = words[i]["text"]
        pause = words[i + 1]["start"] - words[i]["end"]
        if pause >= 0.1:
            count = pause // 0.1
            tag = count * "[..] "
        else:
            tag = ""
        text += f"{word} {tag}"
    # add last word
    text += words[-1]["text"]
    return text


def baseline3_pauses(words):
    text = ""
    for i in range(0, len(words) - 1):
        word = words[i]["text"]
        pause = words[i + 1]["start"] - words[i]["end"]
        if 0.2 <= pause <= 0.6:
            tag = "[..] "
        elif 0.6 < pause < 1.5:
            tag = "[...] "
        elif pause >= 1.5:
            tag = "[....] "
        else:
            tag = ""
        text += f"{word} {tag}"
    # add last word
    text += words[-1]["text"]
    return text


def baseline1_pauses(words):
    text = ""
    for i in range(0, len(words) - 1):
        word = words[i]["text"]
        pause = words[i + 1]["start"] - words[i]["end"]
        if 0.05 <= pause < 0.5:
            tag = "[..] "
        elif 0.5 <= pause <= 2.0:
            tag = "[...] "
        elif pause > 2.0:
            tag = "[....] "
        else:
            tag = ""
        text += f"{word} {tag}"
    # add last word
    text += words[-1]["text"]
    return text


def baseline2_pauses(words):
    text = ""
    for i in range(0, len(words) - 1):
        word = words[i]["text"]
        pause = words[i + 1]["start"] - words[i]["end"]
        if 0.05 <= pause <= 0.1:
            tag = "[..] "
        elif 0.1 < pause <= 0.3:
            tag = "[...] "
        elif 0.3 < pause <= 0.6:
            tag = "[....] "
        elif 0.6 < pause <= 1.0:
            tag = "[.....] "
        elif 1.0 < pause <= 2.0:
            tag = "[......] "
        elif pause > 2.0:
            tag = "[.......] "
        else:
            tag = ""
        text += f"{word} {tag}"
    # add last word
    text += words[-1]["text"]
    return text


def whisper_disfluencies(words):
    text = ""
    for i in range(0, len(words) - 1):
        word = words[i]["text"]
        text += f"{word} "
    # add last word
    text += words[-1]["text"]
    return text

