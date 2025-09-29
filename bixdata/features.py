from transformers import BertTokenizer, BertModel, pipeline, WhisperProcessor, WhisperModel, Wav2Vec2Processor, Wav2Vec2Tokenizer,  WhisperForConditionalGeneration
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from .constants import *
from .custom_feature_extraction_pipeline import CustomFeatureExtractionPipeline
from .custom_whisper_model import CustomWhisperForConditionalGeneration
import whisper_timestamped as whisperTS
from sktdementia.audio import load_audio, load_and_split_audio, load_audio_and_mask_chunked, load_audio_and_mask_webrtc


def feature_extraction_audio(w2v2_model, w2v2_extract_layer, labels, pooling, normalize_audio, nj, logger):
    try:
        feature_extractor = Wav2Vec2Processor.from_pretrained(w2v2_model)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(w2v2_model)

    except OSError as e:
        logger.error(e)
        logger.info('loading preprocessor from facebook/wav2vec2-base-960h instead')
        feature_extractor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

    pipe = pipeline(model=w2v2_model, processor=feature_extractor,
                    tokenizer=tokenizer, task='feature-extraction', pipeline_class=CustomFeatureExtractionPipeline,
                    model_kwargs={'output_hidden_states': True},
                    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    def normalize_audio_data(wav):
        wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-9)
        return wav

    def set_audio(example):
        if normalize_audio:
            example['audio'] = normalize_audio_data(np.array(example['audio']))
        else:
            example['audio'] = np.array(example['audio'])
        return example

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(set_audio)

    embeddings = []

    # model = Wav2Vec2Model(Wav2Vec2Config()).from_pretrained(w2v2_model)
    extract_batch_size = nj
    for idx in tqdm(range(0, len(loaded_wavs['train']), extract_batch_size)):
        with torch.no_grad():
            hidden_states = pipe(loaded_wavs['train'][idx:idx + extract_batch_size]['audio'], chunk_length_s=20,
                                 # option for stride_length_s
                                 return_tensors='pt', padding=True,
                                 sampling_rate=SAMPLING_RATE,
                                 )

            per_layer_embs = [hs[w2v2_extract_layer] for hs in hidden_states]
            if pooling == 'mean':
                embeddings = [*embeddings, *[torch.mean(t.squeeze(), axis=0).numpy() for t in per_layer_embs]]
            elif pooling == 'stats':
                embeddings = [*embeddings,
                              *[torch.concat((torch.mean(t.squeeze(), axis=0), torch.std(t.squeeze(), axis=0))).numpy()
                                for
                                t in
                                per_layer_embs]]
            elif pooling == 'sum':
                embeddings = [*embeddings, *[torch.sum(t.squeeze(), axis=0).numpy() for t in per_layer_embs]]
            else:
                embeddings = [*embeddings, *[t.squeeze().numpy() for t in per_layer_embs]]

    return embeddings


def feature_extraction_text(bert_model, labels, pooling, nj):
    pipe = pipeline(model=bert_model, task='feature-extraction', model_kwargs={'output_hidden_states': True},
                    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    def load_text(example):
        example['text'] = example['text'].strip()
        return example

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_txts = datasets.map(load_text)

    embeddings = []

    extract_batch_size = nj
    for idx in tqdm(range(0, len(loaded_txts['train']), extract_batch_size)):
        with torch.no_grad():
            # FeatureExtractionPipeline returns last_hidden_state = model_output[0]
            last_hidden_states = pipe(loaded_txts['train'][idx:idx + extract_batch_size]['text'],
                                 # option for stride_length_s
                                 return_tensors='pt', padding=True,
                                 )

            if pooling == 'mean':
                embeddings = [*embeddings, *[torch.mean(t.squeeze(), axis=0).numpy() for t in last_hidden_states]]
            elif pooling == 'sum':
                embeddings = [*embeddings, *[torch.sum(t.squeeze(), axis=0).numpy() for t in last_hidden_states]]
            else:
                embeddings = [*embeddings, *[t.squeeze().numpy() for t in last_hidden_states]]

    return embeddings


def feature_extraction_text_tokens(bert_model, labels, pooling, nj):
    model = BertModel.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    pipe = pipeline(model=model, tokenizer=tokenizer, task='feature-extraction', model_kwargs={'output_hidden_states': True},
                    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    def load_text(example):
        example['text'] = example['text'].strip()
        return example

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_txts = datasets.map(load_text)

    embeddings = []
    tokens = []
    extract_batch_size = nj
    for idx in tqdm(range(0, len(loaded_txts['train']), extract_batch_size)):
        with torch.no_grad():
            # FeatureExtractionPipeline returns last_hidden_state = model_output[0]
            input_txt = loaded_txts['train'][idx:idx + extract_batch_size]['text']
            # input_txt = "Affe, [CLS] Elefant, Giraffe, [unused1] Hund [unused1] Katze, [unused1] Kuh [unused2]"
            last_hidden_states = pipe(input_txt, return_tensors='pt', padding=True, )
            # option for stride_length_s
            input_ids = tokenizer(input_txt, return_tensors='pt', padding=True, )['input_ids'].tolist()
            tokens = [*tokens, *[tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]]

            if pooling == 'mean':
                embeddings = [*embeddings, *[torch.mean(t.squeeze(), axis=0).numpy() for t in last_hidden_states]]
            elif pooling == 'sum':
                embeddings = [*embeddings, *[torch.sum(t.squeeze(), axis=0).numpy() for t in last_hidden_states]]
            else:
                embeddings = [*embeddings, *[t.squeeze().numpy() for t in last_hidden_states]]

    return embeddings, tokens


def feature_extraction_whisper(whisper_model, labels, pooling, nj, extract_layer=0):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model = model.to(device)

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_and_split_audio)

    encoder_embeddings, decoder_embeddings, transcriptions = [], [], []

    # Process each chunk separately
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio_chunks'], sampling_rate=SAMPLING_RATE, return_tensors="pt")
            inputs = inputs.to(device, torch.float32)
            # , return_token_timestamps=True
            # torch.cuda.empty_cache()
            outputs = model.generate(**inputs, language="german", task="transcribe", return_dict_in_generate=True, output_hidden_states=True, return_timestamps=True)[0:3]

            transcription = processor.batch_decode(outputs[0].cpu(), skip_special_tokens=True, decode_with_timestamps=True)
            transcription = ''.join(transcription)

            # extract_encoder_hidden_states = outputs['encoder_hidden_states']
            extract_encoder_hidden_states = np.concatenate(outputs[1][extract_layer].cpu().numpy())

            # extract_decoder_hidden_states = outputs['decoder_hidden_states']
            extract_decoder_hidden_states = np.concatenate(np.stack([token[extract_layer].cpu().numpy() for token in outputs[2]], axis=1)).squeeze()

            transcriptions.append(transcription)
            encoder_embeddings.append(extract_encoder_hidden_states)
            decoder_embeddings.append(extract_decoder_hidden_states)
            # torch.cuda.empty_cache()

    return encoder_embeddings, decoder_embeddings, transcriptions


def feature_extraction_encoder(whisper_model, labels, no_repeat_ngram_size=None, extract_layer=-1):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.to(device)
    model.eval()
    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_and_split_audio)
    args = {"language": "german",
            "task": "transcribe",
            "return_timestamps": True,
            "num_beams": BEAM_SIZE,
            "early_stopping": True,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "return_dict_in_generate": True,
            "output_hidden_states": True}
    encoder_embeddings = []
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio_chunks'],
                               return_tensors='pt',
                               truncation=False,
                               sampling_rate=SAMPLING_RATE,
                               return_attention_mask=True,
                               do_normalize=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            extract_encoder_hidden_states = model.generate(**inputs, **args)['encoder_hidden_states'][extract_layer].cpu()
            extract_encoder_hidden_states = np.concatenate(extract_encoder_hidden_states.numpy())
            encoder_embeddings.append(extract_encoder_hidden_states)
    return encoder_embeddings


def feature_extraction_decoder(whisper_model, labels, no_repeat_ngram_size=None, extract_layer=-1):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.to(device)
    model.eval()
    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_and_split_audio)
    args = {"language": "german",
            "task": "transcribe",
            "return_timestamps": True,
            "num_beams": BEAM_SIZE,
            "early_stopping": True,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "return_dict": True}
    decoder_embeddings = []
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio_chunks'],
                               return_tensors='pt',
                               truncation=False,
                               sampling_rate=SAMPLING_RATE,
                               return_attention_mask=True,
                               do_normalize=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            extract_decoder_hidden_states = model.generate(**inputs, **args)['decoder_hidden_states']
            extract_decoder_hidden_states = np.stack([token[extract_layer].cpu().squeeze(1) for token in extract_decoder_hidden_states], axis=1)
            hidden_shape = extract_decoder_hidden_states.shape
            extract_decoder_hidden_states = extract_decoder_hidden_states.reshape(int(hidden_shape[0]/BEAM_SIZE), BEAM_SIZE, hidden_shape[1], hidden_shape[2])
            extract_decoder_hidden_states = np.concatenate(extract_decoder_hidden_states[:, 0, :, :])
            decoder_embeddings.append(extract_decoder_hidden_states)
    return decoder_embeddings


def feature_extraction_encoder_decoder_long(whisper_model, labels, pooling, nj, extract_layer=0):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Step 1: Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    # model = CustomWhisperForConditionalGeneration.from_pretrained(whisper_model, attn_implementation="eager")
    model = CustomWhisperForConditionalGeneration.from_pretrained(whisper_model)
    model = model.to(device)
    # simple_model = WhisperModel.from_pretrained(whisper_model)
    # simple_model = simple_model.to(device)

    # model.eval()
    # forced_decoder_ids = torch.tensor(processor.get_decoder_prompt_ids())
    # pipe = pipeline(model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
    #                 task="automatic-speech-recognition",
    #                 chunk_length_s=30,
    #                 output_hidden_states=True,
    #                 device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    # model = WhisperModel.from_pretrained("openai/whisper-base")
    # feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # input_features = inputs.input_features
    # decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    # last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    # list(last_hidden_state.shape)

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_audio)

    encoder_embeddings, decoder_embeddings, transcriptions, timestamps, my_sequences = [], [], [], [], []
    extract_batch_size = nj
    for idx in tqdm(range(0, len(loaded_wavs['train']), extract_batch_size)):
        with torch.no_grad():
            batch = loaded_wavs['train'][idx:idx + extract_batch_size]['audio']
            # inputs_short = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            # inputs = processor(batch, return_tensors='pt',  padding='max_length', truncation=True,
            #               max_length=960000, return_attention_mask=True, sampling_rate=SAMPLE_RATE)
            inputs = inputs.to(device, torch.float32)
            # input_features = inputs.input_features.to(device, torch.float32)
            # attention_mask = inputs.attention_mask.to(device, torch.float32)
            # decoder_input_ids = processor.tokenizer("<|startoftranscript|>", return_tensors="pt").input_ids
            # decoder_input_ids = torch.tensor(processor.get_decoder_prompt_ids())
            # decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
            # outputs = model(**inputs, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
            # logits = outputs.logits
            # # Get predicted token IDs
            # predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the token IDs to text , temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  return_token_timestamps=True
            # outputs = model.generate(**inputs, language="german", task="transcribe", return_segments=True,
            #                          return_dict_in_generate=True, output_hidden_states=True, return_token_timestamps=True)

            outputs = model.generate(**inputs, language="german", task="transcribe",  return_dict_in_generate=True)
            sequences, batched_segments = outputs['sequences'], outputs['segments']
            transcription = processor.batch_decode(sequences, skip_special_tokens=True)
            # all_encoder_hidden_states = [torch.stack([torch.stack(seg['result']['encoder_hidden_states']) for seg in segments]) for segments in batched_segments]
            # extract_encoder_hidden_states = [torch.cat([seg['result']['encoder_hidden_states'][extract_layer] for seg in segments]) for segments in batched_segments]
            # extract_decoder_hidden_states = [torch.cat([torch.stack([token[0].squeeze() for token in seg['result']['decoder_hidden_states'][1:]]) for seg in segments]) for segments in batched_segments]

            # Extract the encoder hidden states for each batch element of segments
            extract_encoder_hidden_states = [
                torch.cat([seg['result']['encoder_hidden_states'][extract_layer] for seg in segments])
                for segments in batched_segments
            ]

            # Extract and stack the decoder hidden states for each batch element of segments
            extract_decoder_hidden_states = [
                torch.cat([
                    torch.stack([
                        token[0].squeeze() for token in seg['result']['decoder_hidden_states'][1:]
                    ])
                    for seg in segments
                ])
                for segments in batched_segments
            ]
            transcriptions = [*transcriptions, *transcription]

            # Extract the encoder hidden states for each batch element of segments
            ts = [
                torch.cat([seg['result']['token_timestamps'] for seg in segments])
                for segments in batched_segments
            ]

            if pooling == 'mean':
                encoder_embeddings = [*encoder_embeddings, *[torch.mean(t.squeeze(), axis=0).numpy() for t in extract_encoder_hidden_states]]
                decoder_embeddings = [*decoder_embeddings, *[torch.mean(t.squeeze(), axis=0).numpy() for t in extract_decoder_hidden_states]]
            elif pooling == 'sum':
                encoder_embeddings = [*encoder_embeddings, *[torch.sum(t.squeeze(), axis=0).numpy() for t in extract_encoder_hidden_states]]
                decoder_embeddings = [*decoder_embeddings, *[torch.sum(t.squeeze(), axis=0).numpy() for t in extract_decoder_hidden_states]]
            else:
                encoder_embeddings = [*encoder_embeddings, *[t.squeeze().numpy() for t in extract_encoder_hidden_states]]
                decoder_embeddings = [*decoder_embeddings, *[t.squeeze().numpy() for t in extract_decoder_hidden_states]]

    return encoder_embeddings, decoder_embeddings, transcriptions


def feature_extraction_transcription(whisper_model, labels, no_repeat_ngram_size=None):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model.to(device)
    # def get_tokens_as_list(word_list):
    #     "Converts a sequence of words into a list of tokens"
    #     tokens_list = []
    #     for word in word_list:
    #         tokenized_word = processor.tokenizer([word], add_special_tokens=False).input_ids[0]
    #         tokens_list.append(tokenized_word)
    #     return tokens_list
    #
    # bad_words_ids = get_tokens_as_list(word_list=["Vielen", "Dank"])
    model.eval()
    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_audio_and_mask_webrtc)
    args = {"language": "german",
            "task": "transcribe",
            "return_timestamps": True,
            "num_beams": BEAM_SIZE,
            "early_stopping": True,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            # "bad_words_ids": bad_words_ids
            # "repetition_penalty": 1.2,
            # "encoder_repetition_penalty": 1.2,
            # "best_of": 5,
            # "encoder_attention_mask": attention_mask
            }
    transcriptions = []
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio'],
                               return_tensors='pt',
                               truncation=False,
                               sampling_rate=SAMPLING_RATE,
                               return_attention_mask=True,
                               do_normalize=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # attention_mask = torch.tensor(example['silence_mask'], dtype=torch.float32).unsqueeze(dim=0)
            # torch.tensor(example['silence_mask'], dtype=torch.int32).unsqueeze(dim=0)
            # insert silence mask into attention mask
            # attention_mask[:, 0:silence_mask.size(-1)] = silence_mask
            # merge silence attention mask with padding attention mask
            # assert attention_mask.shape == inputs.attention_mask.shape, \
            #     f"Mismatch in shapes: attention_mask.shape = {attention_mask.shape}, " \
            #     f"inputs.attention_mask.shape = {inputs.attention_mask.shape}"

            # assert attention_mask.shape[-1] == input_features.shape[-1], \
            #     f"Mismatch in shapes: attention_mask.shape = {attention_mask.shape[-1]}, " \
            #     f"input_features.shape = {input_features.shape[-1]}"

            # attention_mask = attention_mask * inputs.attention_mask
            # attention_mask = attention_mask.to(device=device, dtype=torch.float32)

            transcription = model.generate(**inputs, **args).cpu()
            transcription = processor.batch_decode(transcription, skip_special_tokens=True, decode_with_timestamps=True)
            transcriptions = [*transcriptions, *transcription]

    return transcriptions


def feature_extraction_whisper_all(whisper_model, labels, extract_layer=12):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    model = model.to(device)

    datasets = DatasetDict({'train': Dataset.from_pandas(labels)})
    loaded_wavs = datasets.map(load_audio_and_mask_chunked)
    transcriptions = []
    decoder_embeddings = []
    encoder_embeddings = []
    model.eval()
    # Process each chunk separately
    for example in tqdm(loaded_wavs['train']):
        with torch.no_grad():
            inputs = processor(example['audio_chunks'], sampling_rate=SAMPLING_RATE, return_tensors="pt")
            inputs = inputs.to(device, torch.float32)
            extract_encoder_hidden_states = model.generate(**inputs,
                                                           language="german",
                                                           task="transcribe",
                                                           return_dict_in_generate=True,
                                                           output_hidden_states=True,
                                                           return_timestamps=True,
                                                           return_token_timestamps=True).cpu()

            [
                'encoder_hidden_states'][extract_layer].numpy()

            transcription = processor.batch_decode(transcription, skip_special_tokens=True, decode_with_timestamps=True)
            transcriptions = [*transcriptions, *transcription]

            extract_decoder_hidden_states = np.concatenate(
                np.stack([token[extract_layer].cpu().numpy() for token in extract_decoder_hidden_states],
                         axis=1)).squeeze()

            decoder_embeddings.append(extract_decoder_hidden_states)

            extract_encoder_hidden_states = np.concatenate(extract_encoder_hidden_states)
            encoder_embeddings.append(extract_encoder_hidden_states)

    return transcriptions, encoder_embeddings, decoder_embeddings


def feature_extraction_openai_whisper(whisper_model, labels, pooling, nj, extract_layer=0):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Step 1: Load the Whisper model
    print('loading model')
    model = whisperTS.load_model(whisper_model, device=device)
    print(f'{whisper_model} model loaded')
    audios = labels['audio_path'].values
    print(len(audios))
    # no_speech_threshold=0.8
    options = dict(language='de', beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True,
                   detect_disfluencies=True)
    transcribe_options = dict(task="transcribe", **options)

    for wav in tqdm(audios):
        audio = whisperTS.load_audio(str(wav))
        audio = whisperTS.pad_or_trim(audio, 960000)
        mel = whisperTS.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        encoder_output = model.encoder(mel.unsqueeze(0))
        encoder_embeddings = encoder_output.last_hidden_state
        out = whisperTS.transcribe(model, str(wav), **transcribe_options)
        print(f'processing{wav}')