import shutil
import sys
import os
from datetime import datetime as dt
import time
import logging
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import click
import torch
torch.cuda.empty_cache()
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, make_scorer, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from .dataloading import load_audio_data, load_meta_data_bix, load_meta_data_mc, load_text_data, CombinedDataset, NormDataset
from .features import feature_extraction_audio, feature_extraction_encoder, feature_extraction_decoder, feature_extraction_transcription, feature_extraction_text_tokens
# BIX_BNT_ANSWER, MC_BNT_ANSWER,
from torch.utils.data import DataLoader
from .models import ScalarFusionModel, CrossAttFusionModel, NormScalarFusionModel, EmbeddingModel, CrossEmbeddingModel # , ConcatenationModel, ParallelModel, AttentionFusionModel, FiLMModel, GNNFusionModel
# from .samplers import StratifiedBatchSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from captum.attr import IntegratedGradients
from sklearn.utils.class_weight import compute_class_weight
from .pytorchtools import EarlyStopping
# from torch.optim.lr_scheduler import LinearLR, StepLR, ExponentialLR, MultiStepLR
from .whisper_transcription import whisper_transcribe, whisper_ts_transcribe
from .constants import *
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLI_NAME = "sktdementia"


def setup_logging(log_level):
    ext_logger = logging.getLogger(f"py.{CLI_NAME}")
    logging.captureWarnings(True)
    level = getattr(logging, log_level)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=level)
    if level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


@click.group()
@click.option("-l", "--log-level", default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
def cli(log_level):
    setup_logging(log_level)


def get_y_and_groups(labels, label_col):
    # groups for split
    groups = labels['subject'].values
    # encode labels
    y = labels[label_col].values
    # convert to int for comma values
    y = np.char.replace(y.astype(str), ',', '.').astype(float).astype(int)
    return y, groups


def load_features(features_path):
    embeddings = torch.load(features_path)
    embeddings = [torch.tensor(element, dtype=torch.float32) for element in embeddings]
    X = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    return X


@cli.command()
@click.option('--features-path', required=True, help='path to extracted features to combine')
@click.option('--labels-path', required=True, help='path to label file')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option('--regression', is_flag=True, help='if to use as regression problem, default is classification')
@click.option('--lr', default=1e-4, help='learning rate')
@click.option('--metric', default='recall_macro', help='metric to optimize')
@click.option('--save-model', is_flag=True, help='if set model dict is saved')
@click.option("--log-dir", default='')
def train_single_scalar_classifier(features_path, labels_path, label_col, ng, random_state, regression, lr, metric, save_model, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    model_name = features_path.split('/')[-1].split('.')[0]
    logger.info(f'MODEL: {model_name}')
    X = load_features(features_path)
    labels = pd.read_csv(labels_path, delimiter=';')
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    S = labels[f'{label_col}_calc'].values
    S = torch.tensor(S, dtype=torch.float32)
    logger.info(f'data length: {len(X)}')
    def grid_search(params):
        results = []
        for param in params:
            train_params = {"epochs": 100, "batch_size": 8, "lr": lr, "save": save_model, "regression": regression,
                            "combined": False, "tests": [int(model_name.split('_')[0][-1])], "hidden_dims": param}
            r = train_n_fold(X=list(zip(S, X)), y=y, groups=groups, model_name=model_name, train_params=train_params, ng=ng,
                             random_state=random_state, start_time=start_time, exp_log=exp_log, log_dir=log_dir, logger=logger)
            print(f'Results for {param}: {r[metric].mean()}+-{r[metric].std()}')
            results.append((param, r[metric].mean(), r[metric].std()))
        return min(results, key=lambda x: x[1]) if regression else max(results, key=lambda x: x[1])
    # lrs = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
    # dims = [[512, 256], [512, 128], [512, 64], [512, 32], [256, 128], [256, 64], [256, 32]]
    dims = [[256, 64]]
    # params = [lr]
    best_param, best_score, std = grid_search(dims)
    print(f'MODEL: {model_name}')
    print("Best Parameters:", best_param)
    print(f'Best {metric}: {best_score}+-{std}')


@cli.command()
@click.option('--features-path1', required=True, help='path to extracted features 1 to combine')
@click.option('--features-path2', required=True, help='path to extracted features 2 to combine')
@click.option('--labels-path', required=True, help='path to label file')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option('--regression', is_flag=True, help='if to use as regression problem, default is classification')
@click.option('--lr', default=1e-4, help='learning rate')
@click.option('--metric', default='recall_macro', help='metric to optimize')
@click.option('--save-model', is_flag=True, help='if set model dict is saved')
@click.option("--log-dir", default='')
def train_duo_scalar_classifier(features_path1, features_path2, labels_path, label_col, ng, random_state, regression, lr, metric, save_model, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    model_name = features_path1.split('/')[-1].split('.')[0] + "_" + features_path2.split('/')[-1].split('.')[0]
    logger.info(f'MODEL: {model_name}')
    X1 = load_features(features_path1)
    X2 = load_features(features_path2)
    labels = pd.read_csv(labels_path, delimiter=';')
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    S = labels[f'{label_col}_calc'].values
    S = torch.tensor(S, dtype=torch.float32)
    logger.info(f'data length: {len(X1)}')

    def grid_search(params):
        results = []
        for param in params:
            train_params = {"epochs": 100, "batch_size": 8, "lr": param, "save": save_model, "regression": regression,
                            "combined": False, "tests": [int(model_name.split('_')[0][-1])]}
            r = train_n_fold(X=list(zip(S, X1, X2)), y=y, groups=groups, model_name=model_name, train_params=train_params, ng=ng,
                             random_state=random_state, start_time=start_time, exp_log=exp_log, log_dir=log_dir, logger=logger)
            print(f'Results for {param}: {r[metric].mean()}+-{r[metric].std()}')
            results.append((param, r[metric].mean(), r[metric].std()))
        return min(results, key=lambda x: x[1]) if regression else max(results, key=lambda x: x[1])
    # lrs = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
    # dims = [[512, 256], [512, 128], [512, 64], [512, 32], [256, 128], [256, 64], [256, 32], [128, 64], [128, 32]]
    params = [lr]
    best_param, best_score, std = grid_search(params)
    print(f'MODEL: {model_name}')
    print("Best Parameters:", best_param)
    print(f'Best {metric}: {best_score}+-{std}')


@cli.command()
@click.option('--features-dir', required=True, help='directory to extracted features to combine')
@click.option('--labels-path', required=True, help='path to extracted labels')
@click.option('--norm-path', required=True, help='path to normalization file')
@click.option('--models-path', required=True, help='path to pretrained models')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option('--tests', default='SKT1_SKT2_SKT3_SKT6_SKT7_SKT8', help='tests to combine')
@click.option('--layer', default=12, help='feature extraction layer')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option('--regression', is_flag=True, help='if to use as regression problem, default is classification')
@click.option('--lr', default=1e-4, help='learning rate')
@click.option('--metric', default='recall_macro', help='metric to optimize')
@click.option('--save-model', is_flag=True, help='if set model dict is saved')
@click.option("--log-dir", default='')
def train_combined_scalar_classifier(features_dir, labels_path, norm_path, models_path, label_col, tests, layer, ng, random_state, regression, lr, metric, save_model, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    ft_paths = [t for t in Path(features_dir).glob(f'**/*{layer}.pkl')]
    tests = tests.split('_')
    labels = pd.read_csv(labels_path, delimiter=';')
    # label_col = [label_col] + tests
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    iq = labels['IQ'].replace(['<90', '90-110', '>110'], [0, 1, 2]).values
    age = labels['age'].values
    X, S, loaded_tests = [], [], []
    model_name = ""
    for test in tests:
        for ft_path in ft_paths:
            split = ft_path.stem.split('_')
            if split[0] == test:
                model_name = '_'.join(ft_path.stem.split('_')[1:])
                x = load_features(ft_path)
                X.append(x)
                s = torch.tensor(labels[f'{test}_calc'].values, dtype=torch.float32)
                S.append(s)
                loaded_tests.append(test)
    model_name = f'{"_".join(loaded_tests)}_{model_name}'
    logger.info(f'data length: {len(X[0])}')
    # loaded_tests = torch.tensor([int(test[-1]) for test in loaded_tests], dtype=torch.float32)
    norm = pd.read_csv(norm_path, delimiter=';')
    norm['iq'] = norm['iq'].replace(['<90', '90-110', '>110'], [0, 1, 2])
    norm = torch.tensor(norm[['test_id', 'iq', 'min_age', 'max_age', 'min_value', 'max_value', 'norm_value']].values)

    def grid_search(lrs):
        results = []
        for lr in lrs:
            train_params = {"epochs": 100, "batch_size": 8, "lr": lr, "save": save_model, "regression": regression, "combined": True, "tests": loaded_tests, "models_path": models_path}
            r = train_n_fold(X=list(zip(age, iq, zip(*S), zip(*X))), y=y, groups=groups, model_name=model_name, train_params=train_params, ng=ng,
                             random_state=random_state, start_time=start_time, exp_log=exp_log, log_dir=log_dir,
                             logger=logger, norm=norm)
            results.append((lr, r[metric].mean(), r[metric].std()))
        return min(results, key=lambda x: x[1]) if regression else max(results, key=lambda x: x[1])
    lrs = [lr]
    # lrs = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
    best_lr, best_score, std = grid_search(lrs)
    print(f'MODEL: {model_name}')
    print("Best learning rate:", best_lr)
    print(f'Best {metric}: {best_score}+-{std}')


@cli.command()
@click.option('--features1-path', required=True, help='path to extracted features1 to combine')
@click.option('--features2-path', required=True, help='path to extracted features2 to combine')
@click.option('--labels-path', required=True, help='path to extracted labels')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option('--regression', is_flag=True, help='if to use as regression problem, default is classification')
@click.option('--lr', default=1e-4, help='learning rate for training')
@click.option('--metric', default='recall_macro', help='metric to optimize')
@click.option("--log-dir", default='/tmp')
def train_combined_attention_classifier(features1_path, features2_path, labels_path, label_col, ng, random_state, regression, lr, metric, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    model_name = features1_path.split('.')[0] + "_combined_" + features2_path.split('.')[0]
    logger.info(f'MODEL: {model_name}')
    labels = torch.load(labels_path)
    X1 = torch.load(features1_path)
    X2 = torch.load(features2_path)
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    logger.info(f'data length 1: {len(X1)}, data length 2: {len(X2)}')

    def grid_search(lrs):
        results = []
        for lr in lrs:
            train_params = {"epochs": 100, "batch_size": 8, "lr": lr, "save": False, "regression": regression, "combined": True}
            r = train_n_fold(X=list(zip(X1, X2)), y=y, groups=groups, model_name=model_name, train_params=train_params, ng=ng,
                             random_state=random_state, start_time=start_time, exp_log=exp_log, log_dir=log_dir,
                             logger=logger)
            results.append((lr, r[metric].mean(), r[metric].std()))
        return min(results, key=lambda x: x[1]) if regression else max(results, key=lambda x: x[1])
    lrs = [lr]
    # lrs = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
    best_lr, best_score, std = grid_search(lrs)
    print(f'MODEL: {model_name}')
    print("Best learning rate:", best_lr)
    print(f'Best {metric}: {best_score}+-{std}')


@cli.command()
@click.option('--audio-dir', required=True, help='directory to audio files')
@click.option('--keyword', default='', help='keyword to filter audio files')
@click.option('--model-name', default='large-v3', help='whisper model name')
@click.option("--log-dir", default='/transcripts')
def transcribe_audio(audio_dir, keyword, model_name, log_dir):
    log_dir = Path(log_dir)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    special_tokens = ["baseline3", "pause_disfl", "continuous"]
    whisper_transcribe(audio_dir=audio_dir, keywords=[keyword], device=device, model_name=model_name, log_dir=log_dir)
    for st in special_tokens:
        whisper_ts_transcribe(audio_dir=audio_dir, keywords=[keyword], device=device, special_tokens=st, model_name=model_name, log_dir=log_dir)


@cli.command()
@click.option('--features-path', required=True, help='path to extracted features')
@click.option('--labels-path', required=True, help='path to extracted labels')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option('--regression', is_flag=True, help='if to use as regression problem, default is classification')
@click.option('--lr', default=1e-4, help='learning rate')
@click.option('--metric', default='recall_macro', help='metric to optimize')
@click.option("--log-dir", default='/tmp')
def train_single_attention_classifier(features_path, labels_path, label_col, ng, random_state, regression, lr, metric, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    model_name = features_path.split('.')[0]
    logger.info(f'MODEL: {model_name}')
    labels = torch.load(labels_path)
    X = torch.load(features_path)
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    logger.info(f'data length: {len(X)}')

    def grid_search(lrs):
        results = []
        for lr in lrs:
            train_params = {"epochs": 100, "batch_size": 8, "lr": lr, "save": False, "regression": regression, "combined": False}
            r = train_n_fold(X=X, y=y, groups=groups, model_name=model_name, train_params=train_params, ng=ng,
                             random_state=random_state, start_time=start_time, exp_log=exp_log, log_dir=log_dir,
                             logger=logger, y_split=labels['type_numeric'])
            results.append((lr, r[metric].mean(), r[metric].std()))
        return min(results, key=lambda x: x[1]) if regression else max(results, key=lambda x: x[1])
    # lrs = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]
    lrs = [lr]
    best_lr, best_score, std = grid_search(lrs)
    print(f'MODEL: {model_name}')
    print("Best learning rate:", best_lr)
    print(f'Best {metric}: {best_score}+-{std}')


@cli.command()
@click.option('--features-path', required=True, help='path to extracted features')
@click.option('--labels-path', required=True, help='path to extracted labels')
@click.option('--label-col', default='type_numeric', help='label column name')
@click.option('--binary', is_flag=True, help='if to use as regression problem, default is classification')
@click.option("--ng", default=5, help='number of folds to split test data')
@click.option('--random-state', default=42, help='set random state for splitting')
@click.option("--log-dir", default='/tmp')
def train_single_svm_classifier(features_path, labels_path, label_col, binary, ng, random_state, log_dir):
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    # set model name
    model_name = features_path.split('/')[-1].split('.')[0]
    logger.info(f'MODEL: {model_name}')
    meta_data = load_meta_data_bix(Path(labels_path))
    features = pd.read_csv(features_path)
    # features['kb_prognose'] = features['kb_prognose'].replace(['JA', 'NEIN'], [1, 0])
    # features['prognose'] = features['prognose'].replace(['gesund', 'lk', 'ad'], [0, 1, 2])
    # features['prognosesicherheit'] = features['prognosesicherheit'].replace(['hoch', 'niedrig'], [1, 0])
    labels = meta_data.merge(features, on='subject', how='inner').sort_values(by=['subject']).reset_index(drop=True)
    y, groups = get_y_and_groups(labels=labels, label_col=label_col)
    if binary:
        y[y > 0] = 1
    # X = labels[['text_coherence', 'lexical_diversity', 'sentence_length', 'word_finding_difficulties']].to_numpy()
    # 'kb_prognose', 'prognosesicherheit'
    X = labels[['text_kohaerenz', 'lexikalische_vielfalt', 'satzlaenge', 'wortfindungsschwierigkeiten']].to_numpy()
    logger.info(f'data length: {len(X)}')
    train_n_fold_svm(X=X, y=y, groups=groups, label_col=label_col, model_name=model_name, pca=False, binary=binary, ng=ng,
                     nj=4, random_state=random_state, start_time=start_time, log_dir=log_dir, logger=logger, pos_label=1)


@cli.command()
@click.option('--text-dir', required=True, help='directory to text files')
@click.option('--labels-dir', required=True, help='directory to label file')
@click.option('--task', default='CERAD1',
              help='mc code of the task to use for classification, default is verbal fluency test')
@click.option('--model', default='bert-base-german-cased', help='required if embeddings is None')
@click.option('--train', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for training')
@click.option('--test', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for testing')
@click.option('--nj', default=4, help='number of parallel compute jobs')
@click.option("--log-dir", default='/tmp')
def extract_text_features_and_labels(text_dir, labels_dir, task, model, train, test, nj, log_dir):
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    model_name = f"{task}_{text_dir.split('/')[-2]}_train_{train}_test_{test}"
    # load metadata
    if train == "bix":
        meta_data = load_meta_data_bix(Path(labels_dir))
    elif train == "mc":
        meta_data = load_meta_data_mc(f'{Path(labels_dir)}/labels.csv')
    else:
        logger.info(f'No implemented method metadata loading yet for dataset {train}')
        exit()
    labels = dataloading(text_dir=text_dir, metadata=meta_data, task=task, dataset_name=train)
    text_embeddings, _ = feature_extraction_text_tokens(bert_model=model, labels=labels, pooling=None, nj=nj)
    save_features_and_labels(X=text_embeddings, y=labels, model_name=model_name, log_dir=log_dir, start_time=start_time, logger=logger)
    return log_dir


@cli.command()
@click.option('--text-dir', required=True, help='directory to text files')
@click.option('--audio-dir', required=True, help='directory to audio files')
@click.option('--task', default='CERAD1',
              help='mc code of the task to use for classification, default is verbal fluency test')
@click.option('--model', default='oliverguhr/wav2vec2-base-german-cv9', help='audio model name')
@click.option('--extract-layer', default=9, help='layer to take extracted audio embeddings from')
@click.option('--train', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for training')
@click.option('--test', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for testing')
@click.option('--nj', default=4, help='number of parallel compute jobs')
@click.option("--log-dir", default='/tmp')
def extract_audio_features_and_labels(text_dir, audio_dir, task, model, extract_layer, train, test, nj, log_dir):
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    model_name = f"{task}_{model.split('/')[-1]}_layer_{extract_layer}_train_{train}_test_{test}"
    # load metadata
    if train == "bix":
        meta_data = load_meta_data_bix(Path(audio_dir))
    elif train == "mc":
        meta_data = load_meta_data_mc(f'{Path(audio_dir).parent}/labels.csv')
    else:
        logger.info(f'No implemented method metadata loading yet for dataset {train}')
        exit()
    labels = dataloading(text_dir=text_dir, audio_dir=audio_dir, metadata=meta_data, task=task, dataset_name=train)
    audio_embeddings = feature_extraction_audio(w2v2_model=model, w2v2_extract_layer=extract_layer,
                                                labels=labels, pooling=None, normalize_audio=True, nj=nj, logger=logger)
    # list of features
    save_features_and_labels(X=audio_embeddings, y=labels, model_name=model_name, log_dir=log_dir, start_time=start_time, logger=logger)
    return log_dir


@cli.command()
@click.option('--audio-dir', required=True, help='directory to audio files')
@click.option('--task', default='CERAD1',
              help='mc code of the task to use for classification, default is verbal fluency test')
@click.option('--model', default='openai/whisper-large-v3', help='model for feature extraction')
@click.option('--extract-layer', default=-1, help='layer to take extracted embeddings from')
@click.option('--train', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for training')
@click.option('--test', type=click.Choice(['bix', 'mc', 'both']), default="mc", help='dataset for testing')
@click.option('--no-repeat-ngram', default=-1, help='maximum size of allowed ngram repetitions during decoding, '
                                                      'prevents whisper from getting stucked in repetition loops')
@click.option("--log-dir", default='/tmp')
def extract_enc_dec_features_and_labels(audio_dir, task, model, extract_layer, train, test, no_repeat_ngram, log_dir):
    log_dir = Path(log_dir)
    start_time = time.time()
    logger = logging.getLogger(f'py.{CLI_NAME}')
    logger.info('loading data')
    model_name = f"{task}_{model.split('/')[-1]}_train_{train}_test_{test}"
    # load metadata
    if train == "bix":
        meta_data = load_meta_data_bix(Path(audio_dir))
    elif train == "mc":
        meta_data = load_meta_data_mc(f'{Path(audio_dir).parent}/labels.csv')
    else:
        logger.info(f'No implemented method metadata loading yet for dataset {train}')
        exit()
    labels = dataloading(audio_dir=audio_dir, metadata=meta_data, task=task, dataset_name=train)
    # for looping only safe once for labels and transcriptions
    if no_repeat_ngram == -1:
        no_repeat_ngram = None
    if extract_layer == -1:
        transcriptions = feature_extraction_transcription(whisper_model=model, labels=labels, no_repeat_ngram_size=no_repeat_ngram)
        save_features_and_labels(X=transcriptions, y=labels, model_name=f"{model_name}_transcript", log_dir=log_dir,
                                 start_time=start_time, logger=logger)
    enc_embeddings = feature_extraction_encoder(whisper_model=model, labels=labels, extract_layer=extract_layer, no_repeat_ngram_size=no_repeat_ngram)
    save_features_and_labels(X=enc_embeddings, y=None, model_name=f"{model_name}_encoder_layer_{extract_layer}", log_dir=log_dir, start_time=start_time, logger=logger)
    dec_embeddings = feature_extraction_decoder(whisper_model=model, labels=labels, extract_layer=extract_layer, no_repeat_ngram_size=no_repeat_ngram)
    save_features_and_labels(X=dec_embeddings, y=None, model_name=f"{model_name}_decoder_layer_{extract_layer}", log_dir=log_dir, start_time=start_time, logger=logger)
    return log_dir


def dataloading(metadata, task, dataset_name, text_dir=None, audio_dir=None):
    # load data
    task = ['1.txt', '2.txt', '3.txt'] if task == "taukdial" else [task]
    if text_dir:
        data = load_text_data(text_dir, task)
    else:
        data = load_audio_data(audio_dir, task)

    if audio_dir and text_dir:
        audio = load_audio_data(audio_dir, task)
        data = data.merge(audio, on=['subject', 'File-ID'], how='inner').reset_index(drop=True)
    # merge and sort
    labels = metadata.merge(data, on='subject', how='inner').sort_values(by=['subject']).reset_index(drop=True)
    labels['dataset'] = dataset_name
    return labels


def get_cross_corpus_split(X, y, groups, train, test, labels):
    train_indices = labels.index[labels['dataset'] == train].tolist()
    test_indices = labels.index[labels['dataset'] == test].tolist()
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    return X_train, X_test, y_train, y_test, groups_train, groups_test, test_indices


def train_sklearn_svm_pipeline(X_train, y_train, pca=False, n_jobs=8, n_groups=5, binary=False, pos_label=0):
    pipeline_steps = []
    if pca:
        pipeline_steps.append(("pca", PCA()))
        parameters = [
            {
                "svc__kernel": ["rbf", "linear"],
                "pca__n_components": [16, 30, 60, 170],
                "svc__gamma": [0.1, 0.01, 0.001, 0.0001],
                "svc__C": [1, 10, 100, 1000],
                "svc__class_weight": ['balanced', None]
            }
        ]
    else:
        parameters = [
            {
                "svc__kernel": ["rbf", "linear"],
                "svc__gamma": [0.1, 0.01, 0.001, 0.0001, 0.00001],
                "svc__C": [0.1, 1, 10, 100],
                "svc__max_iter": [100000],
                # "svc__kernel": ["linear"],
                # "svc__gamma": [0.1],
                # "svc__C": [1],
            }
        ]

    pipeline_steps.append(("svc", SVC(class_weight='balanced')))
    pipe = Pipeline(steps=pipeline_steps)
    if binary:
        clf = GridSearchCV(pipe, parameters, n_jobs=n_jobs, cv=StratifiedKFold(n_splits=n_groups),
                           scoring=make_scorer(f1_score, average='binary', pos_label=pos_label), error_score='raise', verbose=2)
    else:
        clf = GridSearchCV(pipe, parameters, n_jobs=n_jobs, cv=StratifiedKFold(n_splits=n_groups),
                           scoring=make_scorer(recall_score, average='macro'), error_score='raise', verbose=2)

    clf.fit(X=X_train, y=y_train)

    return clf


def train_n_fold_svm(X, y, groups, label_col, model_name, binary, ng, nj, random_state, start_time, log_dir, logger, pos_label, pca=False):
    group_kfold = StratifiedGroupKFold(n_splits=ng, shuffle=True, random_state=random_state)
    fold = 1
    labels, predictions, indices = [], [], []

    results = {}
    for train_idxs, test_idxs in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_idxs], X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        clf = train_sklearn_svm_pipeline(X_train, y_train, pca=False, n_jobs=nj,  n_groups=ng, binary=binary,
                                         pos_label=pos_label)
        y_pred = clf.predict(X_test)
        results[fold] = get_results(y_test, y_pred, clf=clf, args={'fold': fold, 'test_len': len(test_idxs), 'pos_label': pos_label})
        labels = [*labels, *y_test]
        predictions = [*predictions, *y_pred]
        indices = [*indices, *test_idxs]

        logger.info(f'classification results for fold {fold}\n {classification_report(y_test, y_pred)}')
        fold = fold + 1

    results = pd.concat(results.values())
    end_time = dt.now().strftime("%m_%d_%Y_%H_%M_%S")
    classes = "binary" if binary else "multi"
    out_file = log_dir / f"{end_time}_{label_col}_{model_name}_{classes}.csv"
    results.to_csv(out_file, index=False)
    test_results = pd.DataFrame()
    test_results['label'] = labels
    test_results['predictions'] = predictions
    test_results['index'] = indices
    test_results.to_csv(out_file.with_suffix('.predictions.csv'))

    logger.info(f"Finished. Wrote results to {out_file}")
    end_time = time.time()
    logger.info("total time for grid search training: %s", end_time - start_time)


def get_results(y_test, y_pred, probs=None, args=None, clf=None, regression=False):
    results = {k: v for k, v in args.items()}
    if clf:
        results = {**results, **clf.best_params_, 'accuracy': accuracy_score(y_test, y_pred)}
    # fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
    # auc_score = auc(fpr, tpr)
    # , 'auc_score': auc_score

    if 'X_train' in results:
        results.pop('X_train')
    if 'X_test' in results:
        results.pop('X_test')
    if 'y_train' in results:
        results.pop('y_train')
    if 'y_test' in results:
        results.pop('y_test')
    if regression:
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = mean_squared_error(y_test, y_pred, squared=False)
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2'] = r2_score(y_test, y_pred)
    else:
        results['accuracy'] = accuracy_score(y_test, y_pred)

        results = {**results, **{f"recall_{i}": v for i, v in enumerate(recall_score(y_test, y_pred, average=None))}}
        results = {**results, **{f"precision_{i}": v for i, v in enumerate(precision_score(y_test, y_pred, average=None))}}
        results = {**results, **{f"f1_{i}": v for i, v in enumerate(f1_score(y_test, y_pred, average=None))}}

        results['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        results['recall_micro'] = recall_score(y_test, y_pred, average='micro')

        results['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        results['precision_micro'] = precision_score(y_test, y_pred, average='micro')

        results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        results['f1_micro'] = f1_score(y_test, y_pred, average='micro')

    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))


def save_features_and_labels(X, y, model_name, log_dir, start_time, logger):
    os.makedirs(Path(log_dir), exist_ok=True)
    if y is not None:
        out_file_y = Path(log_dir) / f"{model_name}_labels.pkl"
        torch.save(y, out_file_y)
        logger.info(f"Finished. Wrote labels to {out_file_y}")
    if X is not None:
        out_file_X = Path(log_dir) / f"{model_name}.pkl"
        torch.save(X, out_file_X)
        logger.info(f"Finished. Wrote features to {out_file_X}")
    end_time = time.time()
    logger.info("total time for feature extraction: %s", end_time - start_time)


def pytorch_training_pipeline(model, train_loader, test_loader, optimizer, loss_fn, train_params, logger):
    best_loss = np.inf
    best_pred = None

    # Early stopping setup
    early_stopping = EarlyStopping(patience=5, verbose=False, path=train_params['checkpoint'], save=train_params['save'])

    for epoch in range(train_params['epochs']):
        ### Training
        model.train()
        total_loss = 0  # Track total training loss for this epoch
        with tqdm(train_loader, unit="batch") as bar:
            bar.set_description(f"Epoch {epoch}")
            for train_batch in bar:
                optimizer.zero_grad()
                y_batch = train_batch.pop()
                logits = model(*train_batch).squeeze(dim=-1)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                bar.set_postfix(loss=loss.item(), epoch_loss=total_loss / len(train_loader))

        # Evaluate after each epoch
        model.eval()
        test_loss, predictions = [], []
        with torch.no_grad():
            for test_batch in test_loader:
                t_y_batch = test_batch.pop()
                t_logits = model(*test_batch).squeeze(dim=-1)
                t_loss = loss_fn(t_logits, t_y_batch)
                predictions.extend(t_logits.cpu().numpy())
                test_loss.append(t_loss.item())

            avg_test_loss = np.mean(test_loss)
            logger.info(f'Epoch {epoch} - Test Loss: {avg_test_loss}')

            # Check for early stopping
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                best_pred = predictions
                logger.info(f'Saved new best model with test loss: {best_loss}')

            early_stopping(avg_test_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    return best_pred, best_loss


def train_n_fold(X, y, groups, model_name, train_params, ng, random_state, start_time, exp_log, log_dir, logger, y_split=None, norm=None):
    # environment
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # split
    group_kfold = StratifiedGroupKFold(n_splits=ng, shuffle=True, random_state=random_state)
    # cross validation
    fold = 1
    results = {}
    labels, predictions, indices, folds = [], [], [], []
    if y_split is None: y_split = y
    for train_idxs, test_idxs in group_kfold.split(X, y_split, groups):
        X_train, X_test = [X[i] for i in train_idxs], [X[i] for i in test_idxs]
        y_train, y_test = y[train_idxs], y[test_idxs]
        train_params['checkpoint'] = f'{log_dir}/{model_name}_fold_{fold}.pt'
        # training parameters
        classes = np.unique(y_train)
        if train_params['regression']:
            loss_fn = nn.MSELoss()
            # loss_fn = nn.SmoothL1Loss()
            dtype = torch.float32
            out_dim = 1
        else:
            weight = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            weight = torch.tensor(weight, dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))
            dtype = torch.long
            out_dim = len(classes)
        if train_params['combined']:  # multimodal
            # Load pretrained models
            f_dim = X[-1][-1][-1].shape[-1]
            subtest_models = []
            for test in train_params['tests']:
                path = f'{train_params["models_path"]}{test}_{"_".join(model_name.split("_")[-8:])}_fold_{fold}.pt'
                pretrained_model = ScalarFusionModel(f_dim=f_dim, o_dim=out_dim)
                pretrained_model.load_state_dict(torch.load(path))
                pretrained_model.to(device)
                # Freeze all layers initially
                # for param in pretrained_model.parameters():
                #     param.requires_grad = True
                # Unfreeze the last two layers (fusion_fc and output_fc)
                # for param in pretrained_model.fusion_fc.parameters():
                #     param.requires_grad = True
                # for param in pretrained_model.output_fc.parameters():
                #     param.requires_grad = True
                pretrained_model.eval()  # Set model to evaluation mode
                subtest_models.append(pretrained_model)

            tests = torch.tensor([int(test[-1]) for test in train_params['tests']], dtype=torch.float32)
            model = NormScalarFusionModel(subtest_models=subtest_models, o_dim=out_dim, norm=norm.to(device), tests=tests.to(device))  # model
            # dataset
            train_dataset, test_dataset = NormDataset(X=X_train, y=y_train, device=device, dtype=dtype), \
                                          NormDataset(X=X_test, y=y_test, device=device, dtype=dtype)
        else:  # unimodal
            f_dim = X[-1][-1].shape[-1]
            if len(X[0]) == 3:
                model = CrossEmbeddingModel(f_dim=f_dim, o_dim=out_dim)  # model hidden_dims=train_params['hidden_dims'],
            else:
                model = ScalarFusionModel(f_dim=f_dim, o_dim=out_dim, hidden_dims=train_params['hidden_dims'])
            # model = AttentionFusionModel(feature_dim=768)
            # dataset
            train_dataset, test_dataset = CombinedDataset(X=X_train, y=y_train, device=device, dtype=dtype), \
                                          CombinedDataset(X=X_test, y=y_test, device=device, dtype=dtype)
        # dataloader
        # train_batch_sampler = StratifiedBatchSampler(y=y_train, batch_size=train_params['batch_size'])
        train_loader, test_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size']), \
                                    DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'])
        logger.info(f'created model: {model}')
        model.to(device)
        optimizer = Adam(model.parameters(), lr=train_params['lr'])
        y_pred, loss = pytorch_training_pipeline(model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, loss_fn=loss_fn, train_params=train_params, logger=logger)

        results[fold] = get_results(y_test, y_pred, args={'fold': fold, 'test_len': len(test_idxs), 'loss': loss}, regression=train_params['regression'])
        labels = [*labels, *y_test]
        predictions = [*predictions, *y_pred]
        indices = [*indices, *test_idxs]
        folds = [*folds, *np.full(len(y_test), fold)]
        # classification_report(y_test, y_pred)
        logger.info(f'results for fold {fold}\n and model {model_name} \n {results[fold]}')
        if not train_params['regression']:
            logger.info(f'classification report \n {classification_report(y_test, y_pred)}')
        fold = fold + 1

    results = pd.concat(results.values())
    exp_log['task_name'] = model_name.split('_')[0]
    results = pd.concat([results, pd.DataFrame(exp_log, index=[0]).reset_index(drop=True)], axis=1)
    end_time = dt.now().strftime("%m_%d_%Y_%H_%M_%S")
    out_file = f"{log_dir}/{end_time}_{model_name}.csv"
    results.to_csv(out_file, index=False)
    test_results = pd.DataFrame()
    test_results['label'] = labels
    test_results['predictions'] = predictions
    test_results['index'] = indices
    test_results['fold'] = folds
    test_results.to_csv(Path(out_file).with_suffix('.predictions.csv'))
    logger.info(f"Finished. Wrote results to {out_file}")

    logger.info(f'total results for 5-folds LOSS {results["loss"].mean()}')
    # logger.info(f'total results for 5-folds UAR {results["recall_macro"].mean()}')
    # logger.info(f'total results for 5-folds F1_macro {results["f1_macro"].mean()}')
    end_time = time.time()
    logger.info("total time for trial search training: %s", end_time - start_time)
    return results


def get_ig_attributions_and_errors(model, X_test, y_test, device):
    ig_attributions = []
    ig_attributions_errors = []
    baseline1 = torch.zeros(1, len(X_test[0][0])).to(device)
    baseline2 = torch.zeros(1, len(X_test[1][0])).to(device)
    baseline3 = torch.zeros(1, len(X_test[2][0])).to(device)

    def ig_forward_func(x1,x2,x3):
        res, _ = model([x1,x2,x3])
        probs = torch.softmax(res, dim=1)
        return probs

    ig = IntegratedGradients(ig_forward_func)

    for i in range(len(X_test[0])):
        x = (torch.unsqueeze(X_test[0][i], 0), torch.unsqueeze(X_test[1][i], 0), torch.unsqueeze(X_test[2][i], 0))
        attribution, approximation_error = ig.attribute(x,
                                                        target=y_test[i],
                                                        baselines=(baseline1, baseline2, baseline3),
                                                        method='gausslegendre',
                                                        return_convergence_delta=True)
        attribution = [att.cpu().squeeze().numpy() for att in attribution]
        ig_attributions.append(attribution)
        ig_attributions_errors.append(approximation_error.cpu().squeeze().numpy())
    return ig_attributions, ig_attributions_errors


@cli.command()
@click.option('--file-dir', required=True)
@click.option("--log-dir", default='/tmp/')
def preprocess_data(file_dir, log_dir):
    wavs = [w for w in Path(file_dir).glob('**/*.wav')]
    # for w in wavs:
    #     split = w.stem.split('_')
    #     new_name = str(w.parent) + "/" + split[1].strip() + "_" + split[0].strip() + w.suffix
    #     os.rename(w, new_name)
    #
    # text = [t for t in Path(file_dir).glob('**/*.txt')]
    # for t in text:
    #     split = t.stem.split('_')
    #     new_name = str(t.parent) + "/" + split[1].strip() + "_" + split[0].strip() + t.suffix
    #     os.rename(t, new_name)

    for w in wavs:
        name = w.stem.strip()
        ext = w.suffix
        parent = w.parent.name
        if name[-1] == "I":
            new_name = log_dir + "instructions/" + "nsc_" + parent + "_" + name[:-1] + "_I" + ext
        else:
            new_name = log_dir + "tests/" + "nsc_" + parent + "_" + name + ext
        shutil.copy(w, new_name)

    # text = [t for t in Path(file_dir).glob('**/*.txt')]
    # for t in text:
    #     name = t.stem.strip()
    #     ext = t.suffix
    #     parent = t.parent.name
    #     new_name = log_dir + "tests/" + parent + "_" + name + ext
    #     new_name = new_name.replace("I.", "_I.")
    #     shutil.copy(t, new_name)
