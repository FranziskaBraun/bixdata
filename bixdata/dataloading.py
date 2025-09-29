import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
# from scipy.io import wavfile
import torch
from torch.utils.data import Dataset, DataLoader
import re
from .constants import *
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pad_sequence


def code_to_task(code: str):
    return TASK_MAPPING[code]


def task_to_code(task: str):
    return {v: k for k, v in TASK_MAPPING.items()}[task]


def load_meta_data(base_path: Path):
    """Load data from the base_path.

    Args:
        base_path (Path): Path to the data folder.

    Returns:
        pd.DataFrame: Dataframe containing the meta data and file mappings
    """

    meta = []
    ass = []
    base_path = Path(base_path) if isinstance(base_path, str) else base_path
    for file in base_path.glob('**/*.json'):
        if 'storagedata' in file.name:
            # skip storage data files, they are not needed, seems like files(name wise)
            # for assessments that were not completed
            continue
        if 'meta' in file.name:
            tmp = pd.read_json(file, orient='index').T
            tmp['file'] = file.name
            meta.append(tmp)
        else:
            tmp = pd.read_json(file, orient='index').T
            tmp['file'] = file.name
            ass.append(tmp)

    df_meta = pd.concat(meta).reset_index(drop=True)
    df_ass = pd.concat(ass).reset_index(drop=True)

    # dropping columns with no lables
    df_meta['drop'] = df_meta['type'].replace('', np.nan).isna() & df_meta['type_numeric'].isna()
    df_meta = df_meta[~df_meta['drop']]
    # mapping str labels for cols that have either string or int label
    str_to_lbl = {'control': 2, 'patient_dat': 1, 'patient_mci': 0}
    lbl_to_str = {v: k for k, v in str_to_lbl.items()}
    df_meta['type'] = df_meta[['type', 'type_numeric']].apply(
        lambda x: lbl_to_str[x['type_numeric']] if x['type'] == '' else x['type'], axis=1)
    df_meta['type_numeric'] = df_meta[['type', 'type_numeric']].apply(
        lambda x: str_to_lbl[x['type']] if x['type'] == np.nan else x['type_numeric'], axis=1)
    df_meta['type_numeric'] = df_meta['type_numeric'].astype(int)

    # map so file names can be used for joining the dataframes
    # df_meta['subject'] = df_meta['file'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    # df_ass['subject'] = df_meta['file'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    df_meta['subject'] = df_meta['file'].apply(lambda x: int(x.split('_')[1]))
    df_ass['subject'] = df_meta['file'].apply(lambda x: int(x.split('_')[1]))

    # fill out missing diagnosis
    df_meta['informed_consent'].fillna(False, inplace=True)
    df_meta['full_legal_competence'].fillna(False, inplace=True)
    df_meta['diagnosis_of_mci'].fillna(False, inplace=True)
    df_meta['diagnosis_of_AD'].fillna(False, inplace=True)
    df_meta['diagnosis_excludes_dementia'].fillna(False, inplace=True)
    df_meta['no_impairments_german'].fillna(True, inplace=True)
    df_meta['no_impairments_vision_and_hearing'].fillna(True, inplace=True)
    df_meta['no_impairments_speech'].fillna(True, inplace=True)
    df_meta['german_native_speaker'].fillna(True, inplace=True)
    df_meta['no_impairments_speech'].fillna(True, inplace=True)
    df_meta['gender'].fillna('d', inplace=True)
    df_meta['age'].fillna(-1, inplace=True)
    df_meta['age'] = df_meta['age'].astype('int64')
    df_meta['mmse_value_numeric'] = df_meta['mmse_value'].apply(lambda x: -1 if x == '' else int(x))

    return df_meta, df_ass


def load_meta_data_bix(base_path: Path):
    meta_data, _ = load_meta_data(base_path)
    # make bix labels intuitive 2=HC=0 0=MCI=1 1=DEM=2
    meta_data['type_numeric'] = meta_data['type_numeric'].replace([2, 0, 1], [0, 1, 2])
    meta_data = meta_data[['subject', 'gender', 'age', 'type_numeric', 'diagnosis_of_mci', 'diagnosis_of_AD']].reset_index(drop=True)

    return meta_data


def load_meta_data_mc(base_path: Path, label_col='GDS1_mul4'):
    """Load data from the base_path.

    Args:
        base_path (Path): Path to the data folder.

    Returns:
        pd.DataFrame: Dataframe containing the meta data and file mappings
    """
    meta_data = pd.read_csv(base_path, sep=';')
    meta_data['subject'] = meta_data['Nr']
    meta_data['gender'] = meta_data['Geschlecht'].replace(['w', 'm'], ['female', 'male'])
    meta_data['age'] = meta_data['Alter']
    meta_data['education'] = meta_data['Ausbildungsjahre']
    # SKT metadata
    meta_data['SKT'] = meta_data['SKT-Typ']
    meta_data['SKTT'] = meta_data['SKT-Total']
    meta_data['SKTA'] = meta_data['SKT-A']
    meta_data['SKTG'] = meta_data['SKT-G']
    meta_data['SKT1'] = meta_data['SKT-1']
    meta_data['SKT2'] = meta_data['SKT-2']
    meta_data['SKT3'] = meta_data['SKT-3']
    meta_data['SKT4'] = meta_data['SKT-4']
    meta_data['SKT5'] = meta_data['SKT-5']
    meta_data['SKT6'] = meta_data['SKT-6']
    meta_data['SKT7'] = meta_data['SKT-7']
    meta_data['SKT8'] = meta_data['SKT-8']
    meta_data['SKT9'] = meta_data['SKT-9']
    # CERAD metadata
    meta_data['CERAD1'] = meta_data['CERAD-1']
    meta_data['CERAD2'] = meta_data['CERAD-2']
    meta_data['MMSE'] = meta_data['CERAD-3']
    meta_data['CERADTR'] = meta_data['CERAD-T-RAW']
    meta_data['CERADTN'] = meta_data['CERAD-T-NORM']    # diagnosis metadata
    meta_data['type_numeric'] = meta_data[label_col]
    meta_data['diagnosis'] = meta_data['Diagnose_Med']
    meta_data['depression'] = meta_data['Dep_mul3']

    return meta_data[['subject', 'gender', 'age', 'education', 'IQ', 'ICD', 'type_numeric', 'diagnosis', 'depression',
                      'SKT', 'SKTT', 'SKTG', 'SKTA', 'SKT1', 'SKT2', 'SKT3', 'SKT4', 'SKT5', 'SKT6', 'SKT7', 'SKT8', 'SKT9',
                      'CERAD1', 'CERAD2', 'MMSE', 'SKT1N', 'SKT2N', 'SKT3N', 'SKT4N', 'SKT5N', 'SKT6N', 'SKT7N', 'SKT8N', 'SKT9N', 'CERADTR', 'CERADTN']].reset_index(drop=True)


def load_meta_data_brk(base_path: Path, label_col='Gruppe'):
    """Load data from the base_path.

    Args:
        base_path (Path): Path to the data folder.

    Returns:
        pd.DataFrame: Dataframe containing the meta data and file mappings
    """
    meta_data = pd.read_csv(base_path, sep=';')
    meta_data['subject'] = meta_data['Nr']
    meta_data['gender'] = meta_data['Geschlecht'].replace(['w', 'm'], ['female', 'male'])
    meta_data['age'] = meta_data['Alter']
    meta_data['CERAD1'] = meta_data['CERAD-1']
    meta_data['CERAD2'] = meta_data['CERAD-2']
    meta_data['type_numeric'] = meta_data[label_col].replace(['HC', 'DEM'], [0, 1])
    meta_data['diagnosis'] = meta_data['Diagnose_Text']
    meta_data['depression'] = meta_data['Diagnose_Dep']
    return meta_data[['subject', 'gender', 'age', 'type_numeric', 'diagnosis', 'depression', 'CERAD1', 'CERAD2']].reset_index(drop=True)


def load_meta_data_tkd(base_path: Path, label_col='dx'):
    """Load data from the base_path.

    Args:
        base_path (Path): Path to the data folder.

    Returns:
        pd.DataFrame: Dataframe containing the meta data and file mappings
    """
    meta_data = pd.read_csv(base_path)
    tkdname = meta_data['tkdname'].str.split('[-.]', expand=True)
    meta_data['subject'] = tkdname[1].astype('int64')
    meta_data['gender'] = meta_data['sex'].replace(['F', 'M'], ['female', 'male'])
    meta_data['type_numeric'] = meta_data[label_col].replace(['NC', 'MCI'], [0, 1])
    columns = ['subject', 'gender', 'age', 'type_numeric']
    return meta_data[columns].groupby(columns).first().reset_index()


def load_meta_data_dlw(base_path: Path, label_col='label'):
    """Load data from the base_path.

    Args:
        base_path (Path): Path to the data folder.

    Returns:
        pd.DataFrame: Dataframe containing the meta data and file mappings
    """
    meta_data = pd.read_csv(base_path)
    meta_data['subject'] = meta_data['RECORD ID'].astype('int64')
    meta_data['type_numeric'] = meta_data[label_col].astype('int64')
    return meta_data[['subject', 'type_numeric']].reset_index(drop=True)


def load_audio_data(base_path: Union[Path, str], tests, filter_incomplete=True, read_sec=60):
    base_path = Path(base_path) if isinstance(base_path, str) else base_path
    wavs = [w for w in base_path.glob('**/*.wav')]
    files = pd.DataFrame()
    for val in tests:
        tmp_files = [w for w in wavs if val in w.name]
        tmp_wav_info = [pd.DataFrame(
            {'File-ID': val,
             'subject': int(re.split('_', w.stem)[1]),
             'audio_path': str(w)
             # 'audio': [wavfile.read(str(w))[1][:min(wavfile.read(str(w))[0] * read_sec, len(wavfile.read(str(w))[1]))].tolist()],

             }, index=[0]) for w in tmp_files
        ]
        files = pd.concat([files, *tmp_wav_info])
    files = files.reset_index(drop=True)

    if filter_incomplete:
        indexer = files.groupby('subject')['File-ID'].count() < len(files['File-ID'].unique())
        files = files[~files['subject'].isin(indexer[indexer].index)].reset_index(drop=True)

    return files


def load_text_data(base_path: Union[Path, str], tests, filter_incomplete=True):
    base_path = Path(base_path) if isinstance(base_path, str) else base_path
    txts = [t for t in base_path.glob('**/*.txt')]
    files = pd.DataFrame()
    for val in tests:
        tmp_files = [t for t in txts if val in t.name]
        # there should be > 200 files per test, but not all tests for all subjects are available
        tmp_txt_info = [pd.DataFrame(
            {'File-ID': val,
             'subject':  int(re.split('_', t.stem)[1]),
             'text': load_text_file(str(t)) + ' '
             }, index=[0]) for t in tmp_files
        ]
        files = pd.concat([files, *tmp_txt_info])
    files = files.reset_index(drop=True)

    if filter_incomplete:
        indexer = files.groupby('subject')['File-ID'].count() < len(files['File-ID'].unique())
        files = files[~files['subject'].isin(indexer[indexer].index)].reset_index(drop=True)

    return files


def load_text_file(path):
    with open(path) as f:
        text = " ".join(f.readlines()).replace("\n", "")
        return text


def load_bix_text_bnt(data_dir_bix_text):
    text_bix = load_text_data(data_dir_bix_text, BNT)
    text_bix['File-ID'] = pd.Categorical(text_bix['File-ID'], categories=BNT)
    text_bix = text_bix.sort_values(by=['subject', 'File-ID']).reset_index(drop=True)
    text_bix['File-ID'] = text_bix['File-ID'].astype(str)
    text_bix = text_bix.groupby('subject').agg({'text': 'sum'}).reset_index()
    text_bix['File-ID'] = 'bnt'
    return text_bix


def load_bix_audio_bnt(data_dir_bix_audio):
    audio_bix = load_audio_data(data_dir_bix_audio, BNT)
    audio_bix['File-ID'] = pd.Categorical(audio_bix['File-ID'], categories=BNT)
    audio_bix = audio_bix.sort_values(by=['subject', 'File-ID']).reset_index(drop=True)
    audio_bix['File-ID'] = audio_bix['File-ID'].astype(str)
    audio_bix = audio_bix.groupby('subject').agg({'audio': 'sum'}).reset_index()
    return audio_bix


class CombinedDataset(Dataset):
    def __init__(self, X, y, device, dtype=torch.long):
        self.dtype = dtype
        self.X = X
        self.y = y
        self.device = device

    def __len__(self):
        # return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # return one sample from the dataset
        features = [x.to(self.device) for x in self.X[idx]]
        target = torch.tensor(self.y[idx], dtype=self.dtype).to(self.device)
        return *features, target


class NormDataset(Dataset):
    def __init__(self, X, y, device, dtype=torch.long):
        self.dtype = dtype
        self.X = X
        self.y = y
        self.device = device

    def __len__(self):
        # return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # return one sample from the dataset
        x = self.X[idx]
        age = torch.tensor(x[0], dtype=self.dtype).to(self.device)
        iq = torch.tensor(x[1], dtype=self.dtype).to(self.device)
        scalars = [x.to(self.device) for x in x[2]]
        features = [x.to(self.device) for x in x[3]]
        target = torch.tensor(self.y[idx], dtype=self.dtype).to(self.device)
        return age, iq, scalars, features, target


class SingleDataset(Dataset):
    def __init__(self, X, y, dtype=torch.long, one_hot=False, num_classes=2):
        X = [torch.tensor(element, dtype=torch.float32) for element in X]
        self.X = pad_sequence(X, batch_first=True, padding_value=0.0)
        self.y = torch.tensor(y, dtype=dtype)
        if one_hot:
            self.y = F.one_hot(self.y, num_classes=num_classes)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target


class SingleDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=None, batch_sampler=None, dtype=torch.float32):
        self.dtype = dtype
        if batch_size:
            super().__init__(dataset, batch_size=batch_size, collate_fn=self.custom_collate)
        else:
            super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=self.custom_collate)

    def generate_attention_mask(self, x):
        # Create a binary mask with shape (batch_size, padded_sequence_length, padded_sequence_length)
        mask = (x[:, :, 0] != 0).float()
        mask = [m * torch.transpose(m.unsqueeze(dim=0), 0, 1) for m in mask]
        mask = torch.stack(mask)
        return mask

    def custom_collate(self, batch):
        features, labels = zip(*batch)
        features = torch.stack(features, dim=0)
        labels = torch.tensor(labels, dtype=self.dtype)
        # attention_mask = self.generate_attention_mask(features)
        return features, labels