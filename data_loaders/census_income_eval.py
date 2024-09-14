import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import logging

# 패키지 초기화 함수 불러오기
from __init__ import find_datasets_path

if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class Census_Income(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.dataset_directory, self.dataset_path = find_datasets_path(file_name="test_dataset")

        df = pd.read_csv(self.dataset_path)

        logging.debug(
            "\n" + "\n" + " ########################" + \
            "\n" + " ### start preprocess ###" + \
            "\n" + " ########################" + "\n" + \
            "\n" + " Number of Data : {}".format(str(len(df)))+"\n"
        )

        self.questions = df.iloc[:, 0:len(df.columns) - 1]
        self.responses = df.iloc[:, -1:]

        self.length = len(self.questions)

        with open(os.path.join(self.dataset_directory, "field_dict.pkl"), "rb") as f:
            self.field_dict = pickle.load(f)
        with open(os.path.join(self.dataset_directory, "field_index.pkl"), "rb") as f:
            self.field_index = pickle.load(f)
            
    def __getitem__(self, index):
        return FloatTensor(self.questions.loc[index]), FloatTensor(np.array(self.responses.loc[index]))

    def __len__(self):
        return self.length
