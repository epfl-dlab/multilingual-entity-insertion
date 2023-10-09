import os
from glob import glob

import pandas as pd
import torch.utils.data as data


class WikiDataset(data.Dataset):
    def __init__(self, data_dir, split):
        super(WikiDataset, self).__init__()
        self.data = self.get_data(data_dir, split)

    def get_data(self, data_dir, split):
        data_dir = os.path.join(data_dir, split)
        files = glob(os.path.join(data_dir, '*.parquet'))
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            dfs.append(df)
        df = pd.concat(dfs)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
