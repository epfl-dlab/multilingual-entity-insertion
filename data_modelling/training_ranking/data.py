import os
from glob import glob

import pandas as pd
import torch.utils.data as data


class WikiDataset(data.Dataset):
    def __init__(self, data_dir, split, neg_samples):
        super(WikiDataset, self).__init__()
        self.data = self.get_data(data_dir, split)
        self.split = split
        self.neg_samples = neg_samples

    def get_data(self, data_dir, split):
        data_dir = os.path.join(data_dir, split)
        files = glob(os.path.join(data_dir, '*.parquet'))
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            dfs.append(df)
        df = pd.concat(dfs)
        # negative columns have "_neg_{i}" suffix
        # remove all columns with i >= neg_samples
        remove_cols = []
        for column in df:
            if '_neg_' in column:
                if int(column.split('_')[-1]) >= self.neg_samples:
                    remove_cols.append(column)
        df = df.drop(remove_cols, axis=1)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index].to_dict()
        for key in data:
            if 'index' in key:
                data[key] = int(data[key])
        data['split'] = self.split
        return data
