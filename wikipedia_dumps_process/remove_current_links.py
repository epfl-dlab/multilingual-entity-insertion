import argparse
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # check if input dir exists
    if not os.path.exists(args.input_dir):
        raise Exception('Input dir does not exist')
    
    files = glob(os.path.join(args.input_dir, '*.parquet'))
    for file in tqdm(files):
        removed = False
        df = pd.read_parquet(file)
        for column in df.columns:
            if 'current_links' in column:
                df = df.drop(columns=[column])
                removed = True
        if not removed:
            break
        df.to_parquet(file)
