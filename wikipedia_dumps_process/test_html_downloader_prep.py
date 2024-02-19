import pandas as pd
import subprocess
import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input file with page versions')
    parser.add_argument('--output_directory', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # check if input file exists
    if not os.path.exists(args.input_file):
        raise Exception('Input file does not exist')
    # check if output directory exists
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    # read input file
    df = pd.read_parquet(args.input_file).to_dict('records')
    
    data = []
    for row in tqdm(df):
        source_ID = row['source_ID']
        first_version = row['first_version']
        second_version = row['second_version']
        data.append(f"{source_ID}\t{first_version}")
        data.append(f"{source_ID}\t{second_version}")
    data = list(set(data))
    clean_data = []
    print('Finding existing files')
    existing_files = os.listdir(args.output_directory)
    existing_files = set(existing_files)
    for elem in tqdm(data):
        source_ID, version = elem.split('\t')
        if f'{source_ID}_{version}.html' in existing_files:
            continue
        clean_data.append(elem)

    # write output to file
    with open(os.path.join(args.output_directory, 'pages.txt'), 'w') as f:
        for item in clean_data:
            f.write(f"{item}\n")