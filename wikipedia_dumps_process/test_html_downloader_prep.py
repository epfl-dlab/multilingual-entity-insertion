import pandas as pd
import subprocess
import argparse
import os

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
    
    clean_data = set([])
    for row in df:
        source_ID = row['source_ID']
        first_version = row['first_version']
        second_version = row['second_version']
        if f"{source_ID}\t{first_version}" not in clean_data:
            clean_data.add(f"{source_ID}\t{first_version}")
        if f"{source_ID}\t{second_version}" not in clean_data:
            clean_data.add(f"{source_ID}\t{second_version}")
    
    # write output to file
    with open(os.path.join(args.output_directory, 'pages.txt'), 'w') as f:
        for item in clean_data:
            f.write(f"{item}\n")