import argparse
import json
import os
import urllib.parse
from glob import glob

import pandas as pd


def process_title(title):
    return urllib.parse.quote(title.replace(' ', '_'))


def extract_dump(df):
    pages = {}
    for i in range(len(df)):
        if not pd.isna(df['main_entity'][i]):
            page_info = {}
            page_info['title'] = process_title(df['name'][i])
            page_info['ID'] = df['identifier'][i]
            page_info['language'] = df['in_language'][i]['identifier']
            page_info['version'] = f"https://{page_info['language']}.wikipedia.org/w/index.php?title={page_info['title']}&oldid={df['version'][i]['identifier']}"
            page_info['QID'] = df['main_entity'][i]['identifier']
            page_info['HTML'] = df['article_body'][i]
            pages[page_info['title']] = page_info

    return pages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Path to the data folder')

    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    parser.add_argument('--chunksize', type=int, default=10_000,
                        help='Chunksize for reading the json files')
    args = parser.parse_args()

    # check if input dir exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # check if chunksize is positive
    if args.chunksize <= 0:
        raise ValueError(f"Chunksize must be positive, got {args.chunksize}")

    # Read all json files
    files = glob(f"{args.input_dir}/*.ndjson")
    simple_pages = {}
    counter = 0
    for file in files:
        print(f"Processing {file}")
        df = pd.read_json(file, chunksize=args.chunksize, lines=True)
        for chunk in df:
            pages = extract_dump(chunk)
            for page in pages:
                simple_pages[page] = {'ID': pages[page]
                                      ['ID'], 'QID': pages[page]['QID']}
            with open(f"{args.output_dir}/pages_{counter}.json", 'w') as f:
                json.dump(pages, f)
            counter += 1
    with open(f"{args.output_dir}/simple_pages.json", 'w') as f:
        json.dump(simple_pages, f)
