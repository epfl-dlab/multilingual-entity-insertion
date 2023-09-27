import argparse
import json
import os
import urllib.parse
from glob import glob

import pandas as pd
from tqdm import tqdm


def process_title(title):
    return urllib.parse.quote(title.replace(' ', '_'))


def extract_dump(df, language):
    pages = []
    redirect_map = {}
    for i in range(len(df)):
        # get the page details
        page_info = {}
        page_info['title'] = process_title(df['name'][i])
        page_info['ID'] = df['identifier'][i]
        page_info['language'] = language
        page_info['version'] = f"https://{language}.wikipedia.org/w/index.php?title={page_info['title']}&oldid={df['version'][i]['identifier']}"
        if not pd.isna(df['main_entity'][i]):
            page_info['QID'] = df['main_entity'][i]['identifier']
        else:
            page_info['QID'] = None
        page_info['HTML'] = df['article_body'][i]['html']
        pages.append(page_info)

        # get the redirects
        if type(pd.isna(df['redirects'][i])) != bool:
            for redirect in df['redirects'][i]:
                redirect_map[process_title(
                    redirect['name'])] = page_info['title']

    return pages, redirect_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Path to the data folder')
    parser.add_argument('--language', type=str,
                        required=True, help='Language version of the Wikipedia dump')
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
    redirect_map = {}
    counter = 0
    for file in files:
        print(f"Processing {file}")
        df = pd.read_json(file, chunksize=args.chunksize, lines=True)
        for chunk in tqdm(df):
            chunk = chunk.reset_index(drop=True)
            pages, partial_redirect = extract_dump(chunk, args.language)
            for page in pages:
                simple_pages[page['title']] = {
                    'ID': page['ID'], 'QID': page['QID']}
            redirect_map.update(partial_redirect)
            # create dataframe from pages
            df = pd.DataFrame(pages)
            df.to_parquet(f"{args.output_dir}/pages_{counter}.parquet")
            counter += 1
    # create dataframe from simple pages
    df = pd.DataFrame.from_dict(simple_pages, orient='index')
    df.to_parquet(f"{args.output_dir}/simple_pages.parquet")
    # create dataframe from redirect map
    df = pd.DataFrame.from_dict(redirect_map, orient='index')
    df.columns = ['redirect_target']
    df.to_parquet(f"{args.output_dir}/redirect_map.parquet")
