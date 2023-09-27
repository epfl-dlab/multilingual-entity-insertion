import argparse
import json
import os
import urllib.parse
from glob import glob
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm


def process_title(title):
    title = title.replace(' ', '_')
    # decode from url encoding
    title = urllib.parse.unquote(title)
    # reencode into url encoding
    title = urllib.parse.quote(title)
    return title


def extract_dump(data):
    redirect_map = {}
    page_info = {}
    page_info['title'] = process_title(data['name'])
    page_info['ID'] = data['identifier']
    page_info['language'] = data['language']
    page_info['version'] = f"https://{data['language']}.wikipedia.org/w/index.php?title={page_info['title']}&oldid={data['version']['identifier']}"
    if not pd.isna(data['main_entity']):
        page_info['QID'] = data['main_entity']['identifier']
    else:
        page_info['QID'] = None
    page_info['HTML'] = data['article_body']['html']
    
    # get the redirects
    if type(pd.isna(data['redirects'])) != bool:
        for redirect in data['redirects']:
            redirect_map[process_title(
                redirect['name'])] = page_info['title']

    return page_info, redirect_map


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
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes to use for multiprocessing')
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
    full_pages = []
    redirect_map = {}
    counter = 0
    for file in tqdm(files):
        print(f"Processing {file}")
        df = pd.read_json(file, chunksize=args.chunksize, lines=True)
        
        for chunk in tqdm(df):
            list_data = []
            for i in range(len(chunk)):
                list_data.append(chunk.iloc[i].to_dict())
                list_data[-1]['language'] = args.language
            # use list data with pooling
            pool = Pool(min(cpu_count(), args.processes))
            for page, partial_redirect in tqdm(pool.imap_unordered(extract_dump, list_data), total=len(list_data)):
                full_pages.append(page)
                simple_pages[page['title']] = {'ID': page['ID'], 'QID': page['QID']}
                redirect_map.update(partial_redirect)
            # terminate pool
            pool.terminate()
            pool.join()
            # create dataframe from full_pages
            df = pd.DataFrame(full_pages)
            df.to_parquet(f"{args.output_dir}/pages_{counter}.parquet")
            counter += 1
    # create dataframe from simple pages
    df = pd.DataFrame.from_dict(simple_pages, orient='index')
    df.to_parquet(f"{args.output_dir}/simple_pages.parquet")
    # create dataframe from redirect map
    df = pd.DataFrame.from_dict(redirect_map, orient='index')
    df.columns = ['redirect_target']
    df.to_parquet(f"{args.output_dir}/redirect_map.parquet")
