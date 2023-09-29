import argparse
import json
import os
import tarfile
import urllib.parse

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
    page_info['HTML'] = data['article_body']['html']
    page_info['page_length'] = len(data['article_body']['html'])
    if 'abstract' in data:
        page_info['lead_paragraph'] = data['abstract']
    else:
        page_info['lead_paragraph'] = ''
    if 'main_entity' in data:
        page_info['QID'] = data['main_entity']['identifier']
    else:
        page_info['QID'] = None
    if 'redirects' in data:
        for redirect in data['redirects']:
            redirect_map[process_title(
                redirect['name'])] = page_info['title']

    return page_info, redirect_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        required=True, help='Path to the compressed data file')
    parser.add_argument('--language', type=str,
                        required=True, help='Language version of the Wikipedia dump')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    args = parser.parse_args()

    # check if input file exists
    if not os.path.exists(args.input_file):
        raise ValueError(f"Input file {args.input_file} does not exist")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    simple_pages = {}
    redirect_map = {}
    found_names = set([])
    counter = 0

    with tarfile.open(args.input_file, 'r:gz') as tar:
        # iterate through files in tar
        for member in (pbar := tqdm(tar)):
            if member.name.endswith('json'):
                full_pages = []
                file_content = tar.extractfile(member.name).readlines()
                for i, line in enumerate(file_content):
                    if i % 1000 == 0:
                        pbar.set_description(f"Processing file {member.name} at line {i}/{len(file_content)}")
                    entry = json.loads(line)
                    entry['language'] = args.language
                    page, partial_redirect = extract_dump(entry)
                    full_pages.append(page)
                    simple_pages[page['title']] = {
                        'ID': page['ID'], 'QID': page['QID']}
                    redirect_map.update(partial_redirect)
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
