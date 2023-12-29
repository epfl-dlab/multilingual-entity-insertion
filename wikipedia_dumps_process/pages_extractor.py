import argparse
import json
import os
import re
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
import gc


def process_title(title):
    title = title.replace(' ', '_')
    # decode from url encoding
    title = urllib.parse.unquote(title)
    # reencode into url encoding
    title = urllib.parse.quote(title)
    return title


def process_sql_page(input, language):
    with open(input) as f:
        data = f.readlines()
        # read only lines that start with "INSERT INTO"
        # these are the lines containing the data
        data = [x.strip() for x in data if x.startswith('INSERT INTO')]

    pages = []
    redirects = {}
    for line in tqdm(data):
        # remove the "INSERT INTO `page` VALUES (" part and the last semicolon
        line = line.replace('INSERT INTO `page` VALUES (', '')
        line = line.replace(');', '')

        # split the line using the following rules:
        # 1. "),(" is the separator between entries
        # 2. After "),(" there are digits followed by ","; use a positive lookahead to split
        elements = re.split("\),\((?=\d+,)", line)
        for element in elements:
            split_data = element.split(",")
            id = split_data[0]
            namespace = split_data[1]
            title = ",".join(split_data[2:-9]).replace("\\'", "'")
            is_redirect = split_data[-9]
            if namespace != "0":
                continue
            if is_redirect == '1':
                redirects[id] = {'title': process_title(title[1:-1])}
                continue
            pages.append({'ID': id, 'title': process_title(
                title[1:-1]), 'language': language})
    return pages, redirects


def process_sql_redirects(input, redirects):
    with open(input) as f:
        data = f.readlines()
        # read only lines that start with "INSERT INTO"
        # these are the lines containing the data
        data = [x.strip() for x in data if x.startswith('INSERT INTO')]

    for line in tqdm(data):
        # remove the "INSERT INTO `redirect` VALUES (" part and the last semicolon
        line = line.replace('INSERT INTO `redirect` VALUES (', '')
        line = line.replace(');', '')

        # split the line using the following rules:
        # 1. "),(" is the separator between entries
        # 2. After "),(" there are digits followed by ",", use a positive lookahead to split
        elements = re.split("\),\((?=\d+,)", line)
        for element in elements:
            split_data = element.split(",")
            if split_data[0] not in redirects or split_data[1] != '0':
                continue
            redirects[split_data[0]]['redirect'] = process_title(
                ",".join(split_data[2:-2]).replace("\\'", "'")[1:-1])
    return redirects


def process_sql_page_props(input, pages):
    with open(input, encoding='latin-1') as f:
        data = f.readlines()
        # read only lines that start with "INSERT INTO"
        # these are the lines containing the data
        data = [x for x in data if x.startswith('INSERT INTO')]

    qids_map = {}
    for line in tqdm(data):
        # remove the "INSERT INTO `page_props` VALUES (" part and the last semicolon
        line = line.replace('INSERT INTO `page_props` VALUES (', '')
        line = line.replace(');', '')

        # split the line using the following rules:
        # 1. "),(" is the separator between entries
        # 2. After "),(" there are digits followed by ",", use a positive lookahead to split
        elements = re.split("\),\((?=\d+,)", line)
        for element in elements:
            split_data = element.split(",")
            id = split_data[0]
            category = split_data[1]
            qid = split_data[2][1:-1]
            if category != "'wikibase_item'":
                continue
            qids_map[id] = qid

    for i, page in enumerate(pages):
        if page['ID'] in qids_map:
            pages[i]['QID'] = qids_map[page['ID']]
        else:
            pages[i]['QID'] = None
    return pages


def process_tar_html(data):
    page_info = {}
    page_info['title'] = process_title(data['name'])
    page_info['ID'] = str(data['identifier'])
    page_info['version'] = f"https://{data['language']}.wikipedia.org/w/index.php?title={page_info['title']}&oldid={data['version']['identifier']}"
    page_info['HTML'] = data['article_body']['html']
    page_info['page_length'] = len(data['article_body']['html'])
    page_info['language'] = data['language']
    page_info['date_modified'] = data['date_modified']
    # get the abstract if it exists (abstract = lead paragraph)
    if 'abstract' in data:
        page_info['lead_paragraph'] = data['abstract']
    else:
        page_info['lead_paragraph'] = ''
    # get the QID if it is available
    if 'main_entity' in data:
        page_info['QID'] = data['main_entity']['identifier']
    else:
        page_info['QID'] = None

    return page_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Path to the input folder')
    parser.add_argument('--language', type=str,
                        required=True, help='Language version of the Wikipedia dump')
    parser.add_argument('--date', type=str, required=True,
                        help='Date of the Wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    args = parser.parse_args()

    # check if input folder exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input folder {args.input_dir} does not exist")
    # check if input folder contains the required files
    sql_pages = os.path.join(
        args.input_dir, f"{args.language}wiki-{args.date}-page.sql")
    sql_redirects = os.path.join(
        args.input_dir, f"{args.language}wiki-{args.date}-redirect.sql")
    sql_page_props = os.path.join(
        args.input_dir, f"{args.language}wiki-{args.date}-page_props.sql")
    tar_html = os.path.join(
        args.input_dir, f"{args.language}wiki-NS0-{args.date}-ENTERPRISE-HTML.json.tar.gz")
    if not os.path.exists(sql_pages):
        raise ValueError(
            f"Missing file {sql_pages} from input folder. Please run the download script first.")
    if not os.path.exists(sql_redirects):
        raise ValueError(
            f"Missing file {sql_redirects} from input folder. Please run the download script first.")
    if not os.path.exists(sql_page_props):
        raise ValueError(
            f"Missing file {sql_page_props} from input folder. Please run the download script first.")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'pages')):
        os.makedirs(os.path.join(args.output_dir, 'pages'))

    print(f"Processing SQL pages information")
    pages, redirects = process_sql_page(sql_pages, args.language)
    print(f"Parsing SQL redirects information")
    redirects = process_sql_redirects(sql_redirects, redirects)

    # remove redirects that redirect to themselves
    redirects = [redirects[key] for key in redirects if 'redirect' in redirects[key]
                 and redirects[key]['redirect'] != redirects[key]['title']]
    # save into dictionary for faster lookup
    redirect_map = {redirect['title']: redirect['redirect']
                    for redirect in redirects}
    # save redirects
    redirects = pd.DataFrame(redirects).set_index('title')
    redirects.to_parquet(f"{args.output_dir}/redirect_map.parquet")
    # memory management
    del redirects
    gc.collect()

    print("Processing SQL page properties information")
    pages = process_sql_page_props(sql_page_props, pages)
    # create a reduced version of the page data (id, title, qid)
    simple_pages = [{'ID': page['ID'], 'title': page['title'],
                     'QID': page['QID']} for page in pages]
    simple_pages = pd.DataFrame(simple_pages)
    simple_pages = simple_pages.set_index('title')
    simple_pages.to_parquet(f"{args.output_dir}/simple_pages.parquet")
    print(f"{len(simple_pages)} pages found")
    # memory management
    del simple_pages
    gc.collect()

    # create a dictionary with the pages for faster lookup
    pages = {page['title']: page for page in pages}

    print(f'Processing tar HTML data')
    full_pages = []
    used_titles = set()
    counter = 0
    with tarfile.open(tar_html, 'r:gz') as tar:
        # iterate through files in tar
        for member in (pbar := tqdm(tar)):
            if member.name.endswith('json'):
                file_content = tar.extractfile(member.name).readlines()
                for i, line in enumerate(file_content):
                    if i % 1000 == 0:
                        pbar.set_description(
                            f"Processing file {member.name} at line {i}/{len(file_content)}")
                    entry = json.loads(line)
                    entry['language'] = args.language
                    page = process_tar_html(entry)
                    # if the page is a redirect, ignore
                    if page['title'] in redirect_map:
                        continue
                    # if the page has already appeared, ignore
                    if page['title'] in used_titles:
                        continue
                    used_titles.add(page['title'])
                    if page['title'] in pages:
                        # QID is potentially present in two places, the SQL dump and the HTML dump
                        # if it was missing from the SQL dump, but present in the HTML dump, add it
                        if page['QID'] is None and pages[page['title']]['QID'] is not None:
                            page['QID'] = pages[page['title']]['QID']
                        del pages[page['title']]

                    full_pages.append(page)
                    # splitting into multiple files for memory management
                    if len(full_pages) >= 10_000:
                        full_pages = pd.DataFrame(full_pages)
                        full_pages.to_parquet(
                            f"{args.output_dir}/pages/pages_{counter}.parquet")
                        del full_pages
                        gc.collect()
                        full_pages = []
                        counter += 1

    # create a dataframe with the remaining titles
    for title in pages:
        full_pages.append(pages[title])
        full_pages[-1]['HTML'] = None
        full_pages[-1]['page_length'] = None
        full_pages[-1]['lead_paragraph'] = None
        full_pages[-1]['version'] = None
    full_pages = pd.DataFrame(full_pages)
    full_pages.to_parquet(f"{args.output_dir}/pages/pages_{counter}.parquet")
