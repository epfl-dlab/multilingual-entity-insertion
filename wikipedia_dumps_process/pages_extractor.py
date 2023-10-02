import argparse
import bz2
import gzip
import json
import os
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def process_title(title):
    title = title.replace(' ', '_')
    # decode from url encoding
    title = urllib.parse.unquote(title)
    # reencode into url encoding
    title = urllib.parse.quote(title)
    return title


def process_xml_articles(path, language):
    # process xml
    tree = ET.parse(path)
    root = tree.getroot()
    pages = []
    redirects = []
    for child in tqdm(root):
        page_info = {}
        redirect = {}
        valid = True
        if 'siteinfo' in child.tag:
            continue
        for grandchild in child:
            # get the tag name without the http://www.mediawiki.org/xml/export-0.10/ prefix
            tag = grandchild.tag.split('}')[1]
            if tag == 'ns' and grandchild.text != '0':
                valid = False
                break
            if tag == 'title':
                page_info['title'] = process_title(grandchild.text)
                redirect['title'] = process_title(grandchild.text)
            if tag == 'id':
                page_info['ID'] = grandchild.text
                redirect['ID'] = grandchild.text
            if tag == 'revision':
                for greatgrandchild in grandchild:
                    tag = greatgrandchild.tag.split('}')[1]
                    if tag == 'id':
                        page_info['version'] = greatgrandchild.text
                        break
            if tag == 'redirect':
                redirect['redirect'] = grandchild.attrib['title']
        if '1589' in page_info['title']:
            print(page_info)
        if 'redirect' in redirect:
            redirects.append(redirect)
        elif valid:
            page_info['version'] = f"https://{language}.wikipedia.org/w/index.php?title={page_info['title']}&oldid={page_info['version']}"
            pages.append(page_info)
    return pages, redirects


def process_sql_data(path):
    pages = []
    with open(path, 'r', encoding='latin-1') as f:
        data = f.readlines()
        for line in tqdm(data):
            if not line.startswith('INSERT INTO'):
                continue
            line = line.replace('INSERT INTO `page_props` VALUES ', '')
            entries = line.split('),(')
            entries[0] = entries[0].replace('(', '')
            entries[-1] = entries[-1].replace(');', '')
            for entry in entries:
                try:
                    id, category, qid, _ = entry.split(',', 3)
                except:
                    print(entry)
                    continue
                if category != "'wikibase_item'":
                    continue
                pages.append({'ID': id, 'QID': qid[1:-1]})
    return pages


def process_tar_data(data):
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
    parser.add_argument('--input_xml', type=str,
                        required=True, help='Path to the compressed xml input file')
    parser.add_argument('--input_sql', type=str, required=True,
                        help='Path to the compressed sql input file')
    parser.add_argument('--input_tar', type=str, required=True,
                        help='Path to the compressed tar input file')
    parser.add_argument('--language', type=str,
                        required=True, help='Language version of the Wikipedia dump')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    args = parser.parse_args()

    # check if input files exist
    if not os.path.exists(args.input_xml):
        raise ValueError(f"Input file {args.input_xml} does not exist")
    if not os.path.exists(args.input_sql):
        raise ValueError(f"Input file {args.input_sql} does not exist")
    if not os.path.exists(args.input_tar):
        raise ValueError(f"Input file {args.input_tar} does not exist")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # extract pages from xml dump
    print(f"Parsing xml dump {args.input_xml}")
    pages_xml, redirects = process_xml_articles(args.input_xml, args.language)
    # extract pages from sql dump
    print(f"Parsing sql dump {args.input_sql}")
    pages_sql = process_sql_data(args.input_sql)

    print("Saving DataFrames")
    # create single dataframe for pages
    df_xml = pd.DataFrame(pages_xml).set_index('ID')
    df_sql = pd.DataFrame(pages_sql).set_index('ID')

    pages = df_xml.merge(df_sql, left_index=True,
                         right_index=True, how='inner')
    pages.reset_index(level=0, inplace=True)
    pages = pages.rename(columns={'index': 'ID'})

    # create copy of pages with reduced information
    simple_pages = pages[['ID', 'title']]
    simple_pages = simple_pages.set_index('title')

    # create dataframe for redirects
    # use source page as index
    df_redirects = pd.DataFrame(redirects).set_index('title')
    # only keep the redirect column
    df_redirects = df_redirects[['redirect']]

    # save dataframes
    pages.to_parquet(f"{args.output_dir}/pages.parquet")
    simple_pages.to_parquet(f"{args.output_dir}/simple_pages.parquet")
    df_redirects.to_parquet(f"{args.output_dir}/redirect_map.parquet")

    print(f'Processing tar dump {args.input_tar}')
    with tarfile.open(args.input_file, 'r:gz') as tar:
        # iterate through files in tar
        for member in (pbar := tqdm(tar)):
            if member.name.endswith('json'):
                full_pages = []
                file_content = tar.extractfile(member.name).readlines()
                for i, line in enumerate(file_content):
                    if i % 1000 == 0:
                        pbar.set_description(
                            f"Processing file {member.name} at line {i}/{len(file_content)}")
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
