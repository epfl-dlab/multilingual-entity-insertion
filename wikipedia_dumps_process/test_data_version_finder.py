import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os
import bz2
from xml.etree.ElementTree import iterparse
import xml.etree.ElementTree as ET
from html import unescape
import re
import urllib.request
from collections import Counter
import json
from multiprocessing import Pool, cpu_count
import gc
import math
import traceback
tqdm.pandas()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def update_targets(target_name, redirect_map):
    counter = 0
    while target_name in redirect_map:
        target_name = redirect_map[target_name]
        counter += 1
        if counter > 10:
            break
    return target_name


def fill_version(page_title, old_versions):
    if page_title in old_versions:
        return old_versions[page_title]
    return None


def clean_xml(text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = text.replace('\n*', '\n')
    # find all {{...}} elements
    templates = re.findall(r'{{[^}]*}}', text)
    # remove all templates containing 0 | charaters or more than 2 | characters
    templates = [t for t in templates if t.count('|') == 0 or t.count('|') > 2]
    for t in templates:
        text = text.replace(t, '')
    text = text.strip()
    return text


def process_title(title):
    return urllib.parse.unquote(title).replace('_', ' ')


def process_revision_history(input):
    file = input['file']
    # print file size
    link_struc = input['link_struc']
    prev_text = ''
    links = []
    processed_pages = 0
    first_date = pd.to_datetime(args.first_date)
    second_date = pd.to_datetime(args.second_date)
    try:
        print(f'{file}: {round(os.path.getsize(file) / 1024 / 1024, 2)} MB')
        with bz2.open(file, 'rb') as f:
            context = iter(iterparse(f._buffer, events=('start', 'end')))
            _, root = next(context) # get root element
            elem_counter = 0
            for i, (event, elem) in enumerate(pbar := tqdm(context, miniters=1000)):
            # pbar = tqdm(iterparse(f._buffer, events=('end',)))
            # for i, (_, elem) in enumerate(pbar):
                # print size of f in MB
                # print(f'{file}: {round(f.tell() / 1024 / 1024, 2)} MB')
                if elem.tag.endswith('page') and event == 'end':
                    elem_counter = 0
                    pages = []
                    current_id = None
                    skip = False
                    for child in elem:
                        if child.tag.endswith('ns'):
                            if child.text != '0':
                                skip = True
                                break
                        if child.tag.endswith('id'):
                            current_id = int(child.text)
                            pbar.set_description(f"Processing article with ID {current_id} ({len(links)} links found)")
                            if current_id not in link_struc:
                                skip = True
                            break
                    if skip:
                        # remove all references to the element
                        # dont remove root element
                        elem.clear()
                        del elem
                        root.clear()
                        continue
                    # go through all the children of elem in reverse order
                    skip = False
                    for i in range(len(elem) - 1, -1, -1):
                        child = elem[i]
                        if child.tag.endswith('revision'):
                            for revision_data in child:
                                # print(len(pages))
                                if revision_data.tag.endswith('timestamp'):
                                    # timestamp has format 2019-01-01T00:00:00Z
                                    # compare the timestamp with the date of the first dump and the second dump
                                    timestamp = revision_data.text
                                    # print(timestamp)
                                    timestamp = pd.to_datetime(
                                        timestamp).tz_convert(None)
                                    if timestamp > second_date:
                                        break
                                    # if timestamp is more than 7 days before the first date, set leave to True
                                    if timestamp < first_date:
                                        skip = True
                                        
                                if revision_data.tag.endswith('id') and not revision_data.tag.endswith('parentid'):
                                    version_id = int(revision_data.text)
                                if revision_data.tag.endswith('text') and revision_data.text is not None:
                                    clean_text = unescape(revision_data.text)
                                    clean_text = clean_xml(clean_text)
                                    pages.append(
                                        {'version': version_id, 'text': clean_text})
                            elem[i].clear()
                            del elem[i]
                            if skip:
                                break
                    if not pages:
                        elem.clear()
                        # remove all references to the element
                        # dont remove root element
                        del elem
                        root.clear()
                        continue
                    pages = sorted(pages, key=lambda x: x['version'], reverse=True)

                    counts = [{'count': None, 'title': process_title(
                        link['target_title']).lower()} for link in link_struc[current_id]['links']]
                    prev_version = None
                    prev_text = ''
                    for j, page in enumerate(pages):
                        if len(page['text']) < min(0.2 * len(prev_text), 200):  # avoids edit wars
                            continue
                        # find all elements in brackets
                        elems_1 = re.findall(r'\[\[.*?\]\]', page['text'])
                        for i in range(len(elems_1)):
                            elems_1[i] = elems_1[i].lower()
                            if '|' in elems_1[i]:
                                elems_1[i] = elems_1[i].split('|')[0]
                            elems_1[i] = elems_1[i].replace(
                                '[[', '').replace(']]', '')
                            if elems_1[i] in redirect_map_clean:
                                elems_1[i] = redirect_map_clean[elems_1[i]]
                            elems_1[i] = elems_1[i].strip()
                        elems_2 = re.findall(r'\{\{.*?\}\}', page['text'])
                        for i in range(len(elems_2)):
                            elems_2[i] = elems_2[i].lower()
                            if '|' in elems_2[i]:
                                elems_2[i] = elems_2[i].split('|')[1]
                            elems_2[i] = elems_2[i].replace(
                                '{{', '').replace('}}', '')
                            if elems_2[i] in redirect_map_clean:
                                elems_2[i] = redirect_map_clean[elems_2[i]]
                                elems_2[i] = elems_2[i].strip()
                        elems = elems_1 + elems_2
                        # send it to a counter
                        counter = Counter(elems)
                        if j == 0:
                            for k, count in enumerate(counts):
                                counts[k]['count'] = counter.get(
                                    f"{count['title']}", 0)
                        else:
                            for k, count in enumerate(counts):
                                new_count = counter.get(f"{count['title']}", 0)
                                if new_count > count['count']:
                                    counts[k]['count'] = new_count
                                if new_count < count['count']:
                                    links.append({'source_title': link_struc[current_id]['page_title'],
                                                'target_title': link_struc[current_id]['links'][k]['target_title'],
                                                'source_ID': current_id,
                                                'target_ID': link_struc[current_id]['links'][k]['target_ID'],
                                                'first_version': page['version'],
                                                'second_version': prev_version})
                                    counts[k]['count'] = new_count
                        prev_version = page['version']
                        prev_text = page['text']
                    processed_pages += 1
                    elem.clear()
                    # remove all references to the element
                    # dont remove root element
                    del elem
                    root.clear()
                else:
                    elem_counter += 1
                    if elem_counter > 1_000_000: # dont keep more than 1_000_000 elements in memory
                        elem.clear()
                        # remove all references to the element
                        # dont remove root element
                        del elem
                        root.clear()
    except Exception as e:
        # print the exception and any relevant information
        # print traceback
        print(e)
        print(f'Failed to process file {file}')
        return {'links': links, 'processed_pages': processed_pages, 'file_name': 'dummy_file'}
    return {'links': links, 'processed_pages': processed_pages, 'file_name': file}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help='Directory containing the raw data')
    parser.add_argument('--first_month_dir', '-i1', type=str,
                        required=True, help='Input directory for first month')
    parser.add_argument('--second_month_dir', '-i2', type=str,
                        required=True, help='Input directory for second month')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--lang', type=str, required=True,
                        help='language of the wikipedia dump')
    parser.add_argument('--first_date', type=str, required=True,
                        help='date of the first wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--second_date', type=str, required=True,
                        help='date of the second wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--latest_date', type=str, help='date of the latest wikipedia dump in the format YYYYMMDD (relevant if older dumps are not available anymore)')
    parser.add_argument('--download_processes', type=int, default=3, help='Number of processes to use for downloading the revision history files if they are not already downloaded.')
    parser.add_argument('--max_links', type=int, help='Maximum number of links to use')
    parser.add_argument('--download_history', action='store_true', help='Download the revision history files if they are not already downloaded')
    parser.add_argument('--delete_history', action='store_true', help='Delete the revision history files after processing')
    parser.add_argument('--use_available', action='store_true', help='Don\'t download history and use only files available')

    args = parser.parse_args()
    if not args.latest_date:
        args.latest_date = args.second_date
    # check if input directories exist
    if not os.path.exists(args.first_month_dir):
        print(args.first_month_dir)
        raise Exception('First month directory does not exist')
    if not os.path.exists(args.second_month_dir):
        raise Exception('Second month directory does not exist')

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'link_versions')):
        os.makedirs(os.path.join(args.output_dir, 'link_versions'))

    # get all the link files in the month directories
    first_month_link_files = glob(os.path.join(args.first_month_dir, "links*"))
    second_month_link_files = glob(
        os.path.join(args.second_month_dir, "links*"))

    # get the page files for the second month
    first_month_page_files = glob(os.path.join(args.first_month_dir, "pages*"))
    second_month_page_files = glob(
        os.path.join(args.second_month_dir, "pages*"))

    print('Loading data')
    dfs = []
    for file in tqdm(first_month_link_files):
        dfs.append(pd.read_parquet(file, columns=['source_title', 'target_title', 'source_ID', 'target_ID', 'source_version']))
    df_1 = pd.concat(dfs)

    dfs = []
    for file in tqdm(second_month_link_files):
        dfs.append(pd.read_parquet(file, columns=['source_title', 'target_title', 'source_ID', 'target_ID', 'source_version']))
    df_2 = pd.concat(dfs)
    del dfs
    gc.collect()

    redirect_map = pd.concat([pd.read_parquet(os.path.join(args.first_month_dir, 'redirect_map.parquet')), pd.read_parquet(
        os.path.join(args.second_month_dir, 'redirect_map.parquet'))]).drop_duplicates(ignore_index=False)
    redirect_map = redirect_map['redirect'].to_dict()
    redirect_map_clean = {process_title(k).lower(): process_title(
        v).lower() for k, v in redirect_map.items()}

    df_1['target_title'] = df_1['target_title'].progress_apply(
        lambda x: update_targets(x, redirect_map))
    df_2['target_title'] = df_2['target_title'].progress_apply(
        lambda x: update_targets(x, redirect_map))

    df_1 = df_1[['source_title', 'target_title',
                 'source_ID', 'target_ID', 'source_version']]
    df_2 = df_2[['source_title', 'target_title',
                 'source_ID', 'target_ID', 'source_version']]

    # group the links by source and target and count the number of links
    df_1 = df_1.groupby(['source_title', 'target_title', 'source_ID',
                        'target_ID', 'source_version']).size().reset_index(name='count')
    df_2 = df_2.groupby(['source_title', 'target_title', 'source_ID',
                        'target_ID', 'source_version']).size().reset_index(name='count')

    # find all new links added in df_2. Consider two cases
    # 1. The row is not present in df_1
    # 2. The row is present in df_1 but the count is smaller in df_1
    df_2 = df_2.merge(df_1, how='left', on=[
                      'source_title', 'target_title', 'source_ID', 'target_ID'], suffixes=('_2', '_1'))
    del df_1
    gc.collect()
    print('DataFrames merged')

    df_2 = df_2[(df_2['count_1'].isna()) | (df_2['count_2'] > df_2['count_1'])]
    df_2['count_1'] = df_2['count_1'].fillna(0)
    df_2['source_version_1'] = df_2['source_version_1'].fillna('&oldid=0')
    df_2['count'] = df_2['count_2'] - df_2['count_1']
    df_2 = df_2[['source_title', 'target_title', 'source_ID',
                 'target_ID', 'source_version_1', 'source_version_2', 'count']]
    print('Final DataFrame created')

    initial_size = df_2['count'].sum()
    print(f'Initially, there are {df_2["count"].sum()} new candidate links, from {len(df_2)} src-tgt pairs.')

    source_pages = len(df_2['source_title'].unique())

    df_2 = df_2.to_dict('records')

    link_struc = {}
    for link in tqdm(df_2):
        if int(link['source_ID']) in link_struc:
            link_struc[int(link['source_ID'])]['links'].append(link)
        else:
            link_struc[int(link['source_ID'])] = {'links': [link],
                                                  #   'old_version': int(link['source_version_1'].split('&oldid=')[-1]),
                                                  #   'new_version': int(link['source_version_2'].split('&oldid=')[-1]),
                                                  'page_title': link['source_title']}
    del df_2
    gc.collect()
    print("Additional structure created")

    # check if the revision history file exists
    if os.path.exists(os.path.join(args.raw_data_dir, f'{args.lang}wiki-{args.latest_date}-pages-meta-history.xml.bz2')):
        files = {f'{os.path.join(args.raw_data_dir, f"{args.lang}wiki-{args.latest_date}-pages-meta-history.xml.bz2")}': link_struc}
    else:
        if args.download_history:
            print('Could not find single-file revision history. Attempting to download it.')
            url_xml_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.latest_date}/{args.lang}wiki-{args.latest_date}-pages-meta-history.xml.bz2"
            output_xml_path = f"{args.raw_data_dir}/{args.lang}wiki-{args.latest_date}-pages-meta-history.xml.bz2"
            try:
                download_url(url_xml_dump, output_xml_path)
                files = {output_xml_path: link_struc}
            except:
                print(f'Could not download {url_xml_dump}. This most likely means the dump is too large and the revision history is split into multiple files.')
                print('Checking if dumpstatus.json is available')
    
                # check if dumpstatus.json exists
                if not os.path.exists(os.path.join(args.raw_data_dir, 'dumpstatus.json')):
                    backup_json_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.latest_date}/dumpstatus.json"
                    output_backup_path = f"{args.raw_data_dir}/dumpstatus.json"
                    print(backup_json_dump)
                    download_url(backup_json_dump, output_backup_path)
                print('Found dumpstatus.json')
                json_file = os.path.join(args.raw_data_dir, 'dumpstatus.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)
                # check if the revision history is available
                if 'jobs' not in data or 'metahistorybz2dump' not in data['jobs'] or 'files' not in data['jobs']['metahistorybz2dump']:
                    # download dumpstatus.json again
                    backup_json_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.latest_date}/dumpstatus.json"
                    output_backup_path = f"{args.raw_data_dir}/dumpstatus.json"
                    download_url(backup_json_dump, output_backup_path)
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                for file in data['jobs']['metahistorybz2dump']['files']:
                    file_name = os.path.join(args.raw_data_dir, file)
                    if not os.path.exists(file_name):
                        print('Re-downloading dumpstatus.json because the revision history is not available')
                        backup_json_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.latest_date}/dumpstatus.json"
                        output_backup_path = f"{args.raw_data_dir}/dumpstatus.json"
                        download_url(backup_json_dump, output_backup_path)
                        break
                with open(json_file, 'r') as f:
                    data = json.load(f)
                data = data['jobs']['metahistorybz2dump']['files']
                ranges = []
                for key in data:
                    file_name = key
                    url = f"https://dumps.wikimedia.org{data[key]['url']}"
                    # filename is structure as {lang}wiki-{date}-pages-meta-history{num}.xml-p{min_id}p{max_id}.bz2
                    # use regex to extract min_id and max_id
                    min_id = int(re.findall(r'p(\d+)p', file_name)[0])
                    max_id = int(re.findall(r'p\d+p(\d+)', file_name)[0])
                    ranges.append({'min_id': min_id, 
                                'max_id': max_id, 
                                'file_name': os.path.join(args.raw_data_dir, file_name),
                                'url': url})
                files = {}
                needed_files = []
                for id in link_struc:
                    for range_ in ranges:
                        if range_['min_id'] <= id <= range_['max_id']:
                            needed_files.append(range_)
                            if range_['file_name'] not in files:
                                files[range_['file_name']] = {}
                            files[range_['file_name']][id] = link_struc[id]
                            break
                # check if all files are downloaded
                download_files = []
                json_downloaded = False
                for file in needed_files:
                    if not os.path.exists(file['file_name']) and not args.use_available:
                        if (args.download_history) and (file not in download_files):
                            download_files.append(file)
                        elif not args.download_history:
                            raise Exception(f'Could not find {file["file_name"]}')
                if download_files:
                    print(f"Downloading {len(download_files)} revision history files")
                    urls = [file['url'] for file in download_files]
                    file_names = [file['file_name'] for file in download_files]
                    with Pool(args.download_processes) as p:
                        p.starmap(download_url, zip(urls, file_names))

    input = [{'file': file, 'link_struc': link_struc} for file in files]
    # remove any file from the input which does not exist
    input = [file for file in input if os.path.exists(file['file'])]
    # sort by file size 
    input = sorted(input, key=lambda x: os.path.getsize(x['file']), reverse=True)
    if args.max_links is None:
        max_links = float('inf')
    else:
        max_links = args.max_links
    print(f"Finding links in {len(files)} revision history file(s)")
    # read the revision history
    found_links = 0
    processed_pages = 0
    counter = 0
    with Pool(min(4, len(input))) as p:
        for result in (pbar := tqdm(p.imap_unordered(process_revision_history, input), total=len(input))):
            output = result['links']
            found_links += len(output)
            processed_pages += result['processed_pages']
            pbar.set_description(
                f"{found_links} candidate links found ({processed_pages}/{source_pages} pages processed)")  
            print(found_links)
            if len(output) > 0:
                output_df = pd.DataFrame(output)
                output_df.to_parquet(os.path.join(args.output_dir, "link_versions", f"{os.path.basename(result['file_name'])}.parquet"))
                counter += 1
            if args.delete_history:
                try:
                    os.remove(result['file_name'])
                except:
                    print(f'Couldn\t remove file {result["file_name"]}')
            if found_links >= max_links:
                break

    if args.delete_history:
        print('Deleting revision history files')
        for file in files:
            try:
                os.remove(file)
            except:
                print(f'Could not delete {file}')
