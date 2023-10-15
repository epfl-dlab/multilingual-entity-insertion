import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os
import bz2
from xml.etree.ElementTree import iterparse
from html import unescape
import difflib
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help='directory containing the raw data')
    parser.add_argument('--old_month_input_dir', '-i1', type=str,
                        required=True, help='Input directory for old month')
    parser.add_argument('--new_month_input_dir', '-i2', type=str,
                        required=True, help='Input directory for new month')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--lang', type=str, required=True,
                        help='language of the wikipedia dump')
    parser.add_argument('--date', type=str, required=True,
                        help='date of the latest wikipedia dump in the format YYYYMMDD')

    args = parser.parse_args()
    differ = difflib.Differ()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Loading data")
    old_files = glob(os.path.join(args.old_month_input_dir, "links*"))
    new_files = glob(os.path.join(args.new_month_input_dir, "links*"))

    dfs = []
    for file in tqdm(old_files):
        temp_df = pd.read_parquet(file)
        dfs.append(temp_df)
    old_df = pd.concat(dfs).reset_index(drop=True)

    dfs = []
    for file in tqdm(new_files):
        temp_df = pd.read_parquet(file)
        dfs.append(temp_df)
    new_df = pd.concat(dfs).reset_index(drop=True)

    print("Converting data into better structure")
    old_df = old_df.to_dict(orient='records')
    new_df = new_df.to_dict(orient='records')

    old_data = {}
    for link in tqdm(old_df):
        if link['source_title'] not in old_data:
            old_data[link['source_title']] = {}
        if link['target_title'] not in old_data[link['source_title']]:
            old_data[link['source_title']][link['target_title']] = []
        old_data[link['source_title']][link['target_title']].append(link)

    new_data = {}
    for link in tqdm(new_df):
        if link['source_title'] not in new_data:
            new_data[link['source_title']] = {}
        if link['target_title'] not in new_data[link['source_title']]:
            new_data[link['source_title']][link['target_title']] = []
        new_data[link['source_title']][link['target_title']].append(link)

    print("Finding new links")
    new_pages = 0
    new_page_links = 0
    new_links = []

    for source_page in tqdm(new_data):
        if source_page not in old_data:
            new_pages += 1
            new_page_links += len(new_data[source_page])
            continue
        old_version = old_data[source_page][list(old_data[source_page].keys())[0]][0]['source_version']
        for target_page in new_data[source_page]:
            if target_page not in old_data[source_page]:
                for link in new_data[source_page][target_page]:
                    new_links.append(link)
                    new_links[-1]['old_version'] = old_version
            else:
                links_with_id = []
                links_without_id = []
                for link in new_data[source_page][target_page]:
                    if link['link_ID'] is not None:
                        links_with_id.append(link)
                    else:
                        links_without_id.append(link)
                for link in links_with_id:
                    found = False
                    for old_link in old_data[source_page][target_page]:
                        if link['link_ID'] == old_link['link_ID']:
                            found = True
                            break
                    if not found:
                        new_links.append(link)
                        new_links[-1]['old_version'] = old_version

                used = set([])
                for new_link in links_without_id:
                    for i, old_link in enumerate(old_data[source_page][target_page]):
                        if old_link['link_ID'] is None and old_link['mention'] == new_link['mention'] and i not in used:
                            used.add(i)
                            break
                        if i == len(old_data[source_page][target_page]) - 1:
                            new_links.append(new_link)
                            new_links[-1]['old_version'] = old_version

    print("Saving new links into a better structure")
    link_struc = {}
    for link in tqdm(new_links):
        if link['source_ID'] in link_struc:
            link_struc[int(link['source_ID'])]['links'].append(link)
        else:
            link_struc[int(link['source_ID'])] = {'links': [link], 'old_version': int(link['old_version'].split('oldid=')[-1]), 'new_version': int(link['source_version'].split('oldid=')[-1]), 'page_title': link['source_title']}

    # read the revision history
    expected_links = 0
    output_links = []
    print("Finding links in revision history")
    source_file = os.path.join(args.raw_data_dir, f'{args.lang}wiki-{args.date}-pages-meta-history.xml.bz2')
    with bz2.open(source_file, 'rb') as f:
        pbar = tqdm(iterparse(f, events=('end',)))
        for i, (_, elem) in enumerate(pbar):
            if i % 100_000 == 0:
                pbar.set_description(f"{len(output_links)}/{expected_links} links found")
            if elem.tag.endswith('page'):
                pages = []
                current_id = None
                for child in elem:
                    if child.tag.endswith('ns'):
                        if child.text != '0':
                            break
                    if child.tag.endswith('id'):
                        if int(child.text) not in link_struc:
                            break
                        else:
                            current_id = int(child.text)

                    if child.tag.endswith('revision'):
                        for revision_data in child:
                            if revision_data.tag.endswith('id'):
                                if int(revision_data.text) < link_struc[current_id]['old_version']:
                                    break
                                elif int(revision_data.text) > link_struc[current_id]['new_version']:
                                    break
                                else:
                                    version_id = int(revision_data.text)
                            if revision_data.tag.endswith('text'):
                                clean_text = unescape(revision_data.text)
                                pages.append({'version': version_id, 'text': clean_text})
                if not pages:
                    continue
                pages = sorted(pages, key=lambda x: x['version'], reverse=True)
                expected_links += len(link_struc[current_id]['links'])
                
                mention_count = [{'count': None, 'mention': link['mention'], 'target': link['target_title']} for link in link_struc[current_id]['links']]
                prev_text = ''
                prev_version = None
                for i, page in enumerate(pages):
                    # clean_text = page['text']
                    # clean_text = re.sub(r'\[\[(.*?)\]\]', r'\1', clean_text)
                    # clean_text = re.sub(r'\'{2,3}', '', clean_text)
                    # clean_text = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\1', clean_text)
                    if i == 0:
                        for i, mention in enumerate(mention_count):
                            mention_count[i]['count'] = page['text'].count(f"{mention['mention']}]]")
                    else:
                        for i, mention in enumerate(mention_count):
                            new_count = page['text'].count(f"{mention['mention']}]]")
                            if new_count > mention['count']:
                                mention_count[i]['count'] = new_count
                            if new_count < mention['count']:
                                diff = differ.compare(prev_text.split(), page['text'].split())
                                difference_words = [word for word in diff]
                                removed_blocks = []
                                in_delete = False
                                fixed_text = ''
                                curr_deletion = ''
                                for word in difference_words:
                                    if word.startswith('-'):
                                        if in_delete:
                                            curr_deletion += word[2:] + ' '
                                        else:
                                            in_delete = True
                                            curr_deletion = word[2:] + ' '
                                    elif word.startswith('+'):
                                        continue
                                    else:
                                        if in_delete:
                                            removed_blocks.append({'removed_text': curr_deletion, 'index': len(fixed_text)})
                                            curr_deletion = ''
                                            in_delete = False
                                        fixed_text += word + ' '
                                if in_delete:
                                    removed_blocks.append({'removed_text': curr_deletion, 'index': len(fixed_text)})

                                for block in removed_blocks:
                                    if f"{mention['mention']}]]" in block['removed_text']:
                                        start_position = max(block['index'] - 1_000, 0)
                                        end_position = min(block['index'] + 1_000, len(fixed_text))
                                        context = fixed_text[start_position:end_position]
                                        context = re.sub(r'\[\[(.*?)\]\]', r'\1', context)
                                        context = re.sub(r'\'{2,3}', '', context)
                                        context = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\1', context)
                                        context = re.sub(r'=(.*?)=', r'\1', context)
                                        context = re.sub(r'==(.*?)==', r'\1', context)
                                        context = re.sub(r'===(.*?)===', r'\1', context)
                                        output_links.append({'source': link_struc[current_id]['page_title'], 'target': mention['target'], 'context': context, 'mention': mention['mention'], 'old_version': prev_version, 'new_version': page['version']})
                                        print(output_links[-1])
                                                                                
                                mention_count[i]['count'] = new_count
       
                    prev_text = page['text']
                    prev_version = page['version']
    
    print("Saving links")
    output_df = pd.DataFrame(output_links)
    output_df.to_parquet(os.path.join(args.output_dir, f"{args.date}_val_links.parquet"))