import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os
import bz2
from xml.etree.ElementTree import iterparse
from html import unescape
import re
import urllib
from collections import Counter

def update_targets(target_name, redirect_map):
    if target_name in redirect_map:
        return redirect_map[target_name]
    return target_name

def fill_version(page_title, old_versions):
    if page_title in old_versions:
        return old_versions[page_title]
    return None

def process_title(title):
    return urllib.parse.unquote(title).replace('_', ' ')

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
    parser.add_argument('--date', type=str, required=True,
                        help='date of the latest wikipedia dump in the format YYYYMMDD')

    args = parser.parse_args()
    # check if input directories exist
    if not os.path.exists(args.first_month_dir):
        raise Exception('First month directory does not exist')
    if not os.path.exists(args.second_month_dir):
        raise Exception('Second month directory does not exist')

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        dfs.append(pd.read_parquet(file))
    df_1 = pd.concat(dfs)
    
    dfs = []
    for file in tqdm(second_month_link_files):
        dfs.append(pd.read_parquet(file))
    df_2 = pd.concat(dfs)

    dfs = []
    for file in tqdm(first_month_page_files):
        dfs.append(pd.read_parquet(file))
    df_pages_1 = pd.concat(dfs)
    
    redirect_map = pd.read_parquet(os.path.join(args.second_month_dir, 'redirect_map.parquet'))
    redirect_map = redirect_map.to_dict()['redirect']
    
    df_1['target_title'] = df_1['target_title'].apply(lambda x: update_targets(x, redirect_map))

    df_1 = df_1[['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID', 'source_version']]
    df_2 = df_2[['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID', 'source_version']]

    # group the links by source and target and count the number of links
    df_1 = df_1.groupby(['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID', 'source_version']).size().reset_index(name='count')
    df_2 = df_2.groupby(['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID', 'source_version']).size().reset_index(name='count')
    
    # find all new links added in df_2. Consider two cases
    # 1. The row is not present in df_1
    # 2. The row is present in df_1 but the count is smaller in df_1
    df_2 = df_2.merge(df_1, how='left', on=['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID'], suffixes=('_2', '_1'))
    df_2 = df_2[(df_2['count_1'].isna()) | (df_2['count_2'] > df_2['count_1'])]
    df_2['count_1'] = df_2['count_1'].fillna(0)
    df_2['count'] = df_2['count_2'] - df_2['count_1']
    df_2 = df_2[['source_title', 'target_title', 'source_ID', 'target_ID', 'source_QID', 'target_QID', 'source_version_1', 'source_version_2', 'count']]

    no_html = set(df_pages_1[(df_pages_1['HTML'].isna()) | (
        df_pages_1['HTML'] == '')]['title'].tolist())
    no_qid = set(df_pages_1[df_pages_1['QID'].isna()]['title'].tolist())
    no_lead = set(df_pages_1[(df_pages_1['lead_paragraph'].isna()) | (
        df_pages_1['lead_paragraph'] == '')]['title'].tolist())
    short_lead = set(df_pages_1[df_pages_1['lead_paragraph'].apply(
        lambda x: x is not None and len(x.split()) < 6)]['title'].tolist())
    old_pages = set(df_pages_1['title'].tolist())
    # create a dictionary with page title as key and version as value
    old_versions = df_pages_1[['title', 'version']].set_index(
        'title').to_dict()['version']
    
    df_2['source_version_1'] = df_2['source_title'].apply(lambda x: fill_version(x, old_versions))
    
    initial_size = df_2['count'].sum()
    print(f'Initially, there are {df_2["count"].sum()} new candidate links.')
    
    df_2_removed = df_2[~df_2['source_title'].isin(old_pages)]
    df_2 = df_2[df_2['source_title'].isin(old_pages)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the source page was not present in the old data.')
    
    df_2_removed = df_2[~df_2['target_title'].isin(old_pages)]
    df_2 = df_2[df_2['target_title'].isin(old_pages)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the target page was not present in the old data.')
    
    df_2_removed = df_2[df_2['source_title'].isin(no_html)]
    df_2 = df_2[~df_2['source_title'].isin(no_html)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have HTML.')
    
    df_2_removed = df_2[df_2['target_title'].isin(no_html)]
    df_2 = df_2[~df_2['target_title'].isin(no_html)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have HTML.')
    
    df_2_removed = df_2[df_2['source_title'].isin(no_qid)]
    df_2 = df_2[~df_2['source_title'].isin(no_qid)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have a QID.')  
    
    df_2_removed = df_2[df_2['target_title'].isin(no_qid)]
    df_2 = df_2[~df_2['target_title'].isin(no_qid)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have a QID.')  
    
    df_2_removed = df_2[df_2['source_title'].isin(no_lead)]
    df_2 = df_2[~df_2['source_title'].isin(no_lead)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have a lead paragraph.')
    
    df_2_removed = df_2[df_2['target_title'].isin(no_lead)]
    df_2 = df_2[~df_2['target_title'].isin(no_lead)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have a lead paragraph.')
    
    df_2_removed = df_2[df_2['source_title'].isin(short_lead)]
    df_2 = df_2[~df_2['source_title'].isin(short_lead)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the source page had a short lead paragraph.')
    
    df_2_removed = df_2[df_2['target_title'].isin(short_lead)]
    df_2 = df_2[~df_2['target_title'].isin(short_lead)]
    print(f'Out of these, {df_2_removed["count"].sum()} were removed because the target page had a short lead paragraph.')
    
    print(f'In the end, we were left with {df_2["count"].sum()} new candidate links, with {initial_size - df_2["count"].sum()} ({(initial_size - df_2["count"].sum()) / initial_size * 100:.2f}%) removed.')    
    
    source_pages = len(df_2['source_title'].unique())
    
    df_2 = df_2.to_dict('records')

    link_struc = {}
    for link in df_2:
        if link['source_ID'] in link_struc:
            if link['source_ID'] in link_struc:
                link_struc[int(link['source_ID'])]['links'].append(link)
        else:
            link_struc[int(link['source_ID'])] = {'links': [link], 
                                                  'old_version': int(link['source_version_1'].split('&oldid=')[-1]), 
                                                  'new_version': int(link['source_version_2'].split('&oldid=')[-1]), 
                                                  'page_title': link['source_title']}

    
    # read the revision history
    output = []
    print("Finding links in revision history")
    source_file = os.path.join(args.raw_data_dir, f'{args.lang}wiki-{args.date}-pages-meta-history.xml.bz2')
    processed_pages = 0
    prev_text = ''
    with bz2.open(source_file, 'rb') as f:
        pbar = tqdm(iterparse(f, events=('end',)))
        for i, (_, elem) in enumerate(pbar):
            if i % 100_000 == 0:
                pbar.set_description(f"{len(output)} candidate links found ({processed_pages}/{source_pages} pages processed)")
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
                            if revision_data.tag.endswith('id') and not revision_data.tag.endswith('parentid'):
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

                counts = [{'count': None, 'found': 0, 'title': process_title(link['target_title']).lower()} for link in link_struc[current_id]['links']]
                prev_version = None
                prev_text = ''
                for j, page in enumerate(pages):
                    page['text'] = page['text'].lower()
                    if len(page['text']) < 0.5 * len(prev_text): # avoids edit wars
                        continue 
                    # find all elements in brackets
                    elems = re.findall(r'\[\[.*?\]\]', page['text'])
                    for i in range(len(elems)):
                        if '|' in elems[i]:
                            elems[i] = elems[i].split('|')[0] + ']]'
                    # send it to a counter
                    counter = Counter(elems)
                    if j == 0:
                        for k, count in enumerate(counts):
                            counts[k]['count'] = counter.get(f"[[{count['title']}]]", 0)
                    else:
                        for k, count in enumerate(counts):
                            new_count = counter.get(f"[[{count['title']}]]", 0)
                            if new_count > count['count']:
                                counts[k]['count'] = new_count
                            if new_count < count['count']:
                                for _ in range(count['count'] - new_count):
                                    output.append({'source_title': link_struc[current_id]['page_title'],
                                                'target_title': link_struc[current_id]['links'][k]['target_title'],
                                                'source_ID': current_id,
                                                'target_ID': link_struc[current_id]['links'][k]['target_ID'],
                                                'source_QID': link_struc[current_id]['links'][k]['source_QID'],
                                                'target_QID': link_struc[current_id]['links'][k]['target_QID'],
                                                'missing_version': prev_version,
                                                'found_version': page['version']})
                                    counts[k]['found'] += 1
                                counts[k]['count'] = new_count
                    prev_version = page['version']
                    prev_text = page['text']
                processed_pages += 1

    print("Saving versions")
    output_df = pd.DataFrame(output)
    output_df.to_parquet(os.path.join(args.output_dir, f"link_versions.parquet"))