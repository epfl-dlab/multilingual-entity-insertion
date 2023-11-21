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

    redirect_map = pd.concat([pd.read_parquet(os.path.join(args.first_month_dir, 'redirect_map.parquet')), pd.read_parquet(
        os.path.join(args.second_month_dir, 'redirect_map.parquet'))]).drop_duplicates(ignore_index=False)
    redirect_map = redirect_map['redirect'].to_dict()
    redirect_map_clean = {process_title(k).lower(): process_title(
        v).lower() for k, v in redirect_map.items()}

    df_1['target_title'] = df_1['target_title'].apply(
        lambda x: update_targets(x, redirect_map))
    df_2['target_title'] = df_2['target_title'].apply(
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
    df_2 = df_2[(df_2['count_1'].isna()) | (df_2['count_2'] > df_2['count_1'])]
    df_2['count_1'] = df_2['count_1'].fillna(0)
    df_2['source_version_1'] = df_2['source_version_1'].fillna('&oldid=0')
    df_2['count'] = df_2['count_2'] - df_2['count_1']
    df_2 = df_2[['source_title', 'target_title', 'source_ID',
                 'target_ID', 'source_version_1', 'source_version_2', 'count']]

    # no_html = set(df_pages_1[(df_pages_1['HTML'].isna()) | (
    #     df_pages_1['HTML'] == '')]['title'].tolist())
    # no_qid = set(df_pages_1[df_pages_1['QID'].isna()]['title'].tolist())
    # no_lead = set(df_pages_1[(df_pages_1['lead_paragraph'].isna()) | (
    #     df_pages_1['lead_paragraph'] == '')]['title'].tolist())
    # short_lead = set(df_pages_1[df_pages_1['lead_paragraph'].apply(
    #     lambda x: x is not None and len(x.split()) < 6)]['title'].tolist())
    # old_pages = set(df_pages_1['title'].tolist())
    # create a dictionary with page title as key and version as value
    # old_versions = df_pages_1[['title', 'version']].set_index(
    #     'title').to_dict()['version']

    # df_2['source_version_1'] = df_2['source_title'].apply(
    #     lambda x: fill_version(x, old_versions))

    initial_size = df_2['count'].sum()
    print(f'Initially, there are {df_2["count"].sum()} new candidate links, from {len(df_2)} src-tgt pairs.')

    # df_2_removed = df_2[~df_2['source_title'].isin(old_pages)]
    # df_2 = df_2[df_2['source_title'].isin(old_pages)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the source page was not present in the old data.')

    # df_2_removed = df_2[~df_2['target_title'].isin(old_pages)]
    # df_2 = df_2[df_2['target_title'].isin(old_pages)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the target page was not present in the old data.')

    # df_2_removed = df_2[df_2['source_title'].isin(no_html)]
    # df_2 = df_2[~df_2['source_title'].isin(no_html)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have HTML.')

    # df_2_removed = df_2[df_2['target_title'].isin(no_html)]
    # df_2 = df_2[~df_2['target_title'].isin(no_html)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have HTML.')

    # df_2_removed = df_2[df_2['source_title'].isin(no_qid)]
    # df_2 = df_2[~df_2['source_title'].isin(no_qid)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have a QID.')

    # df_2_removed = df_2[df_2['target_title'].isin(no_qid)]
    # df_2 = df_2[~df_2['target_title'].isin(no_qid)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have a QID.')

    # df_2_removed = df_2[df_2['source_title'].isin(no_lead)]
    # df_2 = df_2[~df_2['source_title'].isin(no_lead)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the source page did not have a lead paragraph.')

    # df_2_removed = df_2[df_2['target_title'].isin(no_lead)]
    # df_2 = df_2[~df_2['target_title'].isin(no_lead)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the target page did not have a lead paragraph.')

    # df_2_removed = df_2[df_2['source_title'].isin(short_lead)]
    # df_2 = df_2[~df_2['source_title'].isin(short_lead)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the source page had a short lead paragraph.')

    # df_2_removed = df_2[df_2['target_title'].isin(short_lead)]
    # df_2 = df_2[~df_2['target_title'].isin(short_lead)]
    # print(
    #     f'Out of these, {df_2_removed["count"].sum()} were removed because the target page had a short lead paragraph.')

    # print(
    #     f'In the end, we were left with {df_2["count"].sum()} new candidate links, with {initial_size - df_2["count"].sum()} ({(initial_size - df_2["count"].sum()) / initial_size * 100:.2f}%) removed.')

    source_pages = len(df_2['source_title'].unique())

    df_2 = df_2.to_dict('records')

    link_struc = {}
    for link in df_2:
        if int(link['source_ID']) in link_struc:
            link_struc[int(link['source_ID'])]['links'].append(link)
        else:
            link_struc[int(link['source_ID'])] = {'links': [link],
                                                  #   'old_version': int(link['source_version_1'].split('&oldid=')[-1]),
                                                  #   'new_version': int(link['source_version_2'].split('&oldid=')[-1]),
                                                  'page_title': link['source_title']}

    # read the revision history
    output = []
    print("Finding links in revision history")
    source_file = os.path.join(
        args.raw_data_dir, f'{args.lang}wiki-{args.second_date}-pages-meta-history.xml.bz2')
    processed_pages = 0
    prev_text = ''
    with bz2.open(source_file, 'rb') as f:
        pbar = tqdm(iterparse(f._buffer, events=('end',)))
        output_len = 0
        for i, (_, elem) in enumerate(pbar):
            if i % 100_000 == 0:
                pbar.set_description(
                    f"{len(output)} candidate links found ({processed_pages}/{source_pages} pages processed)")
            if elem.tag.endswith('page'):
                pages = []
                older_pages = []
                current_id = None
                skip = False
                for child in elem:
                    if child.tag.endswith('ns'):
                        if child.text != '0':
                            skip = True
                            break
                    if child.tag.endswith('id'):
                        if int(child.text) not in link_struc:
                            skip = True
                            break
                        else:
                            current_id = int(child.text)
                            break
                if skip:
                    elem.clear()
                    continue
                # go through all the children of elem in reverse order
                old = False
                leave = False
                for child in reversed(elem):
                    if child.tag.endswith('revision'):
                        for revision_data in child:
                            if revision_data.tag.endswith('timestamp'):
                                # timestamp has format 2019-01-01T00:00:00Z
                                # compare the timestamp with the date of the first dump and the second dump
                                timestamp = revision_data.text
                                timestamp = pd.to_datetime(
                                    timestamp).tz_convert(None)
                                first_date = pd.to_datetime(args.first_date)
                                second_date = pd.to_datetime(args.second_date)
                                if timestamp > second_date:
                                    break
                                elif timestamp < first_date:
                                    old = True

                                # if timestamp is more than 1 month before the first date, set leave to True
                                if timestamp < first_date - pd.DateOffset(months=1):
                                    leave = True
                            if revision_data.tag.endswith('id') and not revision_data.tag.endswith('parentid'):
                                # if int(revision_data.text) < link_struc[current_id]['old_version']:
                                #     old += 1
                                #     version_id = int(revision_data.text)
                                # elif int(revision_data.text) > link_struc[current_id]['new_version']:
                                #     break
                                # else:
                                version_id = int(revision_data.text)
                            if revision_data.tag.endswith('text') and revision_data.text is not None:
                                clean_text = unescape(revision_data.text)
                                # remove all comments
                                # remove multi-line comments
                                clean_text = clean_xml(clean_text)
                                if old == 0:
                                    pages.append(
                                        {'version': version_id, 'text': clean_text})
                                else:
                                    older_pages.append(
                                        {'version': version_id, 'text': clean_text})
                        if leave:
                            break
                if not pages:
                    elem.clear()
                    continue
                pages = pages + older_pages
                pages = sorted(pages, key=lambda x: x['version'], reverse=True)
                versions = [page['version'] for page in pages]

                counts = [{'count': None, 'found': 0, 'expected': link['count'], 'title': process_title(
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
                                for _ in range(count['count'] - new_count):
                                    output.append({'source_title': link_struc[current_id]['page_title'],
                                                   'target_title': link_struc[current_id]['links'][k]['target_title'],
                                                   'source_ID': current_id,
                                                   'target_ID': link_struc[current_id]['links'][k]['target_ID'],
                                                   'first_version': page['version'],
                                                   'second_version': prev_version})
                                    counts[k]['found'] += 1
                                counts[k]['count'] = new_count
                    prev_version = page['version']
                    prev_text = page['text']
                processed_pages += 1
                elem.clear()
                output_len = len(output)

    print("Saving versions")
    output_df = pd.DataFrame(output)
    output_df.to_parquet(os.path.join(
        args.output_dir, f"link_versions.parquet"))
