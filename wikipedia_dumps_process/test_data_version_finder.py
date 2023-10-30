import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os
import bz2
from xml.etree.ElementTree import iterparse
from html import unescape
import re

def update_targets(target_name, redirect_map):
    if target_name in redirect_map:
        return redirect_map[target_name]
    return target_name

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
    for file in tqdm(second_month_page_files):
        dfs.append(pd.read_parquet(file))
    df_pages = pd.concat(dfs)
    
    redirect_map = pd.read_parquet(os.path.join(args.second_month_dir, 'redirect_map.parquet'))
    redirect_map = redirect_map.to_dict()['redirect']
    
    df_1['target_title'] = df_1['target_title'].apply(lambda x: update_targets(x, redirect_map))

    print('Converting data into a better structure')
    df_links_1 = df_1.to_dict(orient='records')
    df_links_2 = df_2.to_dict(orient='records')

    for row in tqdm(df_links_1):
        for key in row:
            if 'index' in key and row[key] == row[key]:
                row[key] = int(row[key])

    for row in tqdm(df_links_2):
        for key in row:
            if 'index' in key and row[key] == row[key]:
                row[key] = int(row[key])

    old_data = {}
    for mod_link in tqdm(df_links_1):
        if mod_link['source_title'] not in old_data:
            old_data[mod_link['source_title']] = {}
        if mod_link['target_title'] not in old_data[mod_link['source_title']]:
            old_data[mod_link['source_title']][mod_link['target_title']] = []
        old_data[mod_link['source_title']
                 ][mod_link['target_title']].append(mod_link)

    new_data = {}
    for mod_link in tqdm(df_links_2):
        if mod_link['source_title'] not in new_data:
            new_data[mod_link['source_title']] = {}
        if mod_link['target_title'] not in new_data[mod_link['source_title']]:
            new_data[mod_link['source_title']][mod_link['target_title']] = []
        new_data[mod_link['source_title']
                 ][mod_link['target_title']].append(mod_link)

    no_html = set(df_pages[(df_pages['HTML'].isna()) | (
        df_pages['HTML'] == '')]['title'].tolist())
    no_lead = set(df_pages[(df_pages['lead_paragraph'].isna()) | (
        df_pages['lead_paragraph'] == '')]['title'].tolist())
    short_lead = set(df_pages[df_pages['lead_paragraph'].apply(
        lambda x: x is not None and len(x.split()) < 6)]['title'].tolist())

    print('Finding new links')
    new_pages = 0
    new_page_links = 0
    new_links = []
    no_id_found = 0
    no_id_not_found = 0
    
    for source_page in tqdm(new_data):
        if source_page not in old_data:
            new_pages += 1
            new_page_links += len(new_data[source_page])
            continue
        old_version = old_data[source_page][list(old_data[source_page].keys())[0]][0]['source_version']
        if new_data[source_page][list(new_data[source_page].keys())[0]][0]['source_version'] == old_version:
            continue
        for target_page in new_data[source_page]:
            if target_page not in old_data[source_page]:
                for mod_link in new_data[source_page][target_page]:
                    new_links.append(mod_link)
                    new_links[-1]['old_version'] = old_version
            else:
                # try to find matches in the links without ID
                used = set([])
                for mod_link in new_data[source_page][target_page]:
                    found = False
                    for i, old_link in enumerate(old_data[source_page][target_page]):
                        if old_link['mention'] == mod_link['mention'] and old_link['source_section'] == mod_link['source_section'] and i not in used:
                            used.add(i)
                            found = True
                            no_id_found += 1
                            break
                    if not found:
                        no_id_not_found += 1
                        new_links.append(mod_link)
                        new_links[-1]['old_version'] = old_version

    print(
        f"The new data has {new_pages} new pages and {new_page_links} links in these new pages")
    print(f"There are {len(new_links)} new links in the new data")
    print(f"From the links without ID, {no_id_found} ({no_id_found / (no_id_found + no_id_not_found) * 100:.2f}%) were matched to old links.")

    print('Cleaning the links')
    clean_links = []
    for link in tqdm(new_links):
        if link['target_ID'] is None:
            continue
        if link['source_QID'] is None:
            continue
        if link['source_title'] in no_lead:
            continue
        if link['source_title'] in short_lead:
            continue
        if link['target_QID'] is None:
            continue
        if link['target_title'] in no_html:
            continue
        if link['target_title'] in no_lead:
            continue
        if link['target_title'] in short_lead:
            continue
        if link['target_title'] == mod_link['source_title']:
            continue
        if link['context'] is None:
            continue
        link['context'] = "\n".join(
            line for line in link['context'].split("\n") if line.strip() != '')
        clean_links.append(link)

    print(
        f"Out of the {len(new_links)} new links, {len(clean_links)} ({len(clean_links) / len(new_links) * 100:.2f}%) are valid")
 
    
    link_struc = {}
    for link in clean_links:
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