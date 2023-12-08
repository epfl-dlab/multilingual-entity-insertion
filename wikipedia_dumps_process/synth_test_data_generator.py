import argparse
import os
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import urllib
import re
from ast import literal_eval
import json
import gc
from multiprocessing import Pool
import psutil
import sys

def update_targets(target_name, redirect_map):
    counter = 0
    while target_name in redirect_map:
        target_name = redirect_map[target_name]
        counter += 1
        if counter > 10:
            break
    return target_name

def simplify_html(html):
    if html is None:
        return None
    if html == '':
        return ''
    return 'a'
    
def compare_pages(input):
    output = {page['page_title']: {'links': [], 'found': 0, 'not_found': 0} for page in input}
    
    for page in input:
        old_page = page['old_page']
        new_page = page['new_page']
        old_version = old_page[list(old_page.keys())[0]][0]['source_version']
        if new_page[list(new_page.keys())[0]][0]['source_version'] == old_version:
            continue
        for target_page in new_page:
            if target_page not in old_page:
                for mod_link in new_page[target_page]:
                    output[page['page_title']]['links'].append(mod_link)
                    output[page['page_title']]['links'][-1]['old_version'] = old_version
            else:
                used = set([])
                for mod_link in new_page[target_page]:
                    found = False
                    for i, old_link in enumerate(old_page[target_page]):
                        if old_link['mention'] == mod_link['mention'] and old_link['source_section'] == mod_link['source_section'] and i not in used:
                            used.add(i)
                            found = True
                            output[page['page_title']]['found'] += 1
                            break
                    if not found:
                        output[page['page_title']]['not_found'] += 1
                        output[page['page_title']]['links'].append(mod_link)
                        output[page['page_title']]['links'][-1]['old_version'] = old_version
    return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--first_month_dir', '-i1', type=str,
                        required=True, help='Path to the first month directory')
    parser.add_argument('--second_month_dir', '-i2', type=str,
                        required=True, help='Path to the second month directory')
    parser.add_argument('--output_dir', '-o', type=str,
                        required=True, help='Path to the output directory')
    parser.add_argument('--no_mask_perc', type=float, default=0.4,
                        help='Percentage of links where no masking is applied')
    parser.add_argument('--mask_mention_perc', type=float, default=0.2,
                        help='Percentage of links where mention masking is applied')
    parser.add_argument('--mask_sentence_perc', type=float, default=0.3,
                        help='Percentage of links where sentence masking is applied')
    parser.add_argument('--mask_paragraph_perc', type=float, default=0.1,
                        help='Percentage of links where paragraph masking is applied')

    args = parser.parse_args()

    # check if input directories exist
    if not os.path.exists(args.first_month_dir):
        raise Exception('First month directory does not exist')
    if not os.path.exists(args.second_month_dir):
        raise Exception('Second month directory does not exist')
    
    # check if percentage arguments are valid
    if abs(args.no_mask_perc + args.mask_mention_perc + args.mask_sentence_perc + args.mask_paragraph_perc - 1) > 1e-5:
        raise Exception('The sum of the masking percentages should be 1')
    if args.no_mask_perc < 0 or args.no_mask_perc > 1:
        raise Exception('The no mask percentage should be between 0 and 1')
    if args.mask_mention_perc < 0 or args.mask_mention_perc > 1:
        raise Exception('The mask mention percentage should be between 0 and 1')
    if args.mask_sentence_perc < 0 or args.mask_sentence_perc > 1:
        raise Exception('The mask sentence percentage should be between 0 and 1')
    if args.mask_paragraph_perc < 0 or args.mask_paragraph_perc > 1:
        raise Exception('The mask paragraph percentage should be between 0 and 1')

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get all the link files in the month directories
    first_month_link_files = glob(os.path.join(args.first_month_dir, "links_*"))
    second_month_link_files = glob(
        os.path.join(args.second_month_dir, "links_*"))
    first_month_link_files.sort()
    second_month_link_files.sort()

    # get the page files for the second month
    second_month_page_files = glob(
        os.path.join(args.second_month_dir, "pages_*"))
    
    redirect_map = pd.read_parquet(os.path.join(args.second_month_dir, 'redirect_map.parquet'))
    redirect_map = redirect_map.to_dict()['redirect']
    
    old_data = {}
    new_data = {}
    new_pages = 0
    new_page_links = 0
    new_links = []
    no_id_found = 0
    no_id_not_found = 0
    quit_counter = 0
    needed_columns = ['source_title', 'source_version', 'source_section', 'target_title', 'mention', 'context', 'context_span_start_index', 'context_span_end_index', 'context_sentence_start_index', 'context_sentence_end_index', 'context_mention_start_index', 'context_mention_end_index']
    for i in (pbar := tqdm(range(0, max(len(first_month_link_files), len(second_month_link_files)), 25))):
        new_1 = False
        new_2 = False
        if i < len(first_month_link_files):
            pbar.set_description(f"Processing first month files ({i}/{len(first_month_link_files)})")
            df_1 = pd.concat([pd.read_parquet(file, columns=needed_columns) for file in first_month_link_files[i:i+25]])
            df_1['target_title'] = df_1['target_title'].apply(lambda x: update_targets(x, redirect_map))
            for column in df_1.columns:
                if 'index' in column:
                    df_1[column] = df_1[column].apply(lambda x: int(x) if x == x else x)
            df_1 = df_1.to_dict(orient='records')
            new_1 = True 
        if i < len(second_month_link_files):
            pbar.set_description(f"Processing second month files ({i}/{len(second_month_link_files)})")
            df_2 = pd.concat([pd.read_parquet(file, columns=needed_columns) for file in second_month_link_files[i:i+25]])
            df_2['target_title'] = df_2['target_title'].apply(lambda x: update_targets(x, redirect_map))
            for column in df_2.columns:
                if 'index' in column:
                    df_2[column] = df_2[column].apply(lambda x: int(x) if x == x else x)
            df_2 = df_2.to_dict(orient='records')
            new_2 = True
        
        if new_1:
            pbar.set_description(f"Updating old data ({len(old_data)} pages)")
            for row in df_1:
                if row['source_title'] not in old_data:
                    old_data[row['source_title']] = {}
                if row['target_title'] not in old_data[row['source_title']]:
                    old_data[row['source_title']][row['target_title']] = []
                old_data[row['source_title']][row['target_title']].append(row)
            del df_1
            gc.collect()
        if new_2:
            pbar.set_description(f"Updating new data ({len(new_data)} pages))")
            for row in df_2:
                if row['source_title'] not in new_data:
                    new_data[row['source_title']] = {}
                if row['target_title'] not in new_data[row['source_title']]:
                    new_data[row['source_title']][row['target_title']] = []
                new_data[row['source_title']][row['target_title']].append(row)
            del df_2
            gc.collect()
        
        pool_input = []
        pages = []
        new_pages = list(new_data.keys())
        for page in new_pages:
            if page not in old_data:
                continue
            # check if the versions are the same
            if old_data[page][list(old_data[page].keys())[0]][0]['source_version'] == new_data[page][list(new_data[page].keys())[0]][0]['source_version']:
                print('VERSION MATCH', page)
                del old_data[page]
                del new_data[page]
                continue
            pages.append({'old_page': old_data[page], 'new_page': new_data[page], 'page_title': page})
            if len(pages) == 1000:
                pool_input.append(pages)
                pages = []
        if len(pages) > 0:
            pool_input.append(pages)
        del pages
        gc.collect()
        
        if len(pool_input) == 0:
            continue
        
        pbar.set_description(f"Processing data ({len(old_data)} pages in old_data, {len(new_data)} pages in new_data)")
        counter = 0
        with Pool(5) as p:
            for output in p.imap_unordered(compare_pages, pool_input):
                counter += 1
                pbar.set_description(f"Processing data: {counter}/{len(pool_input)} ({len(old_data)} pages in old_data, {len(new_data)} pages in new_data, {len(new_links)} new links)")
                for page in output:
                    new_links.extend(output[page]['links'])
                    no_id_found += output[page]['found']
                    no_id_not_found += output[page]['not_found']
                    del new_data[page]
                    del old_data[page]
                gc.collect()
        del pool_input
        gc.collect()
        
    new_data_keys = list(new_data.keys())
    for source_page in new_data_keys:
        if source_page not in old_data:
            new_pages += 1
            new_page_links += len(new_data[source_page])
            del new_data[source_page]
            gc.collect()
            continue
        old_version = old_data[source_page][list(old_data[source_page].keys())[0]][0]['source_version']
        if new_data[source_page][list(new_data[source_page].keys())[0]][0]['source_version'] == old_version:
            del new_data[source_page]
            del old_data[source_page]
            gc.collect()
            continue
        for target_page in new_data[source_page]:
            if target_page not in old_data[source_page]:
                for mod_link in new_data[source_page][target_page]:
                    new_links.append(mod_link)
                    new_links[-1]['old_version'] = old_version
            else:
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
        del new_data[source_page]
        del old_data[source_page]
        gc.collect()
    del new_data
    del old_data
    gc.collect()

    print(
        f"The new data has {new_pages} new pages and {new_page_links} links in these new pages")
    print(f"There are {len(new_links)} new links in the new data")
    print(f"From the links without ID, {no_id_found} ({no_id_found / (no_id_found + no_id_not_found) * 100:.2f}%) were matched to old links.")

    print('Loading pages')
    dfs = []
    for file in tqdm(second_month_page_files):
        temp = pd.read_parquet(file, columns=['title', 'HTML', 'lead_paragraph'])
        temp['HTML'] = temp['HTML'].apply(simplify_html)
        dfs.append(temp)
        del temp
    df_pages = pd.concat(dfs)
    del dfs

    no_html = set(df_pages[(df_pages['HTML'].isna()) | (
        df_pages['HTML'] == '')]['title'].tolist())
    no_lead = set(df_pages[(df_pages['lead_paragraph'].isna()) | (
        df_pages['lead_paragraph'] == '')]['title'].tolist())
    short_lead = set(df_pages[df_pages['lead_paragraph'].apply(
        lambda x: x is not None and len(x.split()) < 6)]['title'].tolist())

    del df_pages
    gc.collect()

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

    del new_links
    gc.collect()

    print('Triaging links to know which corruptions can be applied')
    mask_paragraph_links = []
    mask_mention_links = []
    mask_sentence_links = []
    no_mask_links = []
    for link in clean_links:
        # mask span
        if (link['context'][:link['context_span_start_index']] + link['context'][link['context_span_end_index']:]).strip() != '':
            if link['context_span_start_index'] <= link['context_sentence_start_index'] and link['context_span_end_index'] >= link['context_sentence_end_index']:
                mask_paragraph_links.append(link)
                continue

        # mask sentence
        if (link['context'][:link['context_sentence_start_index']] + link['context'][link['context_sentence_end_index']:]).strip() != '':
            if link['context_sentence_start_index'] <= link['context_mention_start_index'] and link['context_sentence_end_index'] > link['context_mention_end_index'] + 1:
                mask_sentence_links.append(link)
                continue

        # mask mention
        if (link['context'][:link['context_mention_start_index']] + link['context'][link['context_mention_end_index']:]).strip() != '':
            mask_mention_links.append(link)
            continue

        # no mask
        no_mask_links.append(link)

    print("Applying the corruptions")
    final_links = []
    random.shuffle(mask_paragraph_links)
    if len(mask_paragraph_links) < args.mask_paragraph_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_paragraph_perc * 100}% paragraph masking request. Using {len(mask_paragraph_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the mask sentence percentage to {args.mask_sentence_perc * 100 + (args.mask_paragraph_perc - len(mask_paragraph_links) / len(clean_links)) * 100:.2f}%')
        args.mask_sentence_perc += args.mask_paragraph_perc - \
            len(mask_paragraph_links) / len(clean_links)
    for link in mask_paragraph_links[:int(len(clean_links) * args.mask_paragraph_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_span_start_index'])] + mod_link['context'][int(mod_link['context_span_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = re.sub('\n+', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_span'
        # current_links = literal_eval(mod_link['current_links'])
        # clean_contexts_links = {}
        # for target in current_links:
        #     if current_links[target]['region'] in ['sentence', 'span']:
        #         continue
        #     clean_contexts_links[target] = current_links[target]
        # mod_link['current_links'] = str(clean_contexts_links)
        final_links.append(mod_link)

    mask_sentence_links.extend(mask_paragraph_links[int(
        len(clean_links) * args.mask_paragraph_perc):])
    random.shuffle(mask_sentence_links)
    if len(mask_sentence_links) < args.mask_sentence_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_sentence_perc * 100}% sentence masking request. Using {len(mask_sentence_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the mask mention percentage to {args.mask_mention_perc * 100 + (args.mask_sentence_perc - len(mask_sentence_links) / len(clean_links)) * 100:.2f}%')
        args.mask_mention_perc += args.mask_sentence_perc - \
            len(mask_sentence_links) / len(clean_links)
    for link in mask_sentence_links[:int(len(clean_links) * args.mask_sentence_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_sentence_start_index'])] + mod_link['context'][int(mod_link['context_sentence_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_sentence'
        # current_links = literal_eval(mod_link['current_links'])
        # clean_contexts_links = {}
        # for target in current_links:
        #     if current_links[target]['region'] in ['sentence']:
        #         continue
        #     clean_contexts_links[target] = current_links[target]
        # mod_link['current_links'] = str(clean_contexts_links)
        final_links.append(mod_link)

    mask_mention_links.extend(mask_sentence_links[int(
        len(clean_links) * args.mask_sentence_perc):])
    random.shuffle(mask_mention_links)
    if len(mask_mention_links) < args.mask_mention_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_mention_perc * 100}% mention masking request. Using {len(mask_mention_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the no mask percentage to {args.no_mask_perc * 100 + (args.mask_mention_perc - len(mask_mention_links) / len(clean_links)) * 100:.2f}%')
        args.no_mask_perc += args.mask_mention_perc - \
            len(mask_mention_links) / len(clean_links)
    for link in mask_mention_links[:int(len(clean_links) * args.mask_mention_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_mention_start_index'])] + mod_link['context'][int(mod_link['context_mention_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_mention'
        final_links.append(mod_link)

    no_mask_links.extend(mask_mention_links[int(
        len(clean_links) * args.mask_mention_perc):])
    random.shuffle(no_mask_links)
    for link in no_mask_links:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'no_mask'
        final_links.append(mod_link)

    print('In the end, we have the following distribution:')
    print(f"\t- Mask span: {len([link for link in final_links if link['noise_strategy'] == 'mask_span'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_span']) / len(final_links) * 100:.2f}%)")
    print(f"\t- Mask sentence: {len([link for link in final_links if link['noise_strategy'] == 'mask_sentence'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_sentence']) / len(final_links) * 100:.2f}%)")
    print(f"\t- Mask mention: {len([link for link in final_links if link['noise_strategy'] == 'mask_mention'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_mention']) / len(final_links) * 100:.2f}%)")
    print(f"\t- No mask: {len([link for link in final_links if link['noise_strategy'] == 'no_mask'])} ({len([link for link in final_links if link['noise_strategy'] == 'no_mask']) / len(final_links) * 100:.2f}%)")

    print("Saving data")
    df = pd.DataFrame(final_links)
    df.to_parquet(os.path.join(args.output_dir, 'val_links.parquet'))
