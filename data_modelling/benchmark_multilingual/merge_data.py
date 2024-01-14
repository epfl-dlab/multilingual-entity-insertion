import pandas as pd
import argparse
import os
from glob import glob
from tqdm import tqdm
import gc
tqdm.pandas()


def fix_title(title, redirect_1, redirect_2):
    counter = 0
    while title in redirect_1:
        title = redirect_1[title]
        counter += 1
        if counter > 10:
            break
    counter = 0
    while title in redirect_2:
        title = redirect_2[title]
        counter += 1
        if counter > 10:
            break
    return title


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing the data')
    parser.add_argument('--langs', type=str, nargs='+',
                        required=True, help='Languages to merge')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output dir')
    parser.add_argument('--max_samples_per_lang', '-n', type=int,
                        default=1000, help='Maximum number of samples per language')

    args = parser.parse_args()
    
    # check if input dir exists
    if not os.path.exists(args.data_dir):
        raise ValueError('Data directory does not exist')
    
    # check if all the language specific files exist
    for lang in args.langs:
        if not os.path.exists(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'eval', 'test_data.parquet')):
            raise ValueError(f'File for language {lang} does not exist')
        if not os.path.exists(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'processed_data', 'redirect_map.parquet')):
            raise ValueError(f'Redirect file 1 for language {lang} does not exist')
        if not os.path.exists(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231101', 'processed_data', 'redirect_map.parquet')):
            raise ValueError(f'Redirect file 2 for language {lang} does not exist')
        if not os.path.exists(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'processed_data', 'good_pages')):
            raise ValueError(f'Pages files for language {lang} do not exist')
    
    # check if output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print('Saving mention maps')
    for i, lang in enumerate(args.langs):
        print(f'Processing language {lang} ({i+1}/{len(args.langs)})')
        df_mentions = pd.read_parquet(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'processed_data', 'mention_map.parquet'))
        df_mentions.to_parquet(os.path.join(args.output_dir, f'{lang}_mention_map.parquet'))
        df_mentions = None
        del df_mentions
        gc.collect()
        print(f'Finished processing language {lang}')
        
    # process each language sequentially
    for i, lang in enumerate(args.langs):
        print(f'Processing language {lang} ({i+1}/{len(args.langs)})')
        
        # load links
        df_links = pd.read_parquet(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'eval', 'test_data.parquet'))
        print('Loaded links')
        
        # load pages
        files = glob(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'processed_data', 'good_pages*'))
        df_pages = pd.concat([pd.read_parquet(file, columns=['title', 'lead_paragraph'])
                             for file in files]).reset_index(drop=True)
        print('Loaded pages')
        
        # load redirects
        df_redirect_1 = pd.read_parquet(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231001', 'processed_data', 'redirect_map.parquet'))
        df_redirect_2 = pd.read_parquet(os.path.join(args.data_dir, f'{lang}wiki-NS0-20231101', 'processed_data', 'redirect_map.parquet'))
        # redirects have 'title' as index, and'redirect' as column
        # convert to dict
        redirect_1 = df_redirect_1.to_dict()['redirect']
        redirect_2 = df_redirect_2.to_dict()['redirect']
        df_redirect_1 = None
        del df_redirect_1
        df_redirect_2 = None
        del df_redirect_2
        print('Loaded redirects')

        # fix titles
        df_pages['title'] = df_pages['title'].progress_apply(lambda x: fix_title(x, redirect_1, redirect_2))
        redirect_1 = None
        del redirect_1
        redirect_2 = None
        del redirect_2
        # create a dict where the keys are the titles and the values are the lead paragraphs
        pages = df_pages.set_index('title').to_dict()['lead_paragraph']
        df_pages = None
        del df_pages
        
        print(f"Originally, there are {len(df_links)} links for language {lang}")
        
        df_removed = df_links['context'] == ''
        df_links = df_links[~df_removed]
        print(f"Removed {df_removed.sum()} links because of empty context")
        
        df_removed = df_links['negative_contexts'] == '[]'
        df_links = df_links[~df_removed]
        print(f"Removed {df_removed.sum()} links because of empty negative contexts")
        
        df_removed = ~df_links['target_title'].isin(pages)
        df_links = df_links[~df_removed]
        print(f"Removed {df_removed.sum()} links because target title is not in source titles")
        
        df_removed = df_links['missing_category'] == 'missing_section'
        df_links = df_links[~df_removed]
        print(f"Removed {df_removed.sum()} links because of missing section")
        
        print(f"After cleaning, there are {len(df_links)} links for language {lang}")
        
        if len(df_links) > args.max_samples_per_lang:
            print(f'There are {len(df_links)} links for language {lang}, sampling {args.max_samples_per_lang}')
            df_links = df_links.sample(args.max_samples_per_lang)
        
        df_links['target_lead'] = df_links['target_title'].apply(lambda x: pages[x])
        df_links = df_links.reset_index(drop=True)
        df_links.to_parquet(os.path.join(args.output_dir, f'{lang}.parquet'))
        
        df_links = None
        del df_links
        pages = None
        del pages
        gc.collect()
        print(f'Finished processing language {lang}')