import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import argparse
tqdm.pandas()


def simplify_html(html):
    if html is None:
        return None
    if html == '':
        return ''
    return 'a'


def split_text(x):
    if x is None:
        return 0
    return len(x.split(' ', 10))


def fix_context(x):
    if x is None:
        return None
    clean_position = x.find('v t e')
    if clean_position == -1:
        return x
    return x[:clean_position]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')

    args = parser.parse_args()

    page_files = glob(os.path.join(args.input_dir, "pages*.parquet"))
    link_files = glob(os.path.join(args.input_dir, "links*.parquet"))

    print('Loading data')
    dfs = []
    for file in tqdm(page_files):
        temp_df = pd.read_parquet(file)
        temp_df['HTML'] = temp_df['HTML'].apply(
            lambda x: simplify_html(x))  # simpify html so it is not too big
        dfs.append(temp_df)
    df_pages = pd.concat(dfs).reset_index(drop=True)

    print('Saving good pages')
    for file in tqdm(page_files):
        df = pd.read_parquet(file)
        df = df[(~df['QID'].isna()) & (~df['HTML'].isna()) & (~df['lead_paragraph'].isna()) & (df['HTML'] != '') & (
            df['lead_paragraph'] != '') & (df['lead_paragraph'].apply(lambda x: split_text(x) >= 6))]
        df = df.reset_index(drop=True)
        df = df.drop(columns=['HTML'])
        basename = os.path.basename(file)
        df.to_parquet(os.path.join(args.output_dir,
                      basename.replace('pages', 'good_pages')))

    print('Building auxiliary data structures')
    no_html = df_pages[(df_pages['HTML'].isna()) | (
        df_pages['HTML'] == '')]['title'].tolist()
    no_lead = df_pages[(df_pages['lead_paragraph'].isna()) | (
        df_pages['lead_paragraph'] == '')]['title'].tolist()
    short_lead = df_pages[(df_pages['lead_paragraph'].apply(
        lambda x: split_text(x) < 6))]['title'].tolist()

    print('Saving good links')
    for file in tqdm(link_files):
        df = pd.read_parquet(file)
        df['context'] = df['context'].apply(fix_context)
        df = df[(~df['target_ID'].isna()) & (~df['source_QID'].isna()) & (~df['target_QID'].isna()) & (~df['target_title'].isin(no_html)) & (~df['target_title'].isin(no_lead)) & (~df['source_title'].isin(
            no_lead)) & (~df['context'].isna()) & (df['context'] != '') & (~df['source_title'].isin(short_lead)) & (~df['target_title'].isin(short_lead)) & (df['source_title'] != df['target_title'])]
        df = df.reset_index(drop=True)
        basename = os.path.basename(file)
        df.to_parquet(os.path.join(args.output_dir,
                      basename.replace('links', 'good_links')))
