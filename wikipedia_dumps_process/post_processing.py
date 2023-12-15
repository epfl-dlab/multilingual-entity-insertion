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
    parser.add_argument('--lang', type=str, required=True,
                        help='Language of the Wikipedia dump')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'good_pages')):
        os.makedirs(os.path.join(args.output_dir, 'good_pages'))
    if not os.path.exists(os.path.join(args.output_dir, 'good_links')):
        os.makedirs(os.path.join(args.output_dir, 'good_links'))

    page_files = glob(os.path.join(args.input_dir, "pages/pages*.parquet"))
    link_files = glob(os.path.join(args.input_dir, "links/links*.parquet"))

    no_html = set([])
    no_lead = set([])
    short_lead = set([])
    print('Saving good pages')
    for file in tqdm(page_files):
        df = pd.read_parquet(file)
        no_html = no_html.union(set(df[df['page_length'].isna()]['title'].tolist()))
        no_lead = no_lead.union(set(df[(df['lead_paragraph'].isna()) | (
            df['lead_paragraph'] == '')]['title'].tolist()))
        if args.lang not in ['ja']: # hard-coded languages where the words are not necessarily separated by spaces
            short_lead = short_lead.union(set(df[(df['lead_paragraph'].apply(
                lambda x: split_text(x) < 6))]['title'].tolist()))
        if args.lang not in ['ja']:
            df = df[(~df['QID'].isna()) & (~df['page_length'].isna()) & (~df['lead_paragraph'].isna()) & (
                df['lead_paragraph'] != '') & (df['lead_paragraph'].apply(lambda x: split_text(x) >= 6))]
        else:
            df = df[(~df['QID'].isna()) & (~df['page_length'].isna()) & (~df['lead_paragraph'].isna()) & (
                df['lead_paragraph'] != '')]
        df = df.reset_index(drop=True)
        if 'HTML' in df.columns:        
            df = df.drop(columns=['HTML'])
        basename = os.path.basename(file)
        new_name = os.path.join(args.output_dir, 'good_pages',
                                basename.replace('pages', 'good_pages'))
        df.to_parquet(new_name)

    print('Saving good links')
    for file in tqdm(link_files):
        df = pd.read_parquet(file)
        df = df[(~df['target_ID'].isna()) & (~df['source_QID'].isna()) & (~df['target_QID'].isna()) & (~df['target_title'].isin(no_html)) & (~df['target_title'].isin(no_lead)) & (~df['source_title'].isin(
            no_lead)) & (~df['context'].isna()) & (df['context'] != '') & (~df['source_title'].isin(short_lead)) & (~df['target_title'].isin(short_lead)) & (df['source_title'] != df['target_title'])]
        df['context'] = df['context'].apply(fix_context)
        df = df.reset_index(drop=True)
        basename = os.path.basename(file)
        new_name = os.path.join(args.output_dir, 'good_links',
                                basename.replace('links', 'good_links'))
        df.to_parquet(new_name)