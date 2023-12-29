import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import argparse
tqdm.pandas()


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

    no_html = []
    no_lead = []
    print('Saving good pages')
    for file in tqdm(page_files):
        df = pd.read_parquet(file)
        no_html.extend(df[df['page_length'].isna()]['title'].tolist())
        no_lead.extend(df[(df['lead_paragraph'].isna()) | (
            df['lead_paragraph'] == '')]['title'].tolist())
        df = df[(~df['QID'].isna()) & (~df['page_length'].isna()) & (~df['lead_paragraph'].isna()) & (
            df['lead_paragraph'] != '')]
        df = df.reset_index(drop=True)
        if 'HTML' in df.columns:        
            df = df.drop(columns=['HTML'])
        basename = os.path.basename(file)
        new_name = os.path.join(args.output_dir, 'good_pages',
                                basename.replace('pages', 'good_pages'))
        df.to_parquet(new_name)
    
    no_html = set(no_html)
    no_lead = set(no_lead)

    print('Saving good links')
    for file in tqdm(link_files):
        df = pd.read_parquet(file)
        df = df[(~df['target_ID'].isna()) & (~df['source_QID'].isna()) & (~df['target_QID'].isna()) & (~df['target_title'].isin(no_html)) & (~df['target_title'].isin(no_lead)) & (~df['source_title'].isin(
            no_lead)) & (~df['context'].isna()) & (df['context'] != '') & (df['source_title'] != df['target_title'])]
        df['context'] = df['context'].apply(fix_context)
        df = df.reset_index(drop=True)
        basename = os.path.basename(file)
        new_name = os.path.join(args.output_dir, 'good_links',
                                basename.replace('links', 'good_links'))
        df.to_parquet(new_name)