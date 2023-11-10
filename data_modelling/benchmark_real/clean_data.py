import pandas as pd
import argparse
import os
from glob import glob


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
    parser.add_argument('--links_file', type=str, required=True,
                        help='Path to the file containing the links')
    parser.add_argument('--pages_dir', type=str, required=True,
                        help='Path to the directory containing the pages')
    parser.add_argument('--redirect_file_1', type=str,
                        required=True, help='Path to the first redirect file')
    parser.add_argument('--redirect_file_2', type=str,
                        required=True, help='Path to the second redirect file')
    parser.add_argument('--output_file', type=str,
                        required=True, help='Path to the output file')

    args = parser.parse_args()

    # check if all input files exist
    if not os.path.exists(args.links_file):
        raise ValueError('Links file does not exist')
    if not os.path.exists(args.pages_dir):
        raise ValueError('Pages directory does not exist')
    if not os.path.exists(args.redirect_file_1):
        raise ValueError('Redirect file 1 does not exist')
    if not os.path.exists(args.redirect_file_2):
        raise ValueError('Redirect file 2 does not exist')

    # check if output file exists
    if os.path.dirname(args.output_file) and not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # load links
    df_links = pd.read_parquet(args.links_file)

    # load pages
    files = glob(os.path.join(args.pages_dir, 'good_pages*'))
    df_pages = pd.concat([pd.read_parquet(file)
                         for file in files]).reset_index(drop=True)

    # load redirects
    df_redirect_1 = pd.read_parquet(args.redirect_file_1)
    df_redirect_2 = pd.read_parquet(args.redirect_file_2)
    # redirects have 'title' as index, and 'redirect' as column
    # convert to dict
    redirect_1 = df_redirect_1.to_dict()['redirect']
    redirect_2 = df_redirect_2.to_dict()['redirect']

    # fix titles in pages
    df_pages['title'] = df_pages['title'].apply(
        lambda x: fix_title(x, redirect_1, redirect_2))
    # create a dictionary where the keys are the titles and the values are the leads
    pages = df_pages.set_index('title').to_dict()['lead_paragraph']

    print(f"Originally, there are {len(df_links)} links")

    df_removed = df_links['context'] == ''
    df_links = df_links[~df_removed]
    print(f"Removed {df_removed.sum()} links because of empty context")

    df_removed = df_links['negative_contexts'] == '[]'
    df_links = df_links[~df_removed]
    print(
        f"Removed {df_removed.sum()} links because of empty negative contexts")

    df_removed = ~df_links['target_title'].isin(pages)
    df_links = df_links[~df_removed]
    print(
        f"Removed {df_removed.sum()} links because target title is not in source titles")
    
    df_removed = df_links['missing_category'] == 'missing_section'
    df_links = df_links[~df_removed]
    print(
        f"Removed {df_removed.sum()} links because of missing section")

    print(f"After cleaning, there are {len(df_links)} links")
    df_links['target_lead'] = df_links['target_title'].apply(
        lambda x: pages[x])
    df_links = df_links.reset_index(drop=True)
    df_links.to_parquet(args.output_file)
