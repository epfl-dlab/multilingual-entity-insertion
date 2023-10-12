import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_month_input_dir', '-i1', type=str,
                        required=True, help='Input directory for old month')
    parser.add_argument('--new_month_input_dir', '-i2', type=str, 
                        required=True, help='Input directory for new month')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')
    
    args = parser.parse_args()
    
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
        for target_page in new_data[source_page]:
            if target_page not in old_data[source_page]:
                new_links.append(new_data[source_page][target_page])
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
                
                used = set([])
                for new_link in links_without_id:
                    for i, old_link in enumerate(old_data[source_page][target_page]):
                        if old_link['link_ID'] is None and old_link['mention'] == new_link['mention'] and i not in used:
                            used.add(i)
                            break
                        if i == len(old_data[source_page][target_page]) - 1:
                            new_links.append(new_link)
    
    print("Saving new links into a better structure")
    link_struc = {}
    for link in new_links:
        if link['source_ID'] in link_struc:
            link_struc[link['source_ID']].append(link['mention'])
        else:
            link_struc[link['source_ID']] = [link['mention']]
            
        