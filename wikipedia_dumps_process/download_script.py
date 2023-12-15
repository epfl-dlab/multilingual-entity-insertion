import argparse
import gzip
import os
import urllib.request

import requests
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json


# code taken from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True,
                        help='language of the wikipedia dump')
    parser.add_argument('--date', type=str, required=True,
                        help='date of the wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--namespace', type=str, default='0',
                        help='namespace of the wikipedia dump')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='output directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing files')
    parser.add_argument('--download_history', action='store_true', default=False,
                        help='download revision history xml dump')

    parser.set_defaults(overwrite=False, download_history=False)
    args = parser.parse_args()

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    url_xml_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-pages-meta-history.xml.bz2"
    output_xml_path = f"{args.output_dir}/{args.lang}wiki-{args.date}-pages-meta-history.xml.bz2"
    
    backup_json_dump = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/dumpstatus.json"
    output_backup_path = f"{args.output_dir}/dumpstatus.json"

    url_html_dump = f"https://dumps.wikimedia.org/other/enterprise_html/runs/{args.date}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"
    output_html_path = f"{args.output_dir}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"

    url_sql_dump_1 = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-page_props.sql.gz"
    output_sql_path_1 = f"{args.output_dir}/{args.lang}wiki-{args.date}-page_props.sql.gz"

    url_sql_dump_2 = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-page.sql.gz"
    output_sql_path_2 = f"{args.output_dir}/{args.lang}wiki-{args.date}-page.sql.gz"

    url_sql_dump_3 = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-redirect.sql.gz"
    output_sql_path_3 = f"{args.output_dir}/{args.lang}wiki-{args.date}-redirect.sql.gz"
            
    if not os.path.exists(output_html_path) or args.overwrite:
        download_url(url_html_dump, output_html_path)
    if not os.path.exists(output_sql_path_1) or args.overwrite:
        download_url(url_sql_dump_1, output_sql_path_1)
    if not os.path.exists(output_sql_path_2) or args.overwrite:
        download_url(url_sql_dump_2, output_sql_path_2)
    if not os.path.exists(output_sql_path_3) or args.overwrite:
        download_url(url_sql_dump_3, output_sql_path_3)

    # extract the downloaded files (sql)
    if not os.path.exists(output_sql_path_1[:-3]) or args.overwrite:
        print(f'Extracting {output_sql_path_1}')
        with gzip.open(output_sql_path_1, 'rb') as f:
            file_content = f.read()
            with open(output_sql_path_1[:-3], 'wb') as f_out:
                f_out.write(file_content)

    if not os.path.exists(output_sql_path_2[:-3]) or args.overwrite:
        print(f'Extracting {output_sql_path_2}')
        with gzip.open(output_sql_path_2, 'rb') as f:
            file_content = f.read()
            with open(output_sql_path_2[:-3], 'wb') as f_out:
                f_out.write(file_content)

    if not os.path.exists(output_sql_path_3[:-3]) or args.overwrite:
        print(f'Extracting {output_sql_path_3}')
        with gzip.open(output_sql_path_3, 'rb') as f:
            file_content = f.read()
            with open(output_sql_path_3[:-3], 'wb') as f_out:
                f_out.write(file_content)
            
    if not args.download_history:
        exit(0)
    print('Downloading revision history xml dump')
    # check if files exist before downloading
    if not os.path.exists(output_xml_path) or args.overwrite:
        try:
            download_url(url_xml_dump, output_xml_path)
        except urllib.error.HTTPError:
            print(f'Could not download {url_xml_dump}. This most likely means the dump is too large and the revision history is split into multiple files.')
            print(f'Downloading json with revision history details.')
            download_url(backup_json_dump, output_backup_path)
            print('Downloading all revision history files using multiprocessing.')
            with open(output_backup_path, 'r') as f:
                backup_json = json.load(f)
            data = backup_json['jobs']['metahistorybz2dump']['files']
            print(f"Downloading {len(data)} files")
            urls = [f"https://dumps.wikimedia.org{data[f]['url']}" for f in data]
            paths = [f"{args.output_dir}/{f}" for f in data if args.overwrite or not os.path.exists(f"{args.output_dir}/{f}")]
            with Pool(3) as p:
                p.starmap(download_url, zip(urls, paths))
            
            
