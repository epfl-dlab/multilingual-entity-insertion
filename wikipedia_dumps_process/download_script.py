import argparse
import requests
import urllib.request
from tqdm import tqdm
import os
import bz2
import gzip

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
    parser.add_argument('--lang', type=str, required=True, help='language of the wikipedia dump')
    parser.add_argument('--date', type=str, required=True, help='date of the wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--namespace', type=str, default='0', help='namespace of the wikipedia dump')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    
    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    url_html_dump = f"https://dumps.wikimedia.org/other/enterprise_html/runs/{args.date}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"
    output_html_path = f"{args.output_dir}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"
    
    url_xml_dump_1 = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-pages-articles.xml.bz2"
    output_xml_path_1 = f"{args.output_dir}/{args.lang}wiki-{args.date}-pages-articles.xml.bz2"
    
    url_xml_dump_2 = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.date}/{args.lang}wiki-{args.date}-page_props.sql.gz"
    output_xml_path_2 = f"{args.output_dir}/{args.lang}wiki-{args.date}-page_props.sql.gz"
    
    # download_url(url_html_dump, output_html_path)
    download_url(url_xml_dump_1, output_xml_path_1)
    download_url(url_xml_dump_2, output_xml_path_2)
    
    # extract the downloaded files (xml and sql only)
    with bz2.open(output_xml_path_1, 'rb') as f:
        file_content = f.read()
        with open(output_xml_path_1[:-4], 'wb') as f_out:
            f_out.write(file_content)
    
    with gzip.open(output_xml_path_2, 'rb') as f:
        file_content = f.read()
        with open(output_xml_path_2[:-3], 'wb') as f_out:
            f_out.write(file_content)