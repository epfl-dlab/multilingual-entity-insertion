import argparse
import requests
import urllib.request
from tqdm import tqdm
import os
import tarfile

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
    parser.add_argument('--untar', dest='untar', action='store_true', help='untar the downloaded file')
    parser.add_argument('--delete_tar', dest='delete_tar', action='store_true', help='delete the tar file after untarring')
    parser.set_defaults(untar=False, delete_tar=False)
    args = parser.parse_args()
    
    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    url = f"https://dumps.wikimedia.org/other/enterprise_html/runs/{args.date}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"
    output_path = f"{args.output_dir}/{args.lang}wiki-NS{args.namespace}-{args.date}-ENTERPRISE-HTML.json.tar.gz"
    
    download_url(url, output_path)
    
    if args.untar:
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall(f"{args.output_dir}/raw_data")
        if args.delete_tar:
            os.remove(output_path)