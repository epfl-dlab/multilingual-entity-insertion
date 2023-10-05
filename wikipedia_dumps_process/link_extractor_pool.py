import argparse
import json
import os
import re
import urllib.parse
import warnings
from glob import glob
from multiprocessing import Pool, cpu_count

import lxml
import pandas as pd
from bs4 import (BeautifulSoup, MarkupResemblesLocatorWarning,
                 NavigableString)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def process_title(title):
    title = title.replace(' ', '_')
    # decode from url encoding
    title = urllib.parse.unquote(title)
    # reencode into url encoding
    title = urllib.parse.quote(title)
    return title


def extract_links(source_page):
    raw_html = source_page['HTML']
    if raw_html is None:
        return [], {}
    parsed_html = BeautifulSoup(raw_html, 'lxml')

    found_links = []
    # find body
    content = parsed_html.find("body")

    # define the sentences
    # split by punctuation if the puctuation is not followed by a letter
    temp_sentences = []
    sentence = ''
    safe = False
    for i, char in enumerate(raw_html):
        if (not safe) and (char in ['.', '!', '?', '\n']) and (i > 1 and raw_html[i-2] != '.' and raw_html[i-2].isalnum()) and (i < len(raw_html) - 1 and not raw_html[i+1].isalnum() and raw_html[i+1] != '_'):
            sentence += char
            temp_sentences.append(sentence)
            sentence = ''
        else:
            if not safe and char == '<' and raw_html[i + 1] == 'a':
                safe = True
            if safe and char == '>' and raw_html[i - 1] == 'a':
                safe = False
            sentence += char
    temp_sentences.append(sentence)

    sentences = []
    start_index = 0
    end_index = 0
    for sentence in temp_sentences:
        # split by <p> or </p>
        for part in re.split('(<p>|</p>)', sentence):
            end_index += len(part)
            sentences.append(
                {'sentence': part, 'start_index': start_index, 'end_index': end_index})
            start_index = end_index

    depth = [0, 0, 0]
    sections = ["Lead"]
    section_text = {'.': source_page['title'], 'Lead': ''}
    search_index_link = 0
    # iterate through all tags
    for section in content.children:
        for tag in section:
            if isinstance(tag, NavigableString):
                continue
            if tag.get("role") == "note":
                continue
            # if the tag is a header tag
            if tag.name in ['h2', 'h3', 'h4']:
                # update section, add or trim section list
                if tag.name == 'h2':
                    sections = [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    if sections[0] not in section_text:
                        section_text[sections[0]] = ''
                    depth[0] += 1
                    depth[1] = 0
                    depth[2] = 0
                elif tag.name == 'h3':
                    sections = sections[:1] + \
                        [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    depth[1] += 1
                    depth[2] = 0
                elif tag.name == 'h4':
                    sections = sections[:2] + \
                        [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    depth[2] += 1
                if sections in [["Notes"], ["References"], ["External links"], ["Further reading"], ["Other websites"]]:
                    break

            # find all p and li tags
            if tag.name == 'p':
                section_text[sections[0]] += tag.text
            elif tag.name == 'li':
                section_text[sections[0]] += tag.text + '\n'
            else:
                tags = tag.find_all(['p', 'li'])
                for t in tags:
                    if tag.name == 'p':
                        section_text[sections[0]] += t.text
                    else:
                        section_text[sections[0]] += t.text + '\n'

            # find links
            links = tag.find_all('a')
            # iterate through the links
            for link in links:
                # if link has class "mw-file-description" then skip it
                if link.get("class") == ["mw-file-description"]:
                    continue
                # if link has class "external text" then skip it
                if link.get("class") == ["external", "text"]:
                    continue
                # if link has class "mw-selflink selflink" then skip it
                if link.get("class") == ["mw-selflink", "selflink"]:
                    continue
                # if link has no href or if href does not start with "/wiki/" then skip it
                if not link.get("rel") or link["rel"] != ["mw:WikiLink"]:
                    continue
                # if link has ':' in it then skip it
                if ':' in link["href"]:
                    continue
                link_data = {}

                # check if there are invalid elements in the tag parents
                # invalid elements are: table, figure, sup
                # also check if the parents have class "mw-editsection"
                invalid = False
                for parent in link.parents:
                    if parent.name in ['table', 'figure', 'sup']:
                        invalid = True
                        break
                    if parent.get("class") == ["mw-editsection"]:
                        invalid = True
                        break
                    if parent.get("class") == ["mw-editsection"]:
                        invalid = True
                        break

                # get the title of the link and the target anchor if it exists
                full_title = link["href"][2:].strip()
                if "#" in full_title:
                    link_data['target_title'] = full_title.split(
                        "#")[0].replace("\\'", "'")
                    link_data['target_section'] = full_title.split(
                        "#")[1].replace("\\'", "'")
                else:
                    link_data['target_title'] = full_title.replace("\\'", "'")
                    link_data['target_section'] = "Lead".replace("\\'", "'")
                link_data['target_title'] = process_title(
                    link_data['target_title'])
                if link_data['target_title'] in redirect_map:
                    link_data['target_title'] = redirect_map[link_data['target_title']]
                link_data['source_title'] = source_page['title']

                if 'redlink' in link_data['target_title'] or link.get("class") == ["new"]:
                    continue

                if link_data['target_title'] not in page_ids:
                    link_data['target_ID'] = None
                    link_data['target_QID'] = None
                else:
                    link_data['target_ID'] = page_ids[link_data['target_title']]['ID']
                    link_data['target_QID'] = page_ids[link_data['target_title']]['QID']

                link_data['source_ID'] = source_page['ID']
                link_data['source_QID'] = source_page['QID']

                # get the text of the link
                link_data['mention'] = link.text

                # get the source section of the links
                link_data['source_section'] = '<sep>'.join(sections)

                # get the start and end index of the link in the text
                if '&' in full_title:
                    full_title = full_title.replace('&', '&amp;')
                try:
                    if "\"" in full_title:
                        try:
                            fixed_title = full_title.replace(
                                '\'', '&apos;')
                            index = raw_html.index(
                                f"./{fixed_title}", search_index_link)
                            full_title = fixed_title
                        except:
                            fixed_title = full_title.replace(
                                '\"', '&quot;')
                            index = raw_html.index(
                                f"./{fixed_title}", search_index_link)
                            full_title = fixed_title
                    else:
                        index = raw_html.index(
                            f"./{full_title}", search_index_link)
                except:
                    link_data['link_start_index'] = None
                    link_data['link_end_index'] = None
                    link_data['sentence'] = None
                    link_data['sentence_start_index'] = None
                    link_data['sentence_end_index'] = None
                else:
                    while raw_html[index] != '<':
                        index -= 1
                    end_index = index
                    while raw_html[end_index-2:end_index+1] != '/a>':
                        end_index += 1
                    search_index_link = end_index
                    link_data['link_start_index'] = index
                    link_data['link_end_index'] = end_index + 1
                    while len(sentences) > 1 and sentences[1]['start_index'] < index:
                        sentences.pop(0)

                    # get the sentence in which the link is present
                    # iterate through the sentences
                    # expensive process, only apply if the link is valid
                    if invalid:
                        continue
                    for j, sentence in enumerate(sentences):
                        # if the link is present in the sentence
                        if f"./{full_title}" in sentence['sentence']:
                            # get the start and end index of the link in the sentence
                            parsed_sentence = BeautifulSoup(
                                sentence['sentence'], 'html.parser')
                            link_data['sentence'] = parsed_sentence.text
                            link_data['sentence_start_index'] = sentence['start_index']
                            link_data['sentence_end_index'] = sentence['end_index']
                            break
                        elif j == len(sentences) - 1:
                            link_data['sentence'] = None
                            link_data['sentence_start_index'] = None
                            link_data['sentence_end_index'] = None
                            print(full_title.encode('utf-8'),
                                  source_page['title'].encode('utf-8'))
                            print('Not found')
                            print(sentences[0]['sentence'])
                link_data['source_page_length'] = len(raw_html)
                link_data['link_section_depth'] = f"{depth[0]}.{depth[1]}.{depth[2]}"

                sentences = sentences[j:]

                found_links.append(link_data)

    return found_links, section_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Path to the data folder')
    parser.add_argument('--page_ids', type=str, required=True,
                        help='Path to the file with IDs and names of all pages')
    parser.add_argument('--redirect_map', type=str, required=True,
                        help='Path to the file with the redirect map')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes to use for multiprocessing')
    args = parser.parse_args()

    # check if input dir exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    # check if page_ids file exists
    if not os.path.exists(args.page_ids):
        raise ValueError(f"Page IDs file {args.page_ids} does not exist")
    # check if redirect map file exists
    if not os.path.exists(args.redirect_map):
        raise ValueError(
            f"Redirect map file {args.redirect_map} does not exist")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def initializer():
        global page_ids
        global redirect_map
        page_ids = pd.read_parquet(args.page_ids)
        redirect_map = pd.read_parquet(args.redirect_map)
        redirect_map = redirect_map[redirect_map.columns[0]].to_dict()
        page_ids = page_ids.to_dict(orient='index')

    # Read all input pages
    files = glob(f"{args.input_dir}/pages*.parquet")
    # remove the page_ids and redirect_map files if they are present
    files = [
        file for file in files if args.page_ids not in file and args.redirect_map not in file]
    files.sort()

    for i, file in (pbar := tqdm(enumerate(files), total=len(files))):
        df = pd.read_parquet(file)
        list_data = []
        for j in range(len(df)):
            list_data.append(df.iloc[j].to_dict())
        links = []
        sections = []

        pool = Pool(min(cpu_count(), args.processes), initializer=initializer)
        for j, (page_links, section_text) in enumerate(pool.imap(extract_links, list_data)):
            if j % 1000 == 0:
                pbar.set_description(
                    f"Processing file {file} at element {j}/{len(df)}")
            for link in page_links:
                links.append(link)
            for section in section_text:
                sections.append(
                    {'section': section, 'text': section_text[section], 'title': section_text['.']})

        pool.close()
        pool.join()

        df_links = pd.DataFrame(links)
        df_links.to_parquet(f"{args.output_dir}/links_{i}.parquet")

        df_sections = pd.DataFrame(sections)
        df_sections.to_parquet(f"{args.output_dir}/sections_{i}.parquet")
