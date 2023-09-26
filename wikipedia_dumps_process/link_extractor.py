import argparse
import json
import os
import re
import urllib.parse
from glob import glob

import lxml
import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString
from tqdm import tqdm


def process_title(title):
    return urllib.parse.quote(title.replace(' ', '_'))


def extract_links(source_page, page_ids, redirect_map):
    raw_html = source_page['HTML']
    parsed_html = BeautifulSoup(raw_html, 'lxml')

    node = {}
    links = []
    # find the div with class mw-parser-output
    content = parsed_html.find("div", {"id": "mw-content-text"})
    content = content.find("div", {"class": "mw-parser-output"})

    # define the sentences
    # split by punctuation if the puctuation is not followed by a letter
    temp_sentences = []
    sentence = ''
    safe = False
    for i, char in enumerate(raw_html):
        if not safe and char in ['.', '!', '?', '\n'] and (i > 1 and raw_html[i-2] != '.' and raw_html[i-2].isalnum()) and (j < len(raw_html) - 1 and not raw_html[i+1].isalnum() and raw_html[i+1] != '_'):
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

    section = ["Lead"]
    lead_paragraph = ''
    search_index_link = 0
    # iterate through all tags
    for tag in content:
        if isinstance(tag, NavigableString):
            continue
        if tag.get("role") == "note":
            continue
        # if the tag is a header tag
        if tag.name in ['h2', 'h3', 'h4']:
            # update section, add or trim section list
            if tag.name == 'h2':
                section = [tag.text.replace('[edit]', '').strip()]
            elif tag.name == 'h3':
                section = section[:1] + \
                    [tag.text.replace('[edit]', '').strip()]
            elif tag.name == 'h4':
                section = section[:2] + \
                    [tag.text.replace('[edit]', '').strip()]
            if section in [["Notes"], ["References"], ["External links"], ["Further reading"]]:
                break

        if tag.name == 'p' and section == ["Lead"]:
            lead_paragraph += tag.text

        # find links
        links = tag.find_all('a')
        # if there are links in the paragraph
        if links:
            # iterate through the links
            for link in links:
                # if link has class "mw-file-description" then skip it
                if link.get("class") == ["mw-file-description"]:
                    continue
                # if link has class "mw-redirect" then skip it
                if link.get("class") == ["mw-redirect"]:
                    continue
                # if link has class "external text" then skip it
                if link.get("class") == ["external", "text"]:
                    continue
                # if link has class "mw-selflink selflink" then skip it
                if link.get("class") == ["mw-selflink", "selflink"]:
                    continue
                # if link has no href or if href does not start with "/wiki/" then skip it
                if not link.get("href") or not link["href"].startswith("/wiki/"):
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
                full_title = link["href"][6:]
                if "#" in full_title:
                    link_data['target_title'] = full_title.split("#")[0]
                    link_data['target_section'] = full_title.split("#")[1]
                else:
                    link_data['target_title'] = full_title
                    link_data['target_section'] = "Lead"
                if link_data['target_title'] in redirect_map:
                    link_data['target_title'] = redirect_map[link_data['target_title']]

                link_data['source_title'] = source_page['title']

                link_data['source_ID'] = source_page['ID']
                link_data['target_ID'] = page_ids[link_data['target_title']]['ID']

                link_data['source_QID'] = source_page['QID']
                link_data['target_QID'] = page_ids[link_data['target_title']]['QID']

                # get the text of the link
                link_data['anchor'] = link.text

                # get the source section of the link
                link_data['source_section'] = '<sep>'.join(section)

                # get the start and end index of the link in the text
                index = raw_html[i].index(
                    f"/wiki/{full_title}", search_index_link)
                while raw_html[i][index] != '<':
                    index -= 1
                end_index = index
                while raw_html[i][end_index] != '>':
                    end_index += 1
                search_index_link = end_index
                link_data['link_start_index'] = index
                link_data['link_end_index'] = end_index
                while len(sentences) > 1 and sentences[1]['start_index'] < index:
                    sentences.pop(0)

                # get the sentence in which the link is present
                # iterate through the sentences
                # expensive process, only apply if the link is valid
                if invalid:
                    continue
                for j, sentence in enumerate(sentences):
                    # if the link is present in the sentence
                    if f"/wiki/{full_title}" in sentence['sentence']:
                        # get the start and end index of the link in the sentence
                        link_data['sentence'] = sentence['sentence']
                        link_data['sentence_start_index'] = sentence['start_index']
                        link_data['sentence_end_index'] = sentence['end_index']
                        break
                    elif j == len(sentences) - 1:
                        print('Not found')
                        print(sentences[0]['sentence'])
                link_data['page_length'] = len(raw_html)

                sentences = sentences[j:]

                # add the link to the node
                links.append(link_data)
    node["page_length"] = len(raw_html)
    node["lead_paragraph"] = lead_paragraph

    return node, links


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

    # Read page IDs
    page_ids = pd.read_parquet(args.page_ids)
    # Read redirect map
    redirect_map = pd.read_parquet(args.redirect_map)

    # Read all input pages
    files = glob(f"{args.input_dir}/*.parquet")
    for i, file in tqdm(enumerate(files)):
        df = pd.read_parquet(file)
        df = df.reset_index(drop=True)
        # Iterate through all pages
        extra_columns = []
        links = []
        for i in range(len(df)):
            node, page_links = extract_links(df[i], page_ids, redirect_map)
            extra_columns.append(node)
            links.extend(page_links)
        # Add extra columns to the dataframe
        df = df.join(pd.DataFrame(extra_columns))
        # Save the dataframe
        df.to_parquet(f"{args.output_dir}/{os.path.basename(file)}")
        # Save the links
        df_links = pd.DataFrame(links)
        df_links.to_parquet(f"{args.output_dir}/links_{i}.parquet")
