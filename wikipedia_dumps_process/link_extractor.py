import argparse
import json
import os
import re
import urllib.parse
from glob import glob

import lxml
import pandas as pd
from bs4 import BeautifulSoup, Comment, NavigableString


def process_title(title):
    return urllib.parse.quote(title.replace(' ', '_'))


def extract_links(raw_html, page_ids):
    parsed_html = BeautifulSoup(raw_html, 'lxml')

    node = {}
    # extract title from HTML
    node['title'] = process_title(parsed_html.find("title").text[:-12])
    # initialize links list
    node['links'] = []
    # all relevant HTML is in div with class mw-parser-output in div with id mw-content-text
    content = parsed_html.find("div", {"id": "mw-content-text"})
    content = content.find("div", {"class": "mw-parser-output"})

    # remove all table tags
    for table in content.find_all("table"):
        table.extract()
    # remove all figure tags
    for figure in content.find_all("figure"):
        figure.extract()
    # remove all sup tags
    for sup in content.find_all("sup"):
        sup.extract()
    # remove all tags with class mw-editsection
    for edit in content.find_all("span", {"class": "mw-editsection"}):
        edit.extract()
    # bring out the content from all meta tags and remove the tag
    for meta in content.find_all("meta"):
        meta.unwrap()

    text_content = str(content)

    section = ["Lead"]
    lead_paragraph = ''
    search_index_link = 0
    search_index_sentence = 0
    # iterate through all tags
    for tag in content:
        if isinstance(tag, NavigableString):
            continue
        if tag.get("role") == "note":
            continue
        # if the tag is a header tag, update section
        if tag.name in ['h2', 'h3', 'h4']:
            if tag.name == 'h2':
                section = [process_title(tag.text)]
            elif tag.name == 'h3':
                section = section[:1] + [process_title(tag.text)]
            elif tag.name == 'h4':
                section = section[:2] + [process_title(tag.text)]
            if section in [["Notes"], ["References"], ["External links"], ["Further reading"]]:
                break

        if tag.name == 'p' and section == ["Lead"]:
            lead_paragraph += tag.text

        text_tag = str(tag)

        # split text into sentences
        # use all punctuation marks and new line as sentence separators
        sentences = re.split(r'[.!?;:\n]', text_tag)

        # find links
        links = tag.find_all('a')
        # if there are links in the tag
        if links:
            # iterate through the links
            for link in links:
                # if link has class "external text" then skip it
                if link.get("class") == ["external", "text"]:
                    continue
                link_data = {}

                # get the title of the link and the target anchor if it exists
                full_title = link["href"][6:]
                if "#" in full_title:
                    link_data['title'] = full_title.split("#")[0]
                    link_data['target_section'] = full_title.split("#")[1]
                else:
                    link_data['title'] = full_title
                    link_data['target_section'] = "Lead"

                # get the text of the link
                link_data['text'] = link.text

                # get the source section of the link
                link_data['source_section'] = '<sep>'.join(section)

                # get the start and end index of the link in the text
                index = text_content.index(str(link), search_index_link)
                search_index_link = index + len(str(link))
                link_data['link_start_index'] = index
                link_data['link_end_index'] = index + len(str(link))

                # get the sentence in which the link is present
                # iterate through the sentences
                for i, sentence in enumerate(sentences):
                    # if the link is present in the sentence
                    if str(link) in sentence:
                        # get the start and end index of the link in the sentence
                        link_data['sentence'] = sentence
                        index = text_content.index(
                            sentence, search_index_sentence)
                        link_data['sentence_start_index'] = index
                        link_data['sentence_end_index'] = index + len(sentence)
                        search_index_sentence = index
                        break
                sentences = sentences[i:]

                # add the link to the node
                node["links"].append(link_data)
                node["page_length"] = len(text_content)
                node["lead_paragraph"] = lead_paragraph

    return node


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        required=True, help='Path to the data folder')
    parser.add_argument('--page_ids', type=str, required=True,
                        help='Path to the json file with IDs and names of all pages')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Path to the output folder')
    args = parser.parse_args()

    # check if input dir exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    # check if page_ids file exists
    if not os.path.exists(args.page_ids):
        raise ValueError(f"Page IDs file {args.page_ids} does not exist")
    # check if output dir exists
    # if it doesn't exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read page IDs
    with open(args.page_ids, 'r') as f:
        page_ids = json.load(f)

    # Read all json files
    files = glob(f"{args.input_dir}/*.ndjson")
    simple_pages = {}
    counter = 0
    for file in files:
        print(f"Processing {file}")
        df = pd.read_json(file, chunksize=args.chunksize, lines=True)
        for chunk in df:
            pages = extract_dump(chunk)
            for page in pages:
                simple_pages[page] = {'ID': pages[page]
                                      ['ID'], 'QID': pages[page]['QID']}
            with open(f"{args.output_dir}/pages_{counter}.json", 'w') as f:
                json.dump(pages, f)
            counter += 1
    with open(f"{args.output_dir}/simple_pages.json", 'w') as f:
        json.dump(simple_pages, f)
