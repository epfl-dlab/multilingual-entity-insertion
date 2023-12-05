import argparse
import json
import os
import re
import urllib.parse
import warnings
from glob import glob
from multiprocessing import Pool, cpu_count
from nltk import sent_tokenize

import lxml
import pandas as pd
from bs4 import (BeautifulSoup, MarkupResemblesLocatorWarning,
                 NavigableString)
from tqdm import tqdm
import random
import gc

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
# set TOKENIZERS_PARALLELISM to True to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def process_title(title):
    title = title.replace(' ', '_')
    # decode from url encoding
    title = urllib.parse.unquote(title)
    # reencode into url encoding
    title = urllib.parse.quote(title)
    return title


def invalid_parent(parent):
    # hard-coded invalid classes found by manual inspection
    invalid_classes = ["mw-editsection", "metadata", "mw-jump-link", "reflist", "side-box-text", "plainlist", "legend",
                       "navbox-styles", "nomobile", "thumbcaption", "side-box", "side-box-right", "plainlinks", "sistersitebox", "navbox"]
    # some parents have multiple classes
    # if any of the classes is invalid, then the parent is invalid
    if parent.get("class"):
        for class_name in parent.get("class"):
            if class_name in invalid_classes:
                return True
    if parent.name in ['table', 'figure', 'sup', 'caption', 'map'] or parent.get("role") in ["note", "navigation"]:
        return True
    return False


def fix_sentence_tokenizer(sentences):
    if not sentences:
        return sentences
    # dont allow parenthesis to be separated across sentences
    i = 0
    first_stage_sentences = []
    join_prev = False
    while i < len(sentences) - 1:
        # find right-most occurrence of ')' and '(' in sentences[i]
        right_paren_before = sentences[i].rfind(')')
        left_paren_before = sentences[i].rfind('(')
        right_paren_before = right_paren_before if right_paren_before != -1 else 0
        left_paren_before = left_paren_before if left_paren_before != -1 else 0
        # find left-most occurrence of ')' and '(' in sentences[i+1]
        right_paren_after = sentences[i+1].find(')')
        left_paren_after = sentences[i+1].find('(')
        right_parent_after = right_paren_after if right_paren_after != - \
            1 else len(sentences[i+1])
        left_paren_after = left_paren_after if left_paren_after != - \
            1 else len(sentences[i+1])

        if join_prev:
            first_stage_sentences[-1] += ' ' + sentences[i]
            join_prev = False
        else:
            first_stage_sentences.append(sentences[i])

        if right_paren_before < left_paren_before and right_parent_after < left_paren_after:
            join_prev = True
        
        i += 1
    if join_prev:
        first_stage_sentences[-1] += ' ' + sentences[i]
    else:
        first_stage_sentences.append(sentences[i])

    # for each sentence, if it is less than 10 characters, merge it with the next and previous sentences
    i = 0
    final_sentences = []
    join_prev = 0
    while i < len(first_stage_sentences):
        if len(first_stage_sentences[i]) < 10:
            if final_sentences == []:
                final_sentences.append(first_stage_sentences[i])
                join_prev = 2
            else:
                final_sentences[-1] += ' ' + first_stage_sentences[i]
                join_prev = 2
        elif join_prev > 0:
            final_sentences[-1] += ' ' + first_stage_sentences[i]
            join_prev -= 1
        else:
            final_sentences.append(first_stage_sentences[i])
        i += 1
    return final_sentences


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
    del temp_sentences

    depth = [0, 0, 0]
    sections = ["Lead"]
    section_links = []
    section_text = {'Lead': {'text': '',
                             'depth': 0,
                             'title': source_page['title'],
                             'links': []}}
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
                    if section_links:
                        # save all the links from this section
                        current_text = section_text[sections[0]]['text'].strip(
                        )
                        current_text = re.sub(r'\[.*?\]', '', current_text)
                        current_text = re.sub(r' +', ' ', current_text)
                        temp = [
                            elem.strip() + "\n" for elem in current_text.split('\n') if elem.strip() != '']
                        if temp:
                            temp[-1] = temp[-1][:-1]
                        section_sentences = []
                        for span in temp:
                            new_sentences = sent_tokenize(span)
                            new_sentences = fix_sentence_tokenizer(
                                new_sentences)
                            if new_sentences:
                                new_sentences[-1] += '\n'
                                section_sentences.extend(new_sentences)
                        if section_sentences and section_sentences[-1]:
                            section_sentences[-1] = section_sentences[-1][:-1]
                        first_sentence = 0
                        first_index = 0
                        for i, link in enumerate(section_links):
                            if link['sentence'] is None:
                                link['context_span_start_index'] = None
                                link['context_span_end_index'] = None
                                link['context_sentence_start_index'] = None
                                link['context_sentence_end_index'] = None
                                link['context_mention_start_index'] = None
                                link['context_mention_end_index'] = None
                                link['context'] = None
                                # print('Context Failure')
                            else:
                                sentence = link['sentence'].strip()
                                sentence = re.sub(r'\[.*?\]', '', sentence)
                                sentence = re.sub(r' +', ' ', sentence)
                                link['sentence'] = sentence
                                # print(first_sentence, len(section_sentences))
                                for j in range(first_sentence, len(section_sentences)):
                                    if j != first_sentence:
                                        first_index = 0
                                    if re.search(re.escape(link['mention']), section_sentences[j][first_index:]) is not None:
                                        first_sentence = j
                                        first_index = section_sentences[j].index(
                                            link['mention'], first_index) + len(link['mention'])
                                        # use as context the sentence in which the mention is present
                                        # as well as the 5 previous and 5 next sentences
                                        context_start = max(0, j - 5)
                                        context_end = min(
                                            len(section_sentences), j + 6)
                                        link['context'] = re.sub(r' +', ' ', ' '.join(
                                            section_sentences[context_start:context_end]))
                                        link['context'] = re.sub(
                                            r'\n +', '\n', link['context'])
                                        link['context_sentence_start_index'] = link['context'].index(
                                            section_sentences[j])
                                        link['context_sentence_end_index'] = link['context_sentence_start_index'] + len(
                                            section_sentences[j])
                                        link['context_mention_start_index'] = link['context_sentence_start_index'] + \
                                            section_sentences[j].index(
                                                link['mention'])
                                        link['context_mention_end_index'] = link['context_mention_start_index'] + len(
                                            link['mention'])

                                        # the span is defined as a random number of sentences around the mention
                                        # the number of sentences is between 1 and 3 to each side
                                        # the number of sentences is chosen randomly
                                        left_sentence = random.randint(max(context_start, j - 3),
                                                                       max(context_start, j - 1))
                                        right_sentence = random.randint(min(context_end-1, j + 1),
                                                                        min(context_end-1, j + 3))
                                        link['context_span_start_index'] = link['context'].index(
                                            section_sentences[left_sentence].strip())
                                        link['context_span_end_index'] = link['context'].index(section_sentences[right_sentence].strip()) + \
                                            len(section_sentences[right_sentence].strip(
                                            ))

                                        # find all the links contained in the context
                                        current_links = {}
                                        mention_count = {}
                                        last_index = link['context_mention_start_index']
                                        for other_link in reversed(section_links[:i]):
                                            if other_link['sentence'] is None:
                                                continue
                                            if other_link['mention'] not in mention_count:
                                                mention_count[other_link['mention']] = 0
                                            if link['context'].count(other_link['mention']) > mention_count[other_link['mention']]:
                                                mention_count[other_link['mention']] += 1
                                                # find the n-th position of the mention in the context
                                                # where n is mention_count[other_link['mention']] + 1
                                                position = link['context'].rfind(
                                                    other_link['mention'], 0, last_index)
                                                if position == -1:
                                                    break
                                                last_index = position

                                                if position >= link['context_sentence_start_index'] and position < link['context_sentence_end_index']:
                                                    region = 'sentence'
                                                elif position >= link['context_span_start_index'] and position < link['context_span_end_index']:
                                                    region = 'span'
                                                else:
                                                    region = 'global'
                                                if other_link['target_title'] not in current_links:
                                                    current_links[other_link['target_title']] = {'region': region,
                                                                                                 'count': 1}
                                                else:
                                                    current_links[other_link['target_title']
                                                                  ]['count'] += 1
                                                    if current_links[other_link['target_title']]['region'] == 'sentence' and region in ['span', 'global']:
                                                        current_links[other_link['target_title']
                                                                      ]['region'] = region
                                                    elif current_links[other_link['target_title']]['region'] == 'span' and region == 'global':
                                                        current_links[other_link['target_title']
                                                                      ]['region'] = region
                                            else:
                                                break
                                        last_index = link['context_mention_end_index']
                                        for other_link in section_links[i+1:]:
                                            if other_link['sentence'] is None:
                                                continue
                                            if other_link['mention'] not in mention_count:
                                                mention_count[other_link['mention']] = 0
                                            if link['context'].count(other_link['mention']) > mention_count[other_link['mention']]:
                                                mention_count[other_link['mention']] += 1
                                                # find the n-th position of the mention in the context
                                                # where n is mention_count[other_link['mention']] + 1
                                                position = link['context'].find(
                                                    other_link['mention'], last_index)
                                                if position == -1:
                                                    break
                                                last_index = position

                                                if position >= link['context_sentence_start_index'] and position < link['context_sentence_end_index']:
                                                    region = 'sentence'
                                                elif position >= link['context_span_start_index'] and position < link['context_span_end_index']:
                                                    region = 'span'
                                                else:
                                                    region = 'global'
                                                if other_link['target_title'] not in current_links:
                                                    current_links[other_link['target_title']] = {'region': region,
                                                                                                 'count': 1}
                                                else:
                                                    current_links[other_link['target_title']
                                                                  ]['count'] += 1
                                                    if current_links[other_link['target_title']]['region'] == 'sentence' and region in ['span', 'global']:
                                                        current_links[other_link['target_title']
                                                                      ]['region'] = region
                                                    elif current_links[other_link['target_title']]['region'] == 'span' and region == 'global':
                                                        current_links[other_link['target_title']
                                                                      ]['region'] = region
                                            else:
                                                break
                                        # link['current_links'] = json.dumps(
                                        #     current_links)
                                        link['current_links'] = str(current_links)
                                        break
                            section_text[sections[0]]['links'].append({'mention': link['mention'],
                                                                       'target_title': link['target_title']})
                            # if 'context' not in link:
                            #     print('SOURCE TITLE', link['source_version'])
                            #     print('MENTION', link['mention'])
                            #     print('#################')
                            #     print(section_sentences)
                            #     print('#################')
                            #     print(current_text)
                            #     print('#################')
                                
                            # if 'context' not in link:
                            #     for key in link:
                            #         print(key, link[key])
                            #     print(len(section_sentences))
                            #     for j in range(len(section_sentences)):
                            #         print(link['mention'], '<sep>', section_sentences[j])
                            #     print('#################')
                            found_links.append(link)
                    section_links = []
                    sections = [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    section_text[sections[0]] = {}
                    section_text[sections[0]]['text'] = re.sub(
                        r'\[.*?\]', '', tag.text).strip() + '\n'
                    section_text[sections[0]]['depth'] = depth[0] + 1
                    section_text[sections[0]]['title'] = source_page['title']
                    section_text[sections[0]]['links'] = []
                    depth[0] += 1
                    depth[1] = 0
                    depth[2] = 0
                elif tag.name == 'h3':
                    sections = sections[:1] + \
                        [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    depth[1] += 1
                    depth[2] = 0
                    section_text[sections[0]]['text'] += re.sub(
                        r'\[.*?\]', '', tag.text).strip() + '\n'
                elif tag.name == 'h4':
                    sections = sections[:2] + \
                        [re.sub(r'\[.*?\]', '', tag.text).strip()]
                    depth[2] += 1
                    section_text[sections[0]]['text'] += re.sub(
                        r'\[.*?\]', '', tag.text).strip() + '\n'
                if sections in [["Notes"], ["References"], ["Sources"], ["External links"], ["Further reading"], ["Other websites"], ["Sources and references"]]:
                    break

            # find all p and li tags
            skip_text = False
            test_elements = [parent for parent in tag.parents]
            test_elements += [tag]
            for parent in test_elements:
                skip_text = invalid_parent(parent)
                if skip_text:
                    break
            if not skip_text:
                if tag.name == 'p' or tag.name == 'dl':
                    section_text[sections[0]]['text'] += urllib.parse.unquote(
                        tag.text).replace('_', ' ') + ' '
                elif tag.name == 'li' or tag.name == 'span':
                    section_text[sections[0]]['text'] += urllib.parse.unquote(
                        tag.text).replace('_', ' ') + ' \n'
                else:
                    tags = tag.find_all(['p', 'li', 'dl', 'span'])
                    for t in tags:
                        for parent in t.parents:
                            skip = invalid_parent(parent)
                            if skip:
                                break
                        if skip:
                            continue
                        if tag.name == 'p':
                            section_text[sections[0]]['text'] += urllib.parse.unquote(
                                t.text).replace('_', ' ') + ' '
                        elif tag.name == 'dl' and tag.get("role") != "note" and tag.get("class") != ["navbox-styles", "nomobile"]:
                            section_text[sections[0]]['text'] += urllib.parse.unquote(
                                t.text).replace('_', ' ') + ' '
                            break
                        elif tag.name == 'span':
                            section_text[sections[0]]['text'] += urllib.parse.unquote(
                                t.text).replace('_', ' ') + ' \n'
                            break
                        else:
                            section_text[sections[0]]['text'] += urllib.parse.unquote(
                                t.text).replace('_', ' ') + ' \n'

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
                    invalid = invalid_parent(parent)
                    if invalid:
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
                counter = 0
                while link_data['target_title'] in redirect_map and counter < 10:
                    link_data['target_title'] = redirect_map[link_data['target_title']]
                    counter += 1
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
                link_data['source_version'] = source_page['version']

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
                    # print(raw_html)
                    # print('#################')
                    # print(full_title)
                    # print('#################')
                    # print(search_index_link)
                    # print('#################')
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
                            clean_sentence = sentence['sentence']
                            if '</table>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '</table>')[1]
                            if '<table' in clean_sentence:
                                clean_sentence = clean_sentence.split('<table')[
                                    0]
                            if '<body' in clean_sentence:
                                clean_sentence = clean_sentence.split('<body')[
                                    1]
                            if '</body>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '</body>')[0]
                            if '</figure>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '</figure>')[1]
                            if '<figure' in clean_sentence:
                                clean_sentence = clean_sentence.split('<figure')[
                                    0]

                            if '<h3' in clean_sentence and '</h3>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '<h3')[0] + clean_sentence.split('</h3>')[1]
                            elif '<h3' in clean_sentence:
                                clean_sentence = clean_sentence.split('<h3')[0]
                            elif '</h3>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '</h3>')[1]

                            if '<h4' in clean_sentence and '</h4>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '<h4')[0] + clean_sentence.split('</h4>')[1]
                            elif '<h4' in clean_sentence:
                                clean_sentence = clean_sentence.split('<h4')[0]
                            elif '</h4>' in clean_sentence:
                                clean_sentence = clean_sentence.split(
                                    '</h4>')[1]

                            index_right_arrow_first = clean_sentence.find(
                                '>') if clean_sentence.find('>') != -1 else len(clean_sentence)
                            index_left_arrow_first = clean_sentence.find(
                                '<') if clean_sentence.find('<') != -1 else len(clean_sentence)
                            if index_right_arrow_first < index_left_arrow_first:
                                clean_sentence = clean_sentence[index_right_arrow_first+1:]

                            index_right_arrow_last = clean_sentence.rfind(
                                '>') if clean_sentence.rfind('>') != -1 else 0
                            index_left_arrow_last = clean_sentence.rfind(
                                '<') if clean_sentence.rfind('<') != -1 else 0
                            if index_right_arrow_last < index_left_arrow_last:
                                clean_sentence = clean_sentence[:index_left_arrow_last]

                            parsed_sentence = BeautifulSoup(
                                clean_sentence, 'html.parser')

                            link_data['sentence'] = parsed_sentence.text
                            link_data['sentence_raw'] = clean_sentence
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

                section_links.append(link_data)

    if section_links:
        # save the last remaining links
        current_text = section_text[sections[0]]['text'].strip()
        current_text = re.sub(r'\[.*?\]', '', current_text)
        # current_text = re.sub(r'\n', ' ', current_text)
        current_text = re.sub(r' +', ' ', current_text)

        temp = [line.strip() + '\n' for line in current_text.split('\n')
                if line != '']
        if temp:
            temp[-1] = temp[-1][:-1]
        section_sentences = []
        for sentence in temp:
            new_sentences = sent_tokenize(sentence)
            new_sentences = fix_sentence_tokenizer(new_sentences)
            if new_sentences:
                new_sentences[-1] += '\n'
                section_sentences.extend(new_sentences)
        if section_sentences and section_sentences[-1]:
            section_sentences[-1] = section_sentences[-1][:-1]
        first_sentence = 0
        first_index = 0
        for i, link in enumerate(section_links):
            if link['sentence'] is None:
                link['context_span_start_index'] = None
                link['context_span_end_index'] = None
                link['context_sentence_start_index'] = None
                link['context_sentence_end_index'] = None
                link['context_mention_start_index'] = None
                link['context_mention_end_index'] = None
                link['context'] = None
                # print('Context Failure')
            else:
                sentence = link['sentence'].strip()
                sentence = re.sub(r'\[.*?\]', '', sentence)
                # sentence = re.sub(r'\n', ' ', sentence)
                sentence = re.sub(r' +', ' ', sentence)
                link['sentence'] = sentence
                # print(first_sentence, len(section_sentences))
                for j in range(first_sentence, len(section_sentences)):
                    if j != first_sentence:
                        first_index = 0
                    if re.search(re.escape(link['mention']), section_sentences[j][first_index:]) is not None:
                        first_sentence = j
                        first_index = section_sentences[j].index(
                            link['mention'], first_index) + len(link['mention'])
                        # use as context the sentence in which the mention is present
                        # as well as the 5 previous and 5 next sentences
                        context_start = max(0, j - 5)
                        context_end = min(len(section_sentences), j + 6)
                        link['context'] = ' '.join(
                            section_sentences[context_start:context_end])
                        link['context_sentence_start_index'] = link['context'].index(
                            section_sentences[j])
                        link['context_sentence_end_index'] = link['context_sentence_start_index'] + \
                            len(section_sentences[j])
                        link['context_mention_start_index'] = link['context_sentence_start_index'] + \
                            section_sentences[j].index(link['mention'])
                        link['context_mention_end_index'] = link['context_mention_start_index'] + \
                            len(link['mention'])
                        # the span is defined as a random number of sentences around the mention
                        # the number of sentences is between 1 and 3 to each side
                        # the number of sentences is chosen randomly
                        left_sentence = random.randint(max(context_start, j - 3),
                                                       max(context_start, j - 1))
                        right_sentence = random.randint(min(context_end - 1, j + 1),
                                                        min(context_end - 1, j + 3))
                        link['context_span_start_index'] = link['context'].index(
                            section_sentences[left_sentence].strip())
                        link['context_span_end_index'] = link['context'].index(section_sentences[right_sentence].strip()) + \
                            len(section_sentences[right_sentence].strip())

                        # find all the links contained in the context
                        current_links = {}
                        mention_count = {}
                        last_index = link['context_mention_start_index']
                        for other_link in reversed(section_links[:i]):
                            if other_link['sentence'] is None:
                                continue
                            if other_link['mention'] not in mention_count:
                                mention_count[other_link['mention']] = 0
                            if link['context'].count(other_link['mention']) > mention_count[other_link['mention']]:
                                mention_count[other_link['mention']] += 1
                                # find the n-th position of the mention in the context
                                # where n is mention_count[other_link['mention']] + 1
                                position = link['context'].rfind(
                                    other_link['mention'], 0, last_index)
                                if position == -1:
                                    break
                                last_index = position

                                if position >= link['context_sentence_start_index'] and position < link['context_sentence_end_index']:
                                    region = 'sentence'
                                elif position >= link['context_span_start_index'] and position < link['context_span_end_index']:
                                    region = 'span'
                                else:
                                    region = 'global'
                                if other_link['target_title'] not in current_links:
                                    current_links[other_link['target_title']] = {'region': region,
                                                                                 'count': 1}
                                else:
                                    current_links[other_link['target_title']
                                                  ]['count'] += 1
                                    if current_links[other_link['target_title']]['region'] == 'sentence' and region in ['span', 'global']:
                                        current_links[other_link['target_title']
                                                      ]['region'] = region
                                    elif current_links[other_link['target_title']]['region'] == 'span' and region == 'global':
                                        current_links[other_link['target_title']
                                                      ]['region'] = region
                            else:
                                break
                        last_index = link['context_mention_end_index']
                        for other_link in section_links[i+1:]:
                            if other_link['sentence'] is None:
                                continue
                            if other_link['mention'] not in mention_count:
                                mention_count[other_link['mention']] = 0
                            if link['context'].count(other_link['mention']) > mention_count[other_link['mention']]:
                                mention_count[other_link['mention']] += 1
                                # find the n-th position of the mention in the context
                                # where n is mention_count[other_link['mention']] + 1
                                position = link['context'].find(
                                    other_link['mention'], last_index)
                                if position == -1:
                                    break
                                last_index = position

                                if position >= link['context_sentence_start_index'] and position < link['context_sentence_end_index']:
                                    region = 'sentence'
                                elif position >= link['context_span_start_index'] and position < link['context_span_end_index']:
                                    region = 'span'
                                else:
                                    region = 'global'
                                if other_link['target_title'] not in current_links:
                                    current_links[other_link['target_title']] = {'region': region,
                                                                                 'count': 1}
                                else:
                                    current_links[other_link['target_title']
                                                  ]['count'] += 1
                                    if current_links[other_link['target_title']]['region'] == 'sentence' and region in ['span', 'global']:
                                        current_links[other_link['target_title']
                                                      ]['region'] = region
                                    elif current_links[other_link['target_title']]['region'] == 'span' and region == 'global':
                                        current_links[other_link['target_title']
                                                      ]['region'] = region
                            else:
                                break
                        # link['current_links'] = json.dumps(
                        #     current_links)
                        link['current_links'] = str(current_links)
                        break
            
            # if 'context' not in link:
            #     print('SOURCE TITLE', link['source_version'])
            #     print('MENTION', link['mention'])
            #     print('#################')
            #     print(section_sentences)
            #     print('#################')
            #     print(current_text)
            #     print('#################')
                
            found_links.append(link)

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

    mention_map = set([])
    counter = 0
    links = []
    sections = []
    pool = Pool(min(cpu_count(), args.processes), initializer=initializer)
    for i, file in (pbar := tqdm(enumerate(files), total=len(files))):
        df = pd.read_parquet(file)
        list_data = df.to_dict(orient='records')
        del df

        for j, (page_links, section_text) in enumerate(pool.imap(extract_links, list_data)):
            if j % 1000 == 0:
                pbar.set_description(
                    f"Processing file {file} at element {j}/{len(list_data)}")
            for link in page_links:
                links.append(link)
                if link['mention'].strip() != '':
                    mention_map.add(
                        f"{link['mention']}<sep>{link['target_title']}")
            for section in section_text:
                sections.append(
                    {'section': section, 
                     'text': section_text[section]['text'], 
                     'depth': section_text[section]['depth'], 
                     'title': section_text[section]['title'],
                     'links': section_text[section]['links']})
                
            if len(links) > 250_000:
                df_links = pd.DataFrame(links)
                df_links.to_parquet(f"{args.output_dir}/links_{counter}.parquet")
                del df_links
                del links

                df_sections = pd.DataFrame(sections)
                df_sections.to_parquet(f"{args.output_dir}/sections_{counter}.parquet")
                del df_sections
                del sections
                counter += 1

                gc.collect()
                links = []
                sections = []
        del list_data
        gc.collect()
    pool.close()
    pool.join()
    
    if links:
        df_links = pd.DataFrame(links)
        df_links.to_parquet(f"{args.output_dir}/links_{counter}.parquet")
        del df_links
        del links
    if sections:
        df_sections = pd.DataFrame(sections)
        df_sections.to_parquet(f"{args.output_dir}/sections_{counter}.parquet")
        del df_sections
        del sections

    mention_map = [{'mention': mention.split('<sep>')[0], 'target_title': mention.split(
        '<sep>')[1]} for mention in mention_map]
    df_mentions = pd.DataFrame(mention_map)
    df_mentions.to_parquet(f"{args.output_dir}/mention_map.parquet")
