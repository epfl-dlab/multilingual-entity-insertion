from calendar import c
from numpy import argsort
import pandas as pd
import argparse
from bs4 import BeautifulSoup, Comment, MarkupResemblesLocatorWarning
import re
import urllib
import difflib
from nltk import sent_tokenize
import math
from tqdm import tqdm
import os
import warnings
from multiprocessing import Pool, cpu_count
tqdm.pandas()

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def fix_target_titles(title, redirect_map_1, redirect_map_2):
    counter = 0
    while title in redirect_map_1:
        title = redirect_map_1[title]
        counter += 1
        if counter == 10:
            break

    counter = 0
    while title in redirect_map_2:
        title = redirect_map_2[title]
        counter += 1
        if counter == 10:
            break
    return title


def simplify_html(html):
    # remove all figures, tables, captions, sup, style
    # figures
    for figure in html.find_all('figure'):
        figure.decompose()
    # tables
    for table in html.find_all('table'):
        table.decompose()
    # captions
    for caption in html.find_all('caption'):
        caption.decompose()
    # sup
    for sup in html.find_all('sup'):
        sup.decompose()
    # style
    for style in html.find_all('style'):
        style.decompose()

    # remove all comments
    comments = html.find_all(string=lambda text: isinstance(text, Comment))
    [comment.extract() for comment in comments]

    # remove all tags with class 'mw-editsection'
    for tag in html.find_all('span', {'class': 'mw-editsection'}):
        tag.decompose()

    # remove all tags with class 'metadata'
    for tag in html.find_all('div', {'class': 'metadata'}):
        tag.decompose()

    # remove all tags with class 'reflist'
    for tag in html.find_all('div', {'class': 'reflist'}):
        tag.decompose()

    # remove all links with class 'external text'
    for tag in html.find_all('a', {'class': 'external text'}):
        tag.decompose()

    # remove all map tags
    for tag in html.find_all('map'):
        tag.decompose()

    # remove all tags with any of the following classes
    # side-box side-box-right plainlinks sistersitebox
    for tag in html.find_all('div', {'class': ['side-box', 'side-box-right', 'plainlinks', 'sistersitebox']}):
        tag.decompose()

    # remove all tags with class 'thumbcaption'
    for tag in html.find_all('div', {'class': 'thumbcaption'}):
        tag.decompose()

    return html


def fix_sentence_tokenizer(sentences):
    # for each sentence, if it is less than 10 characters, merge it with the next and previous sentences
    i = 0
    if len(sentences) > 1:
        while i < len(sentences):
            if len(sentences[i]) < 10:
                if i > 0 and i < len(sentences) - 1:
                    sentences[i-1] = sentences[i-1] + ' ' + \
                        sentences[i] + ' ' + sentences[i+1]
                    del sentences[i]
                    del sentences[i]
                elif i == 0:
                    sentences[i] = sentences[i] + ' ' + sentences[i+1]
                    del sentences[i+1]
                else:
                    sentences[i-1] = sentences[i-1] + ' ' + sentences[i]
                    del sentences[i]
            else:
                i += 1

    # dont allow for links to be separated across sentences
    i = 0
    while i < len(sentences):
        while sentences[i].count('<a ') > sentences[i].count('</a>'):
            sentences[i] = sentences[i] + ' ' + sentences[i+1]
            del sentences[i+1]
        i += 1

    # dont allow parenthesis to be separated across sentences
    i = 0
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

        if right_paren_before < left_paren_before and right_parent_after < left_paren_after:
            sentences[i] = sentences[i] + ' ' + sentences[i+1]
            del sentences[i+1]
        else:
            i += 1

    # dont allow list items to be separated across sentences
    i = 0
    while i < len(sentences) - 1:
        while sentences[i].count('<li>') > sentences[i].count('</li>'):
            sentences[i] = sentences[i] + ' ' + sentences[i+1]
            del sentences[i+1]
        i += 1

    # force </li> to act as a separator
    i = 0
    while i < len(sentences):
        if '</li>' in sentences[i]:
            extra_sentences = [
                s.strip() for s in sentences[i].split('</li>')]
            extra_sentences[:-1] = [s for s in extra_sentences[:-1] if s != '']
            extra_sentences[:-1] = [s + '</li>' for s in extra_sentences[:-1]]
            if extra_sentences[-1] == '':
                extra_sentences.pop()
            del sentences[i]
            sentences = sentences[:i] + extra_sentences + sentences[i:]
        i += 1

    # add a new line to all lists
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace('\n</li>', '</li>')
        sentences[i] = sentences[i].replace('</li>', '\n</li>')

    # dont allow h2 tags to be separated across sentences
    i = 0
    while i < len(sentences):
        while sentences[i].count('<h2>') != sentences[i].count('</h2>'):
            sentences[i] = sentences[i] + ' ' + sentences[i+1]
            del sentences[i+1]
        i += 1

    # force <h2> to act as a separator
    i = 0
    while i < len(sentences):
        if '<h2>' in sentences[i]:
            extra_sentences = [s.strip() for s in sentences[i].split('<h2>')]
            extra_sentences[1:] = ['<h2>' + s for s in extra_sentences[1:]]
            extra_sentences = [s for s in extra_sentences if s != '']
            del sentences[i]
            sentences = sentences[:i] + extra_sentences + sentences[i:]
        i += 1

    return sentences


def fix_text(text):
    text = text.strip()
    text = re.sub(' +', ' ', text)
    text = re.sub('\n+', '\n', text)
    text = re.sub('\n +', '\n', text)
    return text


def find_negative_contexts(section_sentences, mentions, curr_section, index, correct_context):
    contexts = []
    for section in section_sentences:
        curr_sentences = []
        for i, sentence in enumerate(section_sentences[section]):
            found = False
            for mention in mentions:
                if mention.lower() in sentence['clean_sentence'].lower():
                    found = True
                    break
            if section == curr_section and sentence['index'] == index:
                found = True

            if not found:
                curr_sentences.append(sentence)
            else:
                new_contexts = []
                for i in range(len(curr_sentences)):
                    min_index = max(0, i - 5)
                    max_index = min(len(curr_sentences), i + 6)
                    context = " ".join([s['clean_sentence']
                                        for s in curr_sentences[min_index:max_index]]).strip()
                    if len(context.split(' ')) > 10 and context != correct_context:
                        new_contexts.append(context)
                new_contexts = list(set(new_contexts))
                for context in new_contexts:
                    contexts.append({'context': context, 'section': section})
                curr_sentences = []

        if len(curr_sentences) != 0:
            new_contexts = []
            for i in range(len(curr_sentences)):
                min_index = max(0, i - 5)
                max_index = min(len(curr_sentences), i + 6)
                context = " ".join([s['clean_sentence']
                                    for s in curr_sentences[min_index:max_index]]).strip()
                if len(context.split(' ')) > 10 and context != correct_context:
                    new_contexts.append(context)
            new_contexts = list(set(new_contexts))
            for context in new_contexts:
                contexts.append({'context': context, 'section': section})

    return contexts


def process_version(input):
    output = []
    d = difflib.Differ()
    file_1 = os.path.join(
        args.html_dir, f"{input['source_ID']}_{input['first_version']}.html")
    with open(file_1, 'r') as f:
        text_1 = f.read()
    html_1 = BeautifulSoup(text_1, 'html.parser')
    html_1 = html_1.find_all('div', {'class': 'mw-parser-output'})
    if len(html_1) == 0:
        print('No HTML found')
        print(f"{input['source_ID']}_{input['first_version']}.html")
        return []
    html_1 = html_1[-1]
    html_1_clean = simplify_html(html_1)
    text_1 = "\n".join(
        [line for line in str(html_1_clean).split('\n') if line.strip() != ''])
    sentences_1 = sent_tokenize(text_1)
    sentences_1 = fix_sentence_tokenizer(sentences_1)
    for second_version in input['versions']:
        found_links = []
        file_2 = os.path.join(
            args.html_dir, f"{input['source_ID']}_{second_version}.html")
        with open(file_2, 'r') as f:
            text_2 = f.read()
        html_2 = BeautifulSoup(text_2, 'html.parser')
        html_2 = html_2.find_all('div', {'class': 'mw-parser-output'})
        if len(html_2) == 0:
            print('No HTML found')
            print(f"{input['source_ID']}_{second_version}.html")
            continue
        html_2 = html_2[-1]
        html_2_clean = simplify_html(html_2)
        text_2 = "\n".join(
            [line for line in str(html_2_clean).split('\n') if line.strip() != ''])
        sentences_2 = sent_tokenize(text_2)
        sentences_2 = fix_sentence_tokenizer(sentences_2)

        original_lead = ''
        all_sentences = []
        section_original = 'Lead'
        section_new = 'Lead'
        diff = d.compare(sentences_1, sentences_2)
        for line in diff:
            if line.startswith('?'):
                continue
            elif line.startswith('+'):
                soup = BeautifulSoup(line[2:].strip(), 'html.parser')
                clean_text = soup.text.strip(' ')
                if clean_text.strip() == '':
                    continue
                h2 = soup.find('h2')
                if h2 is not None:
                    section_new = h2.text.strip()
                all_sentences.append({'status': 'added',
                                      'index': len(all_sentences),
                                      'match': None,
                                      'section_new': section_new,
                                      'section_original': section_original,
                                      'clean_sentence': clean_text,
                                      'raw_sentence': line[2:].strip()})
                words = [word.strip() for word in all_sentences[-1]
                         ['clean_sentence'].replace('\n', ' ').split(' ') if word != '']
                freqs = {}
                for word in words:
                    freqs[word] = freqs.get(word, 0) + 1
                norm = math.sqrt(sum([freqs[word]**2 for word in freqs]))
                best_match = {'index': None, 'score': 0}
                for i, sentence in enumerate(all_sentences):
                    if sentence['status'] != 'removed' or sentence['match'] is not None:
                        continue
                    words = [word.strip() for word in sentence['clean_sentence'].replace(
                        '\n', ' ').split(' ') if word != '']
                    freqs_2 = {}
                    for word in words:
                        freqs_2[word] = freqs_2.get(word, 0) + 1
                    norm_2 = math.sqrt(
                        sum([freqs_2[word]**2 for word in freqs_2]))
                    score = 0
                    for word in freqs:
                        score += freqs[word] * freqs_2.get(word, 0)
                    score /= (norm * norm_2)
                    if score > best_match['score']:
                        best_match['index'] = i
                        best_match['score'] = score
                if best_match['score'] > 0.5:
                    all_sentences[best_match['index']
                                  ]['match'] = all_sentences[-1]['index']
                    all_sentences[-1]['match'] = all_sentences[best_match['index']]['index']
            elif line.startswith('-'):
                soup = BeautifulSoup(line[2:].strip(), 'html.parser')
                clean_text = soup.text.strip(' ')
                if clean_text.strip() == '':
                    continue
                h2 = soup.find('h2')
                if h2 is not None:
                    section_original = h2.text.strip()
                all_sentences.append({'status': 'removed',
                                      'index': len(all_sentences),
                                      'match': None,
                                      'section_new': section_new,
                                      'section_original': section_original,
                                      'clean_sentence': clean_text,
                                      'raw_sentence': line[2:].strip()})

                words = [word.strip() for word in all_sentences[-1]
                         ['clean_sentence'].replace('\n', ' ').split(' ') if word != '']
                freqs = {}
                for word in words:
                    freqs[word] = freqs.get(word, 0) + 1
                norm = math.sqrt(sum([freqs[word]**2 for word in freqs]))
                best_match = {'index': None, 'score': 0}
                for i, sentence in enumerate(all_sentences):
                    if sentence['status'] != 'added' or sentence['match'] is not None:
                        continue
                    words = [word.strip() for word in sentence['clean_sentence'].replace(
                        '\n', ' ').split(' ') if word != '']
                    freqs_2 = {}
                    for word in words:
                        freqs_2[word] = freqs_2.get(word, 0) + 1
                    norm_2 = math.sqrt(
                        sum([freqs_2[word]**2 for word in freqs_2]))
                    score = 0
                    for word in freqs:
                        score += freqs[word] * freqs_2.get(word, 0)
                    score /= (norm * norm_2)
                    if score > best_match['score']:
                        best_match['index'] = i
                        best_match['score'] = score
                if best_match['score'] > 0.5:
                    all_sentences[best_match['index']
                                  ]['match'] = all_sentences[-1]['index']
                    all_sentences[-1]['match'] = all_sentences[best_match['index']]['index']

                if section_original == 'Lead':
                    original_lead += ' ' + all_sentences[-1]['clean_sentence']
            else:
                soup = BeautifulSoup(line.strip(), 'html.parser')
                clean_text = soup.text.strip(' ')
                if clean_text.strip() == '':
                    continue
                h2 = soup.find('h2')
                if h2 is not None:
                    section_original = h2.text.strip()
                    section_new = section_original
                if section_original == 'Lead':
                    original_lead += ' ' + clean_text
                all_sentences.append({'status': 'neutral',
                                      'index': len(all_sentences),
                                      'match': None,
                                      'section_new': section_new,
                                      'section_original': section_original,
                                      'clean_sentence': clean_text,
                                      'raw_sentence': line.strip()})

        section_sentences = {}
        for sentence in all_sentences:
            if sentence['status'] == 'added':
                continue
            if sentence['section_original'] not in section_sentences:
                section_sentences[sentence['section_original']] = []
            section_sentences[sentence['section_original']].append(sentence)

        for sentence in all_sentences:
            if sentence['status'] != 'added':
                continue
            if sentence['match'] is None:
                direct_match = False
                section = sentence['section_original']
                new_section = sentence['section_new']
                # go through the previous 10 and next 10 sentences
                # if any of these is neutral and has the same section, use it as context
                # if any of these is added and has a match and has the same section, use it as context
                left_index = sentence['index'] - 1
                left_context = []
                counter = 0
                continuous_left_added_text = []
                accept = True
                while left_index >= 0 and counter < 10 and len(left_context) < 5:
                    counter += 1
                    if all_sentences[left_index]['section_original'] != sentence['section_original']:
                        break
                    if all_sentences[left_index]['status'] == 'neutral':
                        left_context.append(
                            all_sentences[left_index]['clean_sentence'])
                        accept = False
                    elif all_sentences[left_index]['status'] == 'added' and accept and all_sentences[left_index]['match'] is None:
                        continuous_left_added_text.append(
                            all_sentences[left_index]['clean_sentence'])
                    elif all_sentences[left_index]['status'] == 'removed':
                        accept = False
                        if all_sentences[left_index]['match'] is not None and all_sentences[left_index]['match'] < sentence['index'] + 11:
                            left_context.append(
                                all_sentences[left_index]['clean_sentence'])
                    left_index -= 1
                right_index = sentence['index'] + 11
                right_context = []
                # rightmost_index = sentence['index'] + 1
                continuous_right_added_text = []
                while right_index > sentence['index']:
                    if right_index >= len(all_sentences) or all_sentences[right_index]['section_original'] != sentence['section_original']:
                        right_index -= 1
                        continue
                    if all_sentences[right_index]['status'] == 'neutral':
                        right_context.append(
                            all_sentences[right_index]['clean_sentence'])
                        continuous_right_added_text = []
                        # rightmost_index = max(rightmost_index, right_index)
                    elif all_sentences[right_index]['status'] == 'added' and all_sentences[right_index]['match'] is None:
                        continuous_right_added_text.append(
                            all_sentences[right_index]['clean_sentence'])
                    elif all_sentences[right_index]['status'] == 'removed':
                        continuous_right_added_text = []
                        if all_sentences[right_index]['index'] is not None and all_sentences[right_index]['index'] > sentence['index'] - 11:
                            right_context.append(
                                all_sentences[right_index]['clean_sentence'])
                            # rightmost_index = max(rightmost_index, right_index)
                    right_index -= 1
                right_context = right_context[-5:]
                # right_index = rightmost_index
                context = ' '.join(
                    left_context[::-1]) + ' '.join(right_context[::-1])
                # added_text = ' '.join(
                #     continuous_left_added_text[::-1]) + sentence['clean_sentence'] + ' '.join(continuous_right_added_text[::-1])
            else:
                direct_match = True
                left_context = []
                left_index = sentence['match'] - 1
                while left_index >= 0 and len(left_context) < 5:
                    if all_sentences[left_index]['status'] == 'added':
                        left_index -= 1
                        continue
                    if all_sentences[left_index]['section_original'] != all_sentences[sentence['match']]['section_original']:
                        break
                    left_context.append(
                        all_sentences[left_index]['clean_sentence'])
                    left_index -= 1
                right_context = []
                right_index = sentence['match'] + 1
                while right_index < len(all_sentences) and len(right_context) < 5:
                    if all_sentences[right_index]['status'] == 'added':
                        right_index += 1
                        continue
                    if all_sentences[right_index]['section_original'] != all_sentences[sentence['match']]['section_original']:
                        break
                    right_context.append(
                        all_sentences[right_index]['clean_sentence'])
                    right_index += 1
                context = ' '.join(left_context[::-1]) + ' ' + all_sentences[sentence['match']
                                                                             ]['clean_sentence'] + ' ' + ' '.join(right_context)
                section = all_sentences[sentence['match']]['section_original']
                new_section = sentence['section_new']

            if '<a ' not in sentence['raw_sentence']:
                continue
            soup = BeautifulSoup(sentence['raw_sentence'], 'html.parser')
            links = soup.find_all('a')
            for link in links:
                # get href
                href = link.get('href')
                if href is None or not href.startswith('/wiki/'):
                    continue
                link_text = link.text.strip()
                if sentence['match'] is None:
                    present = False
                    if new_section not in section_sentences:
                        missing_category = 'missing_section'
                    else:
                        if continuous_left_added_text == [] and continuous_right_added_text == []:
                            if sentence['clean_sentence'].lower().strip() == link_text.lower():
                                missing_category = 'missing_mention'
                            else:
                                missing_category = 'missing_sentence'
                        else:
                            missing_category = 'missing_span'

                else:
                    present = link_text.lower(
                    ) in all_sentences[sentence['match']]['clean_sentence'].lower()
                    missing_category = None if present else 'missing_mention'

                # get target title
                target_title = fix_target_titles(
                    href[6:], redirect_1, redirect_2)
                mentions = mentions_map.get(
                    target_title, [urllib.parse.unquote(target_title).replace('_', ' ')])
                negative_contexts = find_negative_contexts(
                    section_sentences, mentions, section, sentence['index'], context)
                if target_title in input['versions'][second_version]:
                    output.append({
                        'source_title': input['source_title'],
                        'source_ID': input['source_ID'],
                        'target_title': target_title,
                        'context': context,
                        'section': section,
                        'mention_present': present,
                        'source_lead': original_lead.strip(),
                        'first_version': input['first_version'],
                        'second_version': second_version,
                        'direct_match': direct_match,
                        'missing_category': missing_category,
                        'negative_contexts': str(negative_contexts)
                    })
                    found_links.append(output[-1])
        # if len(found_links) != len(input['versions'][second_version]):
        #     print('Details')
        #     print(input['source_title'], input['source_ID'], input['first_version'], second_version)
        #     print('Expected Links')
        #     print(input['versions'][second_version])
        #     print('Found Links')
        #     print(found_links)
        #     print(html_1_clean)
        #     print('SEP')
        #     print(html_2_clean)
        #     print('#############################################')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--versions_file', type=str,
                        required=True, help='Path to the versions file')
    parser.add_argument('--html_dir', type=str, required=True,
                        help='Path to the directory containing all the HTML files')
    parser.add_argument('--redirect_1', type=str, required=True,
                        help='Path to the first redirect file')
    parser.add_argument('--redirect_2', type=str, required=True,
                        help='Path to the second redirect file')
    parser.add_argument('--mention_map', type=str, required=True,
                        help='Path to the mention map file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of processes to use')

    args = parser.parse_args()

    # check if input files exist
    if not os.path.exists(args.versions_file):
        raise Exception('Versions file does not exist')
    if not os.path.exists(args.html_dir):
        raise Exception('HTML directory does not exist')
    if not os.path.exists(args.redirect_1):
        raise Exception('Redirect file 1 does not exist')
    if not os.path.exists(args.redirect_2):
        raise Exception('Redirect file 2 does not exist')
    if not os.path.exists(args.mention_map):
        raise Exception('Mention map file does not exist')

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_versions = pd.read_parquet(args.versions_file)
    redirect_1 = pd.read_parquet(args.redirect_1)
    redirect_2 = pd.read_parquet(args.redirect_2)

    redirect_1 = redirect_1.to_dict()['redirect']
    redirect_2 = redirect_2.to_dict()['redirect']

    print('Fixing target titles')
    df_versions['target_title'] = df_versions['target_title'].progress_apply(
        lambda x: fix_target_titles(x, redirect_1, redirect_2))

    print('Creating revisions dictionary')
    df_versions = df_versions.to_dict('records')
    expected_links = len(df_versions)
    print(f'Expected links: {expected_links}')
    revisions = {}
    for row in tqdm(df_versions):
        if row['first_version'] not in revisions:
            revisions[row['first_version']] = {
                'source_title': row['source_title'],
                'source_ID': row['source_ID'],
                'versions': {
                    row['second_version']: [row['target_title']]
                }
            }
        else:
            if row['second_version'] not in revisions[row['first_version']]['versions']:
                revisions[row['first_version']]['versions'][row['second_version']] = [
                    row['target_title']]
            else:
                revisions[row['first_version']]['versions'][row['second_version']].append(
                    row['target_title'])
    revisions_clean = []
    for revision in revisions:
        revisions_clean.append({
            'first_version': revision,
            'source_title': revisions[revision]['source_title'],
            'source_ID': revisions[revision]['source_ID'],
            'versions': revisions[revision]['versions']
        })

    def initializer():
        global redirect_1
        global redirect_2
        global mentions_map

        redirect_1 = pd.read_parquet(args.redirect_1)
        redirect_2 = pd.read_parquet(args.redirect_2)
        redirect_1 = redirect_1.to_dict()['redirect']
        redirect_2 = redirect_2.to_dict()['redirect']

        mentions_map_pre = pd.read_parquet(args.mention_map)
        mentions_map_pre = mentions_map_pre.to_dict(orient='records')
        mentions_map = {}
        for row in mentions_map_pre:
            target_title = fix_target_titles(
                row['target_title'], redirect_1, redirect_2)
            if target_title not in mentions_map:
                mentions_map[target_title] = [row['mention']]
            else:
                mentions_map[target_title].append(row['mention'])

    output = []
    pool = Pool(min(cpu_count(), args.n_processes), initializer=initializer)
    for result in tqdm(pool.imap(process_version, revisions_clean), total=len(revisions_clean)):
        output.extend(result)
    pool.close()
    pool.join()

    df_output = pd.DataFrame(output)
    df_output['context'] = df_output['context'].progress_apply(fix_text)
    df_output['source_lead'] = df_output['source_lead'].progress_apply(
        fix_text)
    df_output.to_parquet(os.path.join(args.output_dir, 'test_data.parquet'))
