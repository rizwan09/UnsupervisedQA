# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Functionality to do constituency parsing, used for shortening cloze questions. We use AllenNLP and the
Parsing model from Stern et. al, 2018 "A Minimal Span-Based Neural Constituency Parser" arXiv:1705.03919
"""
import attr
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from tqdm import tqdm
from nltk import Tree
from .configs import CONSTITUENCY_MODEL, CONSTITUENCY_BATCH_SIZE, CONSTITUENCY_CUDA, CLOZE_SYNTACTIC_TYPES, SPACY_MODEL
from .generate_clozes import mask_answer
from .data_classes import Cloze
import json
import spacy
import pdb
import numpy as np
import os
import re
from .generate_clozes import generate_cloze_from_rule_based, is_appropriate_answer, is_appropriate_cloze, get_cloze_id
from ordered_set import OrderedSet
import subprocess
from multiprocessing import Pool


np.random.seed(2494876)
DIGITS=['0','1','2','3','4;','5','6','7','8','9','.', '$']

from string import punctuation, whitespace

CUSTOMIZED_STOP_WORDS_LIST = set('and anyway anywhere are being beside besides bottom but call can cannot ca could '\
    ' done call can cannot ca could else elsewhere empty enough even ever further' \
    ' get give go had has have hence here hereafter hereby herein hereupon how however indeed keep '\
    ' least just made make many may meanwhile might mine moreover mostly move much must ' \
    ' name namely nevertheless next noone nothing now nowhere of off often ' \
    ' or otherwise perhaps please put quite rather re really regarding same say see seem '\
    ' seemed seeming seems serious several should show so some somehow still such take than that '\
    ' then thence thereafter thereby therefore therein thereupon these this those though '\
    ' thru thus together too toward towards various very very via was well were '\
    ' whatever whence whereafter whereas whereby wherein whereupon wherever whether '\
    " whither whole will with within without would yet willnot willn't can't couldn't wouldn't".split())

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
CUSTOMIZED_STOP_WORDS_LIST.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        CUSTOMIZED_STOP_WORDS_LIST.add(stopword.replace("'", apostrophe))

nlp = spacy.load(SPACY_MODEL)


def _load_constituency_parser():
    archive = load_archive(CONSTITUENCY_MODEL, cuda_device=CONSTITUENCY_CUDA)
    return Predictor.from_archive(archive, 'constituency-parser')


def get_constituency_parsed_clozes(clozes, predictor=None, verbose=True, desc='Running Constituency Parsing'):
    # import pdb
    # pdb.set_trace()
    if predictor is None:
        predictor = _load_constituency_parser()
    jobs = range(0, len(clozes), CONSTITUENCY_BATCH_SIZE)
    for i in tqdm(jobs, desc=desc, ncols=80) if verbose else jobs:
        input_batch = clozes[i: i + CONSTITUENCY_BATCH_SIZE]
        output_batch = predictor.predict_batch_json([{'sentence': c.source_text} for c in input_batch])
        for c, t in zip(input_batch, output_batch):
            root = _get_root_type(t['trees'])
            if root in CLOZE_SYNTACTIC_TYPES:
                c_with_parse = attr.evolve(c, constituency_parse=t['trees'], root_label=root)
                yield c_with_parse


def _get_root_type(tree):
    try:
        t = Tree.fromstring(tree)
        label = t.label()
    except:
        label = 'FAIL'
    return label


def _get_sub_clauses(root, clause_labels):
    """Simplify a sentence by getting clauses"""
    subtexts = []
    for current in root.subtrees():
        if current.label() in clause_labels:
            subtexts.append(' '.join(current.leaves()))
    return subtexts


def _tokens2spans(sentence, tokens):
    off = 0
    spans = []
    for t in tokens:
        span_start = sentence[off:].index(t) + off
        spans.append((span_start, span_start + len(t)))
        off = spans[-1][-1]
    for t, (s, e) in zip(tokens, spans):
        assert sentence[s:e] == t
    return spans


def _subseq2sentence(sentence, tokens, token_spans, subsequence):
    subsequence_tokens = subsequence.split(' ')
    for ind in (i for i, t in enumerate(tokens) if t == subsequence_tokens[0]):
        if tokens[ind: ind + len(subsequence_tokens)] == subsequence_tokens:
            return sentence[token_spans[ind][0]:token_spans[ind + len(subsequence_tokens) - 1][1]]
    raise Exception('Failed to repair sentence from token list')


def get_sub_clauses(sentence, tree):
    clause_labels = CLOZE_SYNTACTIC_TYPES
    root = Tree.fromstring(tree)
    subs = _get_sub_clauses(root, clause_labels)
    tokens = root.leaves()
    token_spans = _tokens2spans(sentence, tokens)
    return [_subseq2sentence(sentence, tokens, token_spans, sub) for sub in subs]


def shorten_cloze(cloze, use_full_sentence=False):
    """Return a list of shortened cloze questions from the original cloze question"""
    simple_clozes = []
    try:
        subs = get_sub_clauses(cloze.source_text, cloze.constituency_parse)
        subs = sorted(subs)
        for sub in subs:
            if use_full_sentence or sub != cloze.source_text:
                sub_start_index = cloze.source_text.index(sub)
                sub_answer_start_index = cloze.answer_start - sub_start_index
                good_start = 0 <= sub_answer_start_index <= len(sub)
                good_end = 0 <= sub_answer_start_index + len(cloze.answer_text) <= len(sub)
                if good_start and good_end:
                    simple_clozes.append(
                        Cloze(
                            cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                            paragraph=cloze.paragraph,
                            source_text=sub,
                            source_start=cloze.source_start + sub_start_index,
                            cloze_text=mask_answer(sub, cloze.answer_text, sub_answer_start_index, cloze.answer_type),
                            answer_text=cloze.answer_text,
                            answer_start=sub_answer_start_index,
                            constituency_parse=None,
                            root_label=None,
                            answer_type=cloze.answer_type,
                            question_text=None,
                            paragraph_title=cloze.paragraph_title
                        )
                    )
    except:
        print(f'Failed to parse cloze: ID {cloze.cloze_id} Text: {cloze.source_text}')
    return simple_clozes


def splitted_cloze(cloze, unsplit_split_json):
    """Return a list of shortened cloze questions from the original cloze question"""
    simple_clozes = []
    try:
        splitted_sents = unsplit_split_json[cloze.source_text.strip()]
        doc = nlp(splitted_sents)
        for sent in doc.sents:
            sub=sent.text
            # if sub != cloze.source_text:
            if is_vp(sub):
                sub_answer_index = sub.find(cloze.answer_text)
                if sub_answer_index>=0:
                    simple_clozes.append(Cloze(
                            cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                            paragraph=cloze.paragraph,
                            source_text=cloze.source_text,
                            source_start=cloze.source_start,
                            cloze_text=mask_answer(sub, cloze.answer_text, sub_answer_index, cloze.answer_type),
                            answer_text=cloze.answer_text,
                            answer_start=cloze.answer_start,
                            constituency_parse=None,
                            root_label=None,
                            answer_type=cloze.answer_type,
                            question_text=None,
                            paragraph_title=cloze.paragraph_title
                        ))
    except:
        print('keyerror: , unsplitted_sent: ', cloze.source_text.strip())
    return simple_clozes

def remove_bracketed_text(text, answer ='MASK'):
    while(True):
        first_bracket_start = text.find('(')
        first_bracket_end = text.find(')')
        bracketed_text = ''
        if first_bracket_start >-1 and first_bracket_end>-1:
            bracketed_text = text[first_bracket_start: first_bracket_end+1]
        elif first_bracket_start >-1 and first_bracket_end==-1:
            bracketed_text = text[first_bracket_start:]
        if bracketed_text!= '':
            if answer not in bracketed_text:
                if first_bracket_end>-1: text = text[:first_bracket_start].strip() + ' ' + text[first_bracket_end + 1:].strip()
                else: text = text[:first_bracket_start].strip()
                continue
            else:
                doc_ = nlp(bracketed_text)
                for token in doc_:
                    if token.pos_ == 'VERB': return bracketed_text

        second_bracket_start = text.find('{')
        second_bracket_end = text.find('}')
        bracketed_text = ''
        if second_bracket_start >-1 and second_bracket_end>-1:
            bracketed_text = text[second_bracket_start: second_bracket_end+1]
        elif second_bracket_start >-1 and second_bracket_end==-1:
            bracketed_text = text[second_bracket_start:]
        if bracketed_text!= '':
            if answer not in bracketed_text:
                if second_bracket_end>-1: text = text[:second_bracket_start].strip() + ' ' + text[second_bracket_end + 1:]
                else: text = text[:second_bracket_start].strip()
                continue
            else:
                doc_ = nlp(bracketed_text)
                for token in doc_:
                    if token.pos_ == 'VERB': return bracketed_text

        third_bracket_start = text.find('[')
        third_bracket_end = text.find(']')
        bracketed_text = ''
        if third_bracket_start > -1 and third_bracket_end > -1:
            bracketed_text = text[third_bracket_start: third_bracket_end + 1]
        elif third_bracket_start > -1 and third_bracket_end == -1:
            bracketed_text = text[third_bracket_start:]
        if bracketed_text != '':
            if answer not in bracketed_text:
                if third_bracket_end>-1: text = text[:third_bracket_start].strip() + ' ' + text[third_bracket_end + 1:].strip()
                else: text = text[:third_bracket_start].strip()
                continue
            else:
                doc_ = nlp(bracketed_text)
                for token in doc_:
                    if token.pos_ == 'VERB': return bracketed_text

        return text


def is_vp(sent, answer = 'MASK'):
    doc_ = nlp(sent)
    SUBJ_FOUND=False
    for token in doc_:
        if 'VERB' in token.pos_  and answer not in token.text and SUBJ_FOUND:
            return True
        elif 'subj' in token.dep_ : SUBJ_FOUND=True
    return False


def remove_punctuation_text(text, punctuation = ",", answer='MASK'):
    if not is_vp(text): return []
    sent_list = []
    ## for simplicity let's assume the sentence is two comma structured.
    first_comma = text.find(punctuation)
    if text[first_comma-1] in DIGITS: return [text]
    if first_comma>-1:second_comma= first_comma+text[first_comma:].find(punctuation)
    else:return [text]

    phrase_before_first_comma = text[:first_comma].strip(punctuation).strip()

    if second_comma>first_comma:
        phrase_in_two_comma = text[first_comma:second_comma+1].strip(punctuation).strip()
        phrase_after_second_comma = text[second_comma:].strip(punctuation).strip()
    else:
        phrase_in_two_comma=''
        phrase_after_second_comma = text[first_comma:].strip()

    is_phrase_before_first_comma_vp = is_vp(phrase_before_first_comma)
    is_phrase_in_two_comma_vp = is_vp(phrase_in_two_comma)
    is_phrase_after_second_comma_vp = is_vp(phrase_after_second_comma)

    if answer in phrase_in_two_comma:
        if is_phrase_in_two_comma_vp:
            sent_list.append(phrase_in_two_comma)
        # else:
            # if is_phrase_before_first_comma_vp:
            #     sent_list.append((phrase_before_first_comma + ' ' + phrase_in_two_comma).strip())
            # elif is_phrase_after_second_comma_vp:
            #     sent_list.append((phrase_in_two_comma + ' ' + phrase_after_second_comma).strip())
            # else: sent_list.append((phrase_before_first_comma + ' ' + phrase_in_two_comma).strip())

    elif answer in phrase_before_first_comma:
        if is_phrase_before_first_comma_vp:
            sent_list.append(phrase_before_first_comma.strip())
        # else:
        #     if is_phrase_in_two_comma_vp:
        #         sent_list.append((phrase_before_first_comma + ' ' + phrase_in_two_comma).strip())
        #     elif is_phrase_after_second_comma_vp:
        #         sent_list.append((phrase_before_first_comma + ' ' + phrase_after_second_comma).strip())
        #     else: sent_list.append(phrase_before_first_comma.strip())

        last_word_before_first_comma =  ' ' .join(phrase_before_first_comma.split()[-2:])
        if answer in last_word_before_first_comma:
            if is_phrase_in_two_comma_vp: sent_list.append((last_word_before_first_comma + ' ' + phrase_in_two_comma).strip())
            # elif is_phrase_after_second_comma_vp: sent_list.append((last_word_before_first_comma + ' ' + phrase_after_second_comma).strip())
    elif answer in phrase_after_second_comma:
        if is_phrase_after_second_comma_vp:
            sent_list.append(phrase_after_second_comma)
        # else:
        #     if is_phrase_before_first_comma_vp:sent_list.append((phrase_before_first_comma + ' ' + phrase_after_second_comma))
        #     elif is_phrase_in_two_comma_vp: sent_list.append((phrase_in_two_comma + ' ' + phrase_after_second_comma).strip())
    # else:
    #     sent_list.append(text)

    return set(sent_list)



def remove_tailing_stopword(semicolon_sent, answer='MASK'):
    semicolon_sent = semicolon_sent.strip(punctuation).strip(whitespace)
    doc_ = nlp(semicolon_sent)
    while(True):
        if doc_[0] in CUSTOMIZED_STOP_WORDS_LIST and doc_[0].text.lower() not in ['a', 'an', 'the']:
            doc_ = doc_[1:]
        else: break

    while (True):
        if doc_[-1] in CUSTOMIZED_STOP_WORDS_LIST and doc_[-1].text.lower() not in ['a', 'an', 'the']:
            doc_ = doc_[:-1]
        else:
            break
    return doc_.text

def length_minimize(cloze, use_title_addition, answer='MASK'):
    """Return a list of shortened cloze questions from the original cloze question"""
    simple_clozes = []
    splitted_sents = cloze.unmasked_cloze_text.strip()
    doc = nlp(splitted_sents)
    for sent in doc.sents:
        # pdb.set_trace()
        sent = sent.text.replace(' %', '%')
        if answer not in sent: continue
        remove_bracketed_sub=remove_bracketed_text(sent, answer=cloze.answer_text)
        removed_comma_sub = remove_punctuation_text(remove_bracketed_sub, answer=cloze.answer_text)
        remove_and_sub = [removed_and_sent for comma_sent in removed_comma_sub for
                          removed_and_sent in remove_punctuation_text(comma_sent, punctuation="and", answer=cloze.answer_text)]
        remove_semicolon_sub = [remove_tailing_stopword(semicolon_sent, answer=cloze.answer_text) \
                                for and_sent in remove_and_sub for semicolon_sent in remove_punctuation_text(and_sent, punctuation=";", answer=cloze.answer_text)]

        for sub in remove_semicolon_sub:
            sub_answer_index = sub.find(answer)
            if sub_answer_index>=0:

                if 'and wh' in sub or 'and how' in sub \
                        or ', do ' in sub  \
                        or '(' in sub or ')' in sub or '{' in sub \
                        or '[' in sub or ']' in sub or ':' in sub:
                    continue


                if not use_title_addition:
                    simple_clozes.append(Cloze(
                        cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                        paragraph=cloze.paragraph,
                        source_text=cloze.source_text,
                        source_start=cloze.source_start,
                        cloze_text=sub,
                        answer_text=cloze.answer_text,
                        answer_start=cloze.answer_start,
                        constituency_parse=None,
                        root_label=None,
                        answer_type=cloze.answer_type,
                        question_text=None,
                        paragraph_title=cloze.paragraph_title
                    ))
                else:
                    list_prepositions = ['in', 'at']
                    list_position = ['FRONT', 'END', '']
                    doc_ = nlp(cloze.paragraph_title)
                    add_preposition = np.random.choice(list_prepositions)
                    if len(doc_.ents)>0:
                        if 'PERSON' in  doc_.ents[0].label_:   add_preposition = np.random.choice(['by', 'for'])
                    add_position = np.random.choice(list_position)

                    if add_position == 'FRONT' and cloze.paragraph_title.lower() not in sub.lower() and cloze.paragraph_title.lower() not in cloze.answer_text.lower():
                        new_cloze_text = add_preposition+ ' '+cloze.paragraph_title+ ', ' + sub
                    elif add_position == 'END' and cloze.paragraph_title.lower() not in sub.lower() and cloze.paragraph_title.lower() not in cloze.answer_text.lower():
                        new_cloze_text = sub + ' '+ add_preposition+ ' '+cloze.paragraph_title
                    else:
                        new_cloze_text = sub

                    # if len(new_cloze_text.split()) > 25 or len(new_cloze_text.split()) < 3: continue

                    simple_clozes.append(Cloze(
                        cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                        paragraph=cloze.paragraph,
                        source_text=cloze.source_text,
                        source_start=cloze.source_start,
                        cloze_text=mask_answer(new_cloze_text, cloze.answer_text, cloze.answer_start, cloze.answer_type),
                        unmasked_cloze_text=new_cloze_text,
                        answer_text=cloze.answer_text,
                        answer_start=cloze.answer_start,
                        constituency_parse=None,
                        root_label=None,
                        answer_type=cloze.answer_type,
                        question_text=None,
                        paragraph_title=cloze.paragraph_title
                    ))
    return simple_clozes

def get_q_ans_start(rule_based_q_line, source_texts, paragraph):
    rule_based_q_line = rule_based_q_line.replace("``", '"').replace("''", '"')
    line_stream = rule_based_q_line.split('\t')
    title = paragraph.title
    q = line_stream[0]
    try:
        cloze_text = re.sub(r'(\d).\s+(\d)', r'\1.\2', line_stream[1].strip())
    except:
        print(rule_based_q_line, ' was not processed ', flush=True)
        print(line_stream, ' was not processed ', flush=True)
    try:
        answer = re.sub(r'(\d).\s+(\d)', r'\1.\2', line_stream[2].lower().strip())
    except:
        print('line_stream[2] was not found from ', line_stream)
        return None, None, None, None, None, None, None
    context = paragraph.text
    while(True):
        doc = nlp(answer)
        if doc[0].tag_ in ['IN', 'TO', 'DT']  :
            if len(doc)>1:
                answer = doc[1:].text
                continue
            else: return None, None, None, None, None, None, None
        elif doc[-1].tag_ in ['IN', 'TO', 'DT']:
            answer = doc[:-1].text
            continue
        else:
            if len(doc.ents)>0: answer_type = doc.ents[0].label_
            else: answer_type=None
            break
    try:
        source_text = source_texts[cloze_text.strip('.')]
    except:
        try:
            cloze_text = cloze_text.replace(" ' ", "")
            source_text = source_texts[cloze_text.strip('.')]
        except:
            c_text = cloze_text
            source_text = ''
            while(True):
                c_text = c_text[len(c_text)//2]
                for cloze,source in source_texts.items():
                    if c_text in cloze:
                        source_text = source
                        break
                if source_text!='':break
                if len(c_text) == 1: return None, None, None, None, None, None, None

    title_phrase_to_remove = ['in '+title, 'at '+title, 'by '+title, 'for '+title]
    for phrase_to_remove in title_phrase_to_remove:
        if phrase_to_remove in answer and phrase_to_remove not in source_text:
            answer=answer.replace(phrase_to_remove, '')
    source_start = context.find(source_text)
    if source_start==-1:return None, None, None, None, None, None, None
    ref_answer_idx = source_text.lower().find(answer.lower())
    if ref_answer_idx == -1:
        ref_answer_idx = re.sub(r'\b(,|``|''|")\b', '', source_text).lower().find(re.sub(r'\b(,|``|''|")\b', '', answer.lower()))
    if ref_answer_idx == -1:
        # print("question: ", q, 'answer: ', answer, 'cloze_text: ', cloze_text,  'source_text: ', source_text, ' :::::Answer not found in source_text')
        return None, None, None, None, None, None, None
    answer_start=source_start+ref_answer_idx
    answer_text = context[answer_start: answer_start+len(answer)]
    if  answer_text.lower() != answer.lower(): return None, None, None, None, None, None, None
    return q, answer_text, cloze_text, source_text, source_start, ref_answer_idx, answer_type

def rule_based_ques_generation(clozes_p):
    # pdb.set_trace()
    input_text_list = []
    source_texts = {}
    for cloze in clozes_p:
        input_text_list.append(cloze.source_text.strip('.'))
        input_text_list.append(cloze.unmasked_cloze_text.strip('.'))
        source_texts[cloze.unmasked_cloze_text.strip('.').replace("'", '').replace('/','\/')] =  cloze.source_text.strip('.')
        source_texts[cloze.source_text.strip('.').replace("'", '').replace('/','\/')] = cloze.source_text.strip('.')
        source_texts[cloze.source_text.strip('.').replace("'", '').replace('/','\/').replace(" ' ", "")] = cloze.source_text.strip('.')

    owd = os.getcwd()
    input_text = ". ".join(OrderedSet(input_text_list)).replace(';', ',,').replace('(', '\(').replace(')', '\)').replace("'", "\'")
    os.chdir('/export/home/SyntaxQG/')
    rule_based_result = os.popen(' echo ' + input_text + ' | bash /export/home/SyntaxQG/run_no_coref.sh ').read().strip()
    if rule_based_result=='':
        rule_based_result = os.popen(' echo ' + input_text.replace("'", "^^") + ' | bash /export/home/SyntaxQG/run_no_coref.sh ').read().strip()
    if rule_based_result=='':
        rule_based_result = os.popen(' echo ' + input_text.replace("'", "^^").replace("(", "^^^").replace(')', '^^^^') \
                                     + ' | bash /export/home/SyntaxQG/run_no_coref.sh ').read().strip()
    if rule_based_result == '':
        rule_based_result = os.popen(' echo ' + input_text.replace("(", "^^^").replace(')', '^^^^') \
                                     + ' | bash /export/home/SyntaxQG/run_no_coref.sh ').read().strip()
    rule_based_result = rule_based_result.replace(' %', '%').replace(' $', '$'). \
        replace('$ ', '$').replace(' ^ ^ ', "'").replace(' ^ ^ ^ ', '(').replace(" ^ ^ ^ ^ ", ')').replace(' , , ', ';')
    os.chdir(owd)
    questions = []
    simple_clozes = []
    # print('input_text:', input_text)
    for rule_based_q_line in rule_based_result.split('\n'):
        if rule_based_q_line!='':
            # if 'The impressive growth in gross domestic product of the state has been reported by the Ministry of Statistics and Programme Implementation' in clozes_p[0].paragraph.text:
            #     print(f' q: {q} \n answer_text: {answer_text} \n cloze_text: {cloze_text} \n \
            #     source_text: {source_text} \n source_start:{source_start} \n answer_start: {answer_start}')
            #     pdb.set_trace()
            q, answer_text, unmasked_cloze_text, source_text, source_start, answer_start, answer_type = \
                get_q_ans_start(rule_based_q_line=rule_based_q_line, source_texts = source_texts, paragraph=clozes_p[0].paragraph)
            if q and q not in questions and is_appropriate_answer(answer_text) and is_appropriate_cloze(unmasked_cloze_text):
                questions.append(q)
                if answer_type: cloze_text = mask_answer(unmasked_cloze_text, answer_text, answer_start, answer_type)
                else: cloze_text =None
                simple_clozes.append(
                    Cloze(
                        cloze_id=get_cloze_id(clozes_p[0].paragraph.text, unmasked_cloze_text, answer_text),
                        paragraph=clozes_p[0].paragraph,
                        source_text=source_text,
                        source_start=source_start,
                        cloze_text=cloze_text,
                        unmasked_cloze_text=unmasked_cloze_text,
                        answer_text=answer_text,
                        answer_start=answer_start,
                        constituency_parse=None,
                        root_label=None,
                        answer_type=answer_type,
                        question_text=q,
                        paragraph_title=clozes_p[0].paragraph.title
                    )
                )
    return simple_clozes

