# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Module to handle reading, (de)seriaizing and dumping data
"""
import json
import attr
from .data_classes import Cloze, Paragraph
import hashlib
from .configs import MAX_PARAGRAPH_WORDSIZE_THRESHOLD, MAX_PARAGRAPH_CHAR_LEN_THRESHOLD, MAX_PARAGRAPH_WORD_LEN_THRESHOLD


def clozes2squadformat(clozes, out_fobj):
    assert all([c.question_text is not None for c in clozes]), 'Translate these clozes first, some dont have questions'
    data = {cloze.paragraph.paragraph_id: {'context': cloze.paragraph.text, 'qas': []} for cloze in clozes}
    for cloze in clozes:
        qas = data[cloze.paragraph.paragraph_id]
        qas['qas'].append({
            'question': cloze.question_text, 'id': cloze.cloze_id,
            'answers': [{'text': cloze.answer_text, 'answer_start': cloze.answer_start + cloze.source_start}]
        })
    squad_dataset = {
        'version': 1.1,
        'data': [{'title': para_id, 'paragraphs': [payload]} for para_id, payload in data.items()]
    }
    json.dump(squad_dataset, out_fobj)


def _parse_attr_obj(cls, serialized):
    return cls(**json.loads(serialized))


def dumps_attr_obj(obj):
    return json.dumps(attr.asdict(obj))


def parse_clozes(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Cloze, serialized)


def dump_clozes(clozes, fobj):
    for cloze in clozes:
        fobj.write(dumps_attr_obj(cloze))
        fobj.write('\n')


def _get_paragraph_id(text):
    return hashlib.sha1(text.encode()).hexdigest()


def parse_paragraphs_from_txt(fobj):
    for paragraph_text in fobj:
        para_text = paragraph_text.strip('\n')
        if para_text != '' :
           yield Paragraph(
               paragraph_id=_get_paragraph_id(para_text),
               text=para_text,
               title=None
           )


def parse_paragraphs_from_jsonl(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Paragraph, serialized)


def parse_paragraphs_from_jsonl_with_title_and_no_id(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            serialized_json = json.loads(serialized.strip())
            paragraph_text = serialized_json["context"].replace('-based', ' based')
            p_char_len_good = len(paragraph_text) <= MAX_PARAGRAPH_CHAR_LEN_THRESHOLD
            p_word_len_good = len(paragraph_text.split()) <= MAX_PARAGRAPH_WORD_LEN_THRESHOLD
            p_wordsize_good = all([len(w) <= MAX_PARAGRAPH_WORDSIZE_THRESHOLD for w in paragraph_text.split()])
            p_good = p_char_len_good and p_word_len_good and p_wordsize_good
            if p_good:
                yield Paragraph(
                paragraph_id=_get_paragraph_id(serialized),
                text=paragraph_text,
                title=serialized_json["title"]
            )
def parse_paragraphs_from_json(fobj, debug_mode, size=60000):
    input_data = json.load(fobj)["data"]
    count = 0
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            count += 1
            # if not debug_mode and count<=size: continue
            paragraph_text = paragraph["context"]
            para_text = paragraph_text.strip('\n')
            if para_text != '':
                yield Paragraph(
                    paragraph_id=_get_paragraph_id(para_text),
                    text=para_text,
                    title=None
                )

                if debug_mode and count > 10: break
                elif count>size:
                    print('cutoff reading more paragraph after: ',count)
                    break
        if debug_mode and count > 10: break
        elif count>size:
            print('cutoff reading more entry after: ', count)
            break



