# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Wrapper for UnsupervisedNMT inference-time functionality
"""
import attr
import os
from .configs import CLOZE_MASKS, XLM_MODEL, PATH_TO_XLM, XLM_DATA_DIR, SRC_PREPROCESSING, XLM_FASTBPE, BPE_CODES, SRC_VOCAB, TRANSLATOR
from src.utils import restore_segmentation
import subprocess
import tempfile
import sys
sys.path.append(PATH_TO_XLM) # simple hack on the path to import Unsupervised NMT functionality
import torch
import pdb
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout (or leave in train mode to finetune)


def _dump_questions_for_translation(clozes, dump_path, wh_heuristic):

    def _wh_heurisistic(cloze):
        cloze_mask = CLOZE_MASKS[cloze.answer_type]
        cloze_text = cloze_mask + ' ' + cloze.question_text
        return cloze_text

    with open(dump_path, 'w') as fobj:
        for c in clozes:
            cloze_text = _wh_heurisistic(c) if wh_heuristic else c.question_text
            fobj.write(cloze_text)
            fobj.write('\n')


def _associate_clozes_to_clozes(clozes, translation_output_file, wh_heuristic):

    def _clean_wh_heurisistic(question_text):
        return ' '.join(question_text.split(' ')[1:])

    clozes_with_clozes = []
    translations = []

    for line in open(translation_output_file):
        if line.strip() != '':
            outp = line.strip()
            translations.append(_clean_wh_heurisistic(outp) if wh_heuristic else outp)
    assert len(clozes) == len(translations), "mismatch between number of clozes and translations"
    for c, q in zip(clozes, translations):
        cloze_mask = CLOZE_MASKS[c.answer_type]
        sent1 = c.cloze_text.replace(cloze_mask, c.answer_text)
        sent2 = ' '.join([w if 'MASK' not in w else c.answer_text for w in q.split()])
        # print('cloze: ', c.cloze_text, 'sent1: ', sent1)
        # print('question: ', c.question_text, 'unmasked_cloze: ', q, 'sent2: ', sent2)
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        # print('prediction: ', prediction)
        if prediction!=1:
            c_with_q = attr.evolve(c, unmasked_cloze_text=q)
            clozes_with_clozes.append(c_with_q)
    return clozes_with_clozes


def get_XLM_clozes_for_questions(clozes):


    with tempfile.TemporaryDirectory() as tempdir:
        raw_cloze_file = os.path.join(tempdir, 'dev.nq.en')
        _dump_questions_for_translation(clozes, raw_cloze_file, wh_heuristic=True)

        tok_cloze_file = os.path.join(tempdir, 'dev.nq.tok')
        cmd = f'cat {raw_cloze_file} | {SRC_PREPROCESSING} > {tok_cloze_file}'
        subprocess.check_call(cmd, shell=True)

        bpe_cloze_file = os.path.join(tempdir, 'dev.nq.tok.bpe')
        cmd = f'{XLM_FASTBPE} applybpe {bpe_cloze_file} {tok_cloze_file} {BPE_CODES} {SRC_VOCAB}'
        subprocess.check_call(cmd, shell=True)

        cmd = f'rm -r {XLM_DATA_DIR}/data/translate_nq_cloze_temp'
        if os.path.exists(os.path.join(XLM_DATA_DIR,'data/translate_nq_cloze_temp')):
            subprocess.check_call(cmd, shell=True)
        translation_output_path = os.path.join(tempdir, 'dev.cloze.en')

        cmd = f'cat {bpe_cloze_file} | python {TRANSLATOR} --exp_name translate_nq_cloze_temp ' \
            f'--src_lang nq.en --tgt_lang cloze.en --model_path {XLM_MODEL} --output_path {translation_output_path}'
        subprocess.check_call(cmd, shell=True)
        restore_segmentation(translation_output_path)

        clozes_with_questions = _associate_clozes_to_clozes(clozes, translation_output_path, wh_heuristic=False)

    return clozes_with_questions

