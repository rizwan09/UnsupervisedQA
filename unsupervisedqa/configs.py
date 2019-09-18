# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Config file for UnsupervisedQA
"""
import os

HERE = os.path.dirname(os.path.realpath(__file__))
SEED = 10





## Sensible filter thresholds:
## Cloze selection criteria
MIN_CLOZE_WORD_LEN = 5
MAX_CLOZE_WORD_LEN = 40
MIN_CLOZE_WORDSIZE = 1
MAX_CLOZE_WORDSIZE = 20
MIN_CLOZE_CHAR_LEN = 30
MAX_CLOZE_CHAR_LEN = 300
MIN_ANSWER_WORD_LEN = 1
MAX_ANSWER_WORD_LEN = 20
MIN_ANSWER_CHAR_LEN = 3
MAX_ANSWER_CHAR_LEN = 50
## remove  with more characters than this
MAX_PARAGRAPH_CHAR_LEN_THRESHOLD = 2000
MAX_QUESTION_CHAR_LEN_THRESHOLD = 200
## remove items with more words than this
MAX_PARAGRAPH_WORD_LEN_THRESHOLD = 400
MAX_QUESTION_WORD_LEN_THRESHOLD = 40
## remove items that have words with more characters than this
MAX_PARAGRAPH_WORDSIZE_THRESHOLD = 20
MAX_QUESTION_WORDSIZE_THRESHOLD = 20


## Spacy Configs:
SPACY_MODEL = 'en'

## constituency parser configs:
CONSTITUENCY_MODEL = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
CONSTITUENCY_BATCH_SIZE = 32
CONSTITUENCY_CUDA = 0
CLOZE_SYNTACTIC_TYPES = {'S', }


# UNMT configs:
PATH_TO_UNMT = os.path.join(HERE, '../UnsupervisedMT/NMT')
UNMT_DATA_DIR = os.path.join(HERE, '../data')
UNMT_MODEL_SUBCLAUSE_NE_WH_HEURISTIC = os.path.join(UNMT_DATA_DIR, 'subclause_ne_wh_heuristic')
UNMT_MODEL_SUBCLAUSE_NE = os.path.join(UNMT_DATA_DIR, 'subclause_ne')
UNMT_MODEL_SENTENCE_NE = os.path.join(UNMT_DATA_DIR, 'sentence_ne')
UNMT_MODEL_SENTENCE_NP = os.path.join(UNMT_DATA_DIR, 'sentence_np')


# UNMT Tools:
TOOLS_DIR = os.path.join(PATH_TO_UNMT, 'tools')
MOSES_DIR = os.path.join(TOOLS_DIR, 'mosesdecoder')
FASTBPE_DIR = os.path.join(TOOLS_DIR, 'fastBPE')
N_THREADS_PREPRO = 16
UNMT_BEAM_SIZE = 0
UNMT_BATCH_SIZE = 30


# XLM configs:
PATH_TO_XLM = os.path.join(HERE, '../../../../XLM')
XLM_DATA_DIR = os.path.join(HERE, '../../../../XLM/data/nq_to_cloze2')
XLM_MODEL= os.path.join(PATH_TO_XLM, 'dumped/UNMT_NQ_CLOZE/lsyt0bufl4/checkpoint.pth ')
TOOLS_PATH = os.path.join(PATH_TO_XLM, 'tools')
TRANSLATOR = os.path.join(PATH_TO_XLM,"translate.py")
XLM_FASTBPE = os.path.join(TOOLS_PATH,'fastBPE/fast')
PROC_PATH = os.path.join(XLM_DATA_DIR, 'processed/nq.en-cloze.en')
BPE_CODES=os.path.join(PROC_PATH,'codes')
SRC_VOCAB=os.path.join(PROC_PATH,'vocab.nq.en')
TGT_VOCAB=os.path.join(PROC_PATH,'vocab.cloze.en')
FULL_VOCAB=os.path.join(PROC_PATH,'vocab.nq.en-cloze.en')
MOSES=os.path.join(TOOLS_PATH,'mosesdecoder')
REPLACE_UNICODE_PUNCT=os.path.join(MOSES,'scripts/tokenizer/replace-unicode-punctuation.perl')
NORM_PUNC=os.path.join(MOSES, 'scripts/tokenizer/normalize-punctuation.perl')
REM_NON_PRINT_CHAR=os.path.join(MOSES, 'scripts/tokenizer/remove-non-printing-char.perl')
TOKENIZER=os.path.join(MOSES,'scripts/tokenizer/tokenizer.perl')
INPUT_FROM_SGM=os.path.join(MOSES,'scripts/ems/support/input-from-sgm.perl')
SRC_PREPROCESSING=f'{REPLACE_UNICODE_PUNCT} | {NORM_PUNC} -l nq.en | {REM_NON_PRINT_CHAR} | {TOKENIZER} -l nq.en -no-escape -threads 8'



# CLOZE MASKS:
NOUNPHRASE_LABEL = 'NOUNPHRASE'
NOUN_TAG = 'NOUNTAG'
CLOZE_MASKS = {
    'PERSON': 'IDENTITYMASK',
    'NORP': 'IDENTITYMASK',
    'FAC': 'PLACEMASK',
    'ORG': 'IDENTITYMASK',
    'GPE': 'PLACEMASK',
    'LOC': 'PLACEMASK',
    'PRODUCT': 'THINGMASK',
    'EVENT': 'THINGMASK',
    'WORKOFART': 'THINGMASK',
    'WORK_OF_ART': 'THINGMASK',
    'LAW': 'THINGMASK',
    'LANGUAGE': 'THINGMASK',
    'DATE': 'TEMPORALMASK',
    'TIME': 'TEMPORALMASK',
    'PERCENT': 'NUMERICMASK',
    'MONEY': 'NUMERICMASK',
    'QUANTITY': 'NUMERICMASK',
    'ORDINAL': 'NUMERICMASK',
    'CARDINAL': 'NUMERICMASK',
    NOUNPHRASE_LABEL: 'NOUNPHRASEMASK'
}

CLOZE_FINEGRAINED__MASKS = {
    'PERSON': 'PERSONMASK',
    'NORP': 'NORPMASK',
    'FAC': 'FACMASK',
    'ORG': 'ORGMASK',
    'GPE': 'GPEMASK',
    'LOC': 'LOCMASK',
    'PRODUCT': 'PRODUCTMASK',
    'EVENT': 'EVENTMASK',
    'WORKOFART': 'WORKOFARTMASK',
    'WORK_OF_ART': 'WORK_OF_ARTMASK',
    'LAW': 'LAWMASK',
    'LANGUAGE': 'LANGUAGEMASK',
    'DATE': 'DATEMASK',
    'TIME': 'TEMPORALMASK',
    'PERCENT': 'PERCENTMASK',
    'MONEY': 'MONEYMASK',
    'QUANTITY': 'QUANTITYMASK',
    'ORDINAL': 'ORDINALMASK',
    'CARDINAL': 'CARDINALMASK',
    NOUNPHRASE_LABEL: 'NOUNPHRASEMASK',
    NOUN_TAG: 'NOUNTAGMASK'

}


HEURISTIC_CLOZE_TYPE_QUESTION_MAP = {
    'PERSON': ['Who', ],
    'NORP': ['Who', ],
    'FAC': ['Where', ],
    'ORG': ['Who', ],
    'GPE': ['Where', ],
    'LOC': ['Where', ],
    'PRODUCT': ['What', ],
    'EVENT': ['What', ],
    'WORKOFART': ['What', ],
    'WORK_OF_ART': ['What', ],
    'LAW': ['What', ],
    'LANGUAGE': ['What', ],
    'DATE': ['When', ],
    'TIME': ['When', ],
    'PERCENT': ['How much', 'How many'],
    'MONEY': ['How much', 'How many'],
    'QUANTITY': ['How much', 'How many'],
    'ORDINAL': ['How much', 'How many'],
    'CARDINAL': ['How much', 'How many'],
}