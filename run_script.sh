#!/usr/bin/env bash

export  unsplit_split_f='./data/predicted_unsplit_split.json '
export input_file=./data/sampled_wiki_contexts.jsonl
#export input_file=./data/dummy_input.jsonl_with_title_and_no_id
### Just to check the default code works or not'''
python -m unsupervisedqa.generate_synthetic_qa_data $input_file ./data/sample_wiki_rule_based  \
    --input_file_format "jsonl_with_title_and_no_id" \
    --output_file_format "jsonl,squad" \
    --translation_method unmt \
    --use_named_entity_clozes \
    --use_subclause_clozes \
    --use_wh_heuristic \
    --unsplit_split_json_file $unsplit_split_f \
    --use_prohibit_combined_question \
    --use_length_minimization \
    --use_title_addition &>>data/log_no_subclause_no_split.txt
