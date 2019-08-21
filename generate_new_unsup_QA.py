import wikipedia as wiki
import pdb
import numpy as np
import json
from itertools import groupby
import spacy
nlp = spacy.load("en")
np.random.seed(2494876)
MASK_WORD = "_MASK_"

def sample_contexts_from_english_wiki(how_many_titles, how_many_contexts_for_each_title, input_titles, output_file, input_size = 10000):
    with open(input_titles, 'r') as title_reader:
        title_lines = title_reader.readlines()
    random_title_ids = sorted(np.random.choice(input_size, how_many_titles, replace=False))
    print(random_title_ids)
    total_contxts = 0
    with open(output_file, 'w', encoding='utf-8') as writer:
        title_c = 0
        for random_line_number in random_title_ids:
            try:
                title = title_lines[random_line_number].split('\t')[1].strip()
                cntxt_array = wiki.page(title.strip()).content
            except:
                print('title: ',title, ' has no page', ' with id: ', random_line_number)
                try:
                    while(True):
                        try:
                            random_line_number = np.random.randint(random_title_ids[title_c]+1, random_title_ids[title_c+1])
                            print('new number generated btwn ', random_title_ids[title_c]+1, random_title_ids[title_c+1], ' is ', random_line_number)
                            title = title_lines[random_line_number].split('\t')[1].strip()
                            cntxt_array = wiki.page(title.strip()).content
                            break
                        except:continue
                except:
                    continue
                print('new title: ', title)
            contexts_candidates = [cntxt for cntxt in cntxt_array.split('\n') if  len(cntxt.split())<=500 and len(cntxt.split())>=10 and '==' not in cntxt ]
            num_contexts=len(contexts_candidates)
            if how_many_contexts_for_each_title>num_contexts: random_context_ids = np.random.choice(num_contexts, num_contexts, replace=False)
            else: random_context_ids =  np.random.choice(num_contexts, how_many_contexts_for_each_title, replace=False)
            for context_id in random_context_ids:
                    context_object = {"title": title, "context": contexts_candidates[context_id].strip()}
                    json.dump(context_object, writer)
                    writer.write('\n')
                    total_contxts+=1
            title_c+=1
            print('processed pages: ', title_c, ' #context: ', total_contxts)
    print('total #context: ', total_contxts)

def coref_part(predicted_splitted_file, temp_coref_input_file, temp_coref_output_file):
    with open(temp_coref_input_file, 'w') as writer, open(predicted_splitted_file, 'r') as splitted_f:
        unsplt_splt_dict = {}
        for spltd_line in splitted_f:
            spltd_line = ' '.join([elm[0] for elm in groupby(spltd_line.split())]). \
                replace(' @@ ', '').replace('@@ ', '').replace(' @@', '').replace('@@', ''). \
                replace(" ' ", "'").replace(" ?", "?").replace(" .", ".").replace(" ,", ","). \
                replace(" ;", ";").replace(" !", "!").replace(" :", ":"). \
                replace(' ".', '".').strip()

            ## Extract contexts from SQuAD and generate like followings:
            ''''{
                "clusters": [],
                "doc_key": "nw",
                "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
                "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
            }'''

            context_dict = {"clusters": [], "doc_key": "nw"}
            sentences = []
            speakers = []
            doc = nlp(spltd_line)
            for sentence in doc.sents:
                words = [token.text for token in sentence]
                sentences.append(words)
                speakers.append(["" for i in range(len(words))])
            context_dict["sentences"] = sentences
            context_dict["speakers"] = speakers
            json.dump(context_dict, writer)
            writer.write('\n')

    ### manual bash run
    UGQA_DIR = '/export/home/UGQA/examples/'
    SQUAD_DIR = '/export/home/SQuAD/'
    E2E_COREF_DIR = '/export/home/e2e-coref/'
    print('cd ' + E2E_COREF_DIR)
    print('source activate allennlp')
    print('python predict.py final ' + temp_coref_input_file + ' ' + temp_coref_output_file)
    print('source deactivate allennlp')
    print('cd ' + UGQA_DIR)


def make_spliited_sentence_in_json(unsplitted_file, predicted_splitted_file, output_json_file,\
                                   temp_coref_input_file, temp_coref_output_file):

    # coref_part(predicted_splitted_file, temp_coref_input_file, temp_coref_output_file)
    predicted_splitted_sentences_dict={}
    with open(unsplitted_file, 'r') as unsplitted_f, \
             open(output_json_file, 'w') as unsplit_split_f, open(temp_coref_output_file, 'r') as coreferenced_file:
        num = 0
        for unspltd_line, coref_line in zip(unsplitted_f, coreferenced_file):
            num = num + 1
            coref_json = json.loads(coref_line)
            unspltd_line = unspltd_line.strip()
            if len(unspltd_line.split())<2:
                predicted_splitted_sentences_dict[unspltd_line]=unspltd_line
                continue
            else:
                splitted_string = ''
                flat_list_words = [word for sentence in coref_json["sentences"] for word in sentence]
                if len(coref_json['predicted_clusters'])>0:
                    for cluster in coref_json["predicted_clusters"]:
                        ref = ' '.join(flat_list_words[cluster[0][0]:cluster[0][1]+1])
                        corefs = [ flat_list_words[coref[0]:coref[1]+1] for coref in cluster[1:]]
                        for coref in cluster[1:]:
                            flat_list_words[coref[0]] = ref
                            for i in range(coref[0] + 1, coref[1]+1):
                                flat_list_words[i] = ""
                splitted_string = ' '.join(filter(None,flat_list_words)).replace(" \'","'").replace(' - ', '-')
                predicted_splitted_sentences_dict[unspltd_line] = splitted_string
                if len(predicted_splitted_sentences_dict)%5000==0: print("#processed: ", len(predicted_splitted_sentences_dict))

        print("#unique sentences: ", len(predicted_splitted_sentences_dict), ' num: ', num)
        json.dump(predicted_splitted_sentences_dict, unsplit_split_f)



def combine_two_QA_json_files(file1, file2, file, coref_issue=False):
    if not coref_issue:
        with open(file1, "r", encoding='utf-8') as reader:
            print(' Loading QA file1')
            input_data1 = json.load(reader)
            print(' Loaded QA file1')
        with open(file2, "r", encoding='utf-8') as reader:
            print(' Loading QA file2')
            input_data2 = json.load(reader)
            print(' Loaded QA file2')

        print('first elemets: ', len(input_data1['data']))
        print('second elemets: ', len(input_data2['data']))
        input_data1['data'] += input_data2['data']
        print('merge size: ', len(input_data1['data']))
        if file:
            with open(file, 'w') as writer:
                print('writing')
                json.dump(input_data1, writer)
                print('Done')



if __name__=='__main__':

    ## 'top_ranked_wiki_titles.txt is from https://www.nayuki.io/res/computing-wikipedias-internal-pageranks/wikipedia-top-pageranks.txt'
    # sample_contexts_from_english_wiki(how_many_titles=1000, how_many_contexts_for_each_title=50, \
    #                                   input_titles='./data/top_ranked_wiki_titles.txt', output_file='./data/sampled_wiki_contexts.jsonl')

    ## Then train a XLM unsplitted.en-splitted.en model
    ## Collect spliitted prediction for each sentence identified by spacy and coref it by running script_translate.sh.
    ## Then make a json file: key: each sent in context, value: coref splitted sentence (REMOVE_BPE)
    ## has some manual run inside

    # make_spliited_sentence_in_json(unsplitted_file='/export/home/XLM/data/mono/web-wiki-split-data/wiki_sample_qa_sentences.no_new_line.txt', \
    #                                predicted_splitted_file='/export/home/XLM/data/SPL_tranlsated.txt', \
    #                                output_json_file="/export/home/UGQA/examples/UnsupervisedQA/data/predicted_unsplit_split.json", \
    #                                temp_coref_input_file = '/export/home/e2e-coref/data/splitted_sample_sentences.jsonl', \
    #                                temp_coref_output_file = '/export/home/e2e-coref/splitted_sample_sentences.coreferenced.jsonl')
    ## In UnsupervisedQA dir run run_script.sh
    ## Combine this with shard 0
    combine_two_QA_json_files('/export/home/UGQA/examples/subset_qa_train_start_idx_0_end_idx_250000.json', \
                        '/export/home/UGQA/examples/UnsupervisedQA/data/sample_wiki_no_subclause_add_title_length_minimized.squad.json',
                        '/export/home/UGQA/examples/UnsupervisedQA/data/unsup_shard_0_and_sample_wiki_no_subclause_add_title_length_minimized.squad.json',  coref_issue=False)
