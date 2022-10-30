import os
import random
import json
import csv
import sys
import os
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from ftfy import fix_text
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer

# from utils import detokenizer


'''

In general, examples should be of the form:

{
'options': [opt_1, opt_2, ..., opt_m]
'label': l  # index of correct option
}

opt_i is an option of the form:

{
'premise': premise # the question premise (string)
'hypothesis': h # hypothesis answer (str) we calculate conditional P(hypothesis|premise)
'unc_presmise': up # the premise for calculating uncond likelihood (str)
'unc_hypothesis': uh # the hypothesis used for calculating uncond likelihood P(hypothesis) 
                     # this will often just be hypothesis but may differ slightly for format

}

'''


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def detokenizer(string):
    # ari custom
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    string = string.replace("` ", '"')
    string = string.replace(" ' ", '" ')
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" ,", ",")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    # string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    # string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    # ari custom
    string = string.replace(" n't ", "n't ")
    string = string.replace(" 'd ", "'d ")
    string = string.replace(" 'm ", "'m ")
    string = string.replace(" 're ", "'re ")
    string = string.replace(" 've ", "'ve ")
    return string


def load_label(path):
    label2token = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            for w in line.strip().split(", "):
                label2token[i].append(f" {w}")

    # trim to the same length
    min_len = min([len(v) for v in label2token.values()])
    for k, v in label2token.copy().items():
        label2token[k] = v[:min_len]
    # print("label2token: ", label2token)
    return label2token

def load_examples_copa(path, return_tuple = False):
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
            
        # legacy, using tuples
        if return_tuple:
            examples_copa  += [{'options': [(' '+p[:-1] ,bridge + a1),
                                                (' '+p[:-1] , bridge + a2)], 
                      'label':int(value)-1, 'asks-for': asks_for, 'bridge':bridge}]
        else:
            examples_copa  += [{'options': [{'premise':' '+p[:-1] + bridge,
                                             'hypothesis': ' '+ a1,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a1},
                                           {'premise':' '+p[:-1] + bridge,
                                             'hypothesis': ' '+a2,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a2}], 
                      'label':int(value)-1}]
    return examples_copa

'''

This loads COPA, putting hypothesis before the premise

(so forward LM score is PMI)

'''
## Loads from an xml
def load_examples_copa_rev(path):
    
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text[:1].lower() +  children[0].text[1:]
        a1 = children[1].text[:1].lower() +  children[1].text[1:-1]
        a2 = children[2].text[:1].lower() +  children[2].text[1:-1]
        if asks_for =='effect':
            bridge = ' because'
        elif asks_for =='cause':
            bridge = ' so'
        else: 
            assert(False)
            
        examples_copa  += [{'options': [{'premise':' '+a1 + bridge,
                                         'hypothesis':  ' ' +p,
                                         'uncond_premise':bridge,
                                         'uncond_hypothesis':' ' +p},
                                       {'premise':' '+a2 + bridge,
                                         'hypothesis': ' ' +p,
                                         'uncond_premise':bridge,
                                         'uncond_hypothesis':' '+p}], 
                            'label':int(value)-1, }]
    
    return examples_copa

def load_examples_storycloze(path, return_tuple=False, interpolate=False):
    data = []
    with open(path) as fp:
        reader = csv.DictReader(fp, delimiter = "\t")
        for row in reader:
            d = {}
            if interpolate is False:
                premise = row["InputSentence1"]
                premise = f'{premise} {row["InputSentence2"]}'
                premise = f'{premise} {row["InputSentence3"]}'
                premise = f'{premise} {row["InputSentence4"]}'
                d['premise'] = premise
            else:
                d['premise'] = row["retrieve"]
            similarity = float(row["sim"]) if "sim" in row else None
            d["sim"] = similarity
            hypotheses = [ row['RandomFifthSentenceQuiz1'], row['RandomFifthSentenceQuiz2'] ]
            d['hypotheses'] =  hypotheses
            correct_hypothesis = int(row['AnswerRightEnding']) - 1
            d['correct_hypothesis'] = correct_hypothesis
            d['id'] = row['InputStoryid']
            data.append(d)

    examples = []
    for d in data:
        end = '.'
        # take the punctuation from the end of the story as a prefix to 
        # the last sentence, so that we have something to condition on
        # for P(final_sentence)
        if d['premise'][-1] in '!.':
            end = d['premise'][-1] 
            d['premise'] = d['premise'][:-1]

        if return_tuple:
            examples += [{'options':[(d['premise'],end +' ' +h) for h in d['hypotheses']],
                        'label':d['correct_hypothesis'],
                        'sim': d['sim']}]
        else:
            examples += [{'options':[{'premise':d['premise'] + end,
                                      'hypothesis':  ' ' +h,
                                      'uncond_premise': ' The story continues:' ,
                                      'uncond_hypothesis':  ' ' + h,
                                      }for h in d['hypotheses']],
                        'label':d['correct_hypothesis'],
                        'sim': d['sim']}]
    return examples

def load_examples_hellaswag(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]
    examples = []
    for d in data:
        premise = d["ctx"].strip()
        last_space_index = premise.rfind(' ')
        uncond_premise = premise[last_space_index:]

        options = []
        for hypothesis in d['endings']:
            o = { 'premise' : premise, 'uncond_premise' : uncond_premise } 
            o['hypothesis'] = ' ' + hypothesis
            o['uncond_hypothesis'] = ' ' + hypothesis
            options.append(o)
        label = d['label']
        examples.append( { 'options' : options, 'label' : label } )
    return examples


def load_examples_rotten_tomatoes(path, args):
    hypotheses = [' negative', ' positive']
    label_list = [' terrible', ' great']
    label_path = "./task_data/sst2/label_names_sentidict.txt"
    label2synonym = load_label(label_path)
    prompt = " It was"

    '''
    load train examples
    '''
    icl_str = ""
    train_path = path.replace("dev", "train")
    train_path = train_path.replace("test", "train")
    if args.k_shot > 0:
        train_examples = []
        with open(train_path, 'r') as json_file:
            json_list = list(json_file)

        for row in json_list:
            row = json.loads(row)
            label_str = " " + row["output"]
            label = hypotheses.index(label_str)
            summary = row["input"]
            premise = f'{summary}{prompt}'
            options = []
            for h in label_list:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            train_examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label_list})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)

    examples = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

        for row in json_list:
            row = json.loads(row)
            label_str = " " + row["output"]
            label = hypotheses.index(label_str)
            summary = row["input"]
            premise = f'{summary}{prompt}'
            options = []
            for h in label_list:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym,
                                   'label_list': label_list})
        print("examples: ", examples[0]['options'][0]['premise'])
    return examples



def load_examples_cqa(path, return_tuple=False):
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            ## use the '?' as a bridge
            if not premise[-1] in '?.!':
                print(premise)
            else:
                premise = premise[:-1] ## trim the punctuation, will add a question mark

            if return_tuple:
                options = [ '? the answer is: "{}"'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [(premise,opt) for opt in options], 
                  'label':label}]
            else:
                options = [ '? {}'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [{'premise':premise + '? the answer is:' ,
                                          'hypothesis': ' "{}"'.format(c['text'].lower()),
                                           'uncond_premise': ' the answer is:',
                                           'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                          'label':label}]

    return examples


def load_examples_rte(path, args):
    label2synonym = {0: [' true', ' yes', ' accurate', ' correct', " faithful"], 1: [' false', ' no', ' incorrect', ' wrong', ' untrue', " unfaithful"]}
    # label2synonym = {0: [' true'], 1: [' false']}
    hypotheses = [' true', ' false']
    prompt = ' true or false?\n answer:'

    train_data = []
    train_path = path.replace("dev", "train")
    with open(train_path) as f:
        for line in f:
            train_data += [json.loads(line)]

    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        for d in train_data:
            premise = f" {d['premise']}\n question: {d['hypothesis']} true or false?\n answer:"
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = ' true or false?\n answer:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = 0 if d['label'] == 'entailment' else 1
            train_examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" {d['premise']}\n question: {d['hypothesis']} {prompt}"
        options = []
        for h in hypotheses:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = prompt
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 0 if d['label'] == 'entailment' else 1
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples

def load_examples_cb(path, args):
    data = []
    label2synonym = {0: [' true', ' yes', ' accurate', ' correct', " faithful"], 1: [' false', ' no', ' incorrect', ' wrong', ' untrue', " unfaithful"], 2: [' neither']}
    # label2synonym = {0: [' true'], 1: [' false'], 2: [' neither']}
    hypotheses = [' true', ' false', ' neither']
    prompt = " true, false, or neither?\n answer:"

    train_data = []
    train_path = path.replace("dev", "train")
    train_path = path.replace("test", "train")
    with open(train_path) as f:
        for line in f:
            train_data += [json.loads(line)]

    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        for d in train_data:
            premise = f" question: Given that \"{d['premise']}\" Is \"{d['hypothesis']}\" true, false, or neither?\n answer:"
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = ' the answer is:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = ["entailment", 'contradiction', 'neutral'].index(d['label'])
            train_examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" question: Given that \"{d['premise']}\" Is \"{d['hypothesis']}\" true, false, or neither?\n answer:"
        options = []
        for h in hypotheses:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' the answer is:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = ["entailment",'contradiction','neutral'].index(d['label'])
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print(examples[0])
    return examples


def load_examples_sst2(path, args):
    # hypotheses = [' bad', ' good']
    label_list = [' terrible', ' great']
    label_path = "./task_data/sst2/label_names_sentidict.txt"

    label2synonym = load_label(label_path)
    prompt = " It was"

    '''
    load train examples
    '''
    train_data = []
    train_path = path.replace("dev", "train")
    with open(train_path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            d = {}
            d['correct_hypothesis'] = 1 if label > 0 else 0
            d['sentence'] = s
            train_data.append(d)

    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        for d in train_data:
            # demonstration = "It 's not a great monster movie. My overall feeling was that the movie was great. \n A party-hearty teen flick that scalds like acid. My overall feeling was that the movie was bad.\n or its 100 minutes running time , you 'll wait in vain for a movie to happen . My overall feeling was that the movie was great. \n It 's a road-trip drama with too many wrong turns . My overall feeling was that the movie was bad."
            premise = f"{d['sentence']}{prompt}"
            options = []
            for h in label_list:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = f'{prompt}'  # The quote has a tone that is
                o['uncond_hypothesis'] = h
                options.append(o)
            label = d['correct_hypothesis']
            train_examples.append(
                {'options': options, 'label': label, 'label2synonym': label2synonym, 'label_list': label_list})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    data = []
    with open(path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            d = {}
            d['correct_hypothesis'] = 1 if label > 0 else 0
            d['sentence'] = s
            data.append(d)


    examples = []
    for d in data:
        # demonstration = "It 's not a great monster movie. My overall feeling was that the movie was great. \n A party-hearty teen flick that scalds like acid. My overall feeling was that the movie was bad.\n or its 100 minutes running time , you 'll wait in vain for a movie to happen . My overall feeling was that the movie was great. \n It 's a road-trip drama with too many wrong turns . My overall feeling was that the movie was bad."
        premise = f"{d['sentence']}{prompt}"
        options = []
        for h in label_list:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = f'{prompt}' # The quote has a tone that is
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples


def get_sst2_variant_template(variant):
    if variant == 0:
        premise_template = ' Review: {sentence}\n Answer:' 
        uncond_premise = ' Positive or Negative?\n Answer:' 
        hypotheses = [' Negative', ' Positive']
    elif variant == 1:
        premise_template = ' Review: {sentence}\n Answer:' 
        uncond_premise = ' Was the film good or bad?\n Answer:' 
        hypotheses = [' bad', ' good']
    elif variant == 2:
        premise_template = ' My review for last night\'s film: {sentence} The critics agreed that this movie was' 
        uncond_premise = ' The critics agreed that this movie was' 
        hypotheses = [' bad', ' good']
    elif variant == 3:
        premise_template = ' Here is what our critics think for this month\'s films.\n One of our critics wrote "{sentence}". Her sentiment towards the film was'
        uncond_premise = ' Her sentiment towards the film was'
        hypotheses = [' negative.', ' positive.']
    elif variant == 4:
        premise_template = ' Critical reception [ edit ]\n In a contemporary review, Roger Ebert wrote "{sentence}". Entertainment Weekly agreed, and the overall critical reception of the film was'
        uncond_premise = '  Entertainment Weekly agreed, and the overall critical reception of the film was'
        hypotheses = [' bad.', ' good.']
    elif variant == 5:
        premise_template = ' Review: {sentence}\n Positive Review?'
        uncond_premise = ' Is this a Positive Review?' 
        hypotheses = [' No', ' Yes']
    elif variant == 6:
        premise_template = ' Review: {sentence}\n Question: Is the sentiment of the above review Positive or Negative?\n Answer:'
        uncond_premise = ' Positive or Negative?\n Answer:' 
        hypotheses = [' Negative', ' Positive']
    elif variant == 7:
        premise_template = ' Review: {sentence}\n Question: Did the author think that the movie was good or bad?\n Answer:'
        uncond_premise = 'the movie was good or bad?\n Answer:'
        hypotheses = [' bad', ' good']
    elif variant == 8:
        premise_template = ' Question: Did the author of the following tweet think that the movie was good or bad?\n Tweet: {sentence}\n Answer:'
        uncond_premise =  ' Was the movie was good or bad?\n Tweet: <redacted>\n Answer:'
        hypotheses = [' bad', ' good']
    elif variant == 9:
        premise_template = ' {sentence} My overall feeling was that the movie was'
        uncond_premise =  '  My overall feeling was that the movie was'
        hypotheses = [' bad', ' good']
    elif variant == 10:
        premise_template = ' {sentence} I' 
        uncond_premise =  ' After watching the movie, I decided I'
        hypotheses = [' hated', ' liked']
    elif variant == 11:
        premise_template = ' {sentence} My friend asked me if I would give the movie 0 or 5 stars, I said' 
        uncond_premise =  ' My friend asked me if I would give the movie 0 or 5 stars, I said'
        hypotheses = [' 0', ' 5']
    elif variant == 12:
        premise_template = ' Input: {sentence}\n Sentiment:'
        uncond_premise =  ' Analyze the sentiment of the previous statement.\n Sentiment:'
        hypotheses = [' Negative', ' Positive']
    elif variant == 13:
        premise_template = ' Review: {sentence}\n Positive:'
        uncond_premise =  ' Positive:'
        hypotheses = [' False', ' True']
    elif variant == 14:
        premise_template = ' Review: {sentence} Stars:'
        uncond_premise =  ' How many stars would you give this movie:' 
        hypotheses = [' 0', ' 5']
    else:
        raise NotImplementedError

    return premise_template, uncond_premise, hypotheses

def load_examples_sst2_variants(path, variant):
    premise_template, uncond_premise, hypotheses = get_sst2_variant_template(variant)

    data = []
    with open(path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            d = {}
            d['correct_hypothesis'] = 1 if label > 0 else 0
            # print(d['correct_hypothesis'])
            d['sentence'] = s
            data.append(d)

    examples = []
    for d in data:
        premise = premise_template.format(sentence=d['sentence'])
        options = []
        for h in hypotheses:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = uncond_premise
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label })
    return examples


def load_examples_hyp(path, args):
    hypotheses = [' neutral', ' partisan']
    label2synonym = {0: [' neutral', ' fair', ' objective'], 1: [' partisan', ' biased', ' unfair']}
    # label2synonym = {0: [' neutral'], 1: [' partisan', ' biased', ' unfair']}

    prompt = '\n neutral or partisan? Answer:'

    '''
    load train examples
    '''
    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        train_path = path.replace("dev", "train")
        with open(train_path) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label = int(row['label'])
                summary = row['text'].strip()
                premise = f"{summary}{prompt}"
                options = []
                for h in hypotheses:
                    o = {}
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = prompt
                    o['uncond_hypothesis'] = h
                    options.append(o)
                similarity = float(row["similarity"]) if "similarity" in row else None
                train_examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)

    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text'].strip()
            premise = f"{summary}{prompt}"
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples


def load_examples_amazon(path):
    hypotheses = ['useless', 'useful']
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text']
            premise = f' Review: {summary}\n Question: Is the above review useless or useful?\n Answer:'
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = ' ' + h.lower()
                o['uncond_premise'] = ' useless or useful?\n Answer:'
                o['uncond_hypothesis'] = ' ' + h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity})
    return examples

def load_examples_imdb(path):
    hypotheses = [' terrible', ' great']
    # label2synonym = {0: [' hated', " hate", " disliked", 'hate'], 1: [' liked', " like", " love"]}
    # label2synonym = {0: [' awful', ' bad', ' stupid', ' worse', ' worst', ' mediocre', ' pointless', ' forgotten',  ' terrible', ' boring', ' awful', " poor", " horrible", " misleading"], 1: [' recommended', ' enjoyable', ' awesome', ' engaging', ' better', ' great', ' good',  ' nice', ' exciting', ' excellent']}
    # label2synonym = {0: [' bad'], 1: [' good']}
    label_path = "./task_data/sst2/label_names_sentidict.txt"
    label2synonym = load_label(label_path)
    # label2synonym = {0: [' bad'], 1: [' good']}

    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text']
            premise = f' {summary}. My overall feeling was that the movie was'
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = '. My overall feeling was that the movie was'
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
    return examples


def load_examples_cr(path, args):
    hypotheses = [' negative', ' positive']
    label_list = [' terrible', ' great']

    label_path = "./task_data/sst2/label_names_sentidict.txt"

    label2synonym = load_label(label_path)
    prompt = " It was"

    '''
    load train examples
    '''
    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        train_path = path.replace("dev", "train")
        with open(train_path) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label = int(row['label'])
                summary = row['text']
                premise = f'{summary}{prompt}'
                options = []
                for h in label_list:
                    o = {}
                    o['premise'] = premise
                    o['hypothesis'] = h.lower()
                    o['uncond_premise'] = prompt
                    o['uncond_hypothesis'] = h.lower()
                    options.append(o)
                similarity = float(row["similarity"]) if "similarity" in row else None
                train_examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym,
                                 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)

    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text']
            premise = f'{summary}{prompt}'
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples

def construct_icl_examples(examples, k):
    np.random.seed(88)
    demo_examples = np.random.choice(examples, size=k)
    icl_str = ""
    for ex in demo_examples:
        icl_str += f"{ex['options'][0]['premise']}{ex['label_list'][ex['label']]}\n"
    # print("icl_str:", icl_str)
    return icl_str


def load_examples_mr(path, args):
    # hypotheses = [' terrible', ' great']
    hypotheses = [' terrible', ' great']

    label_path = "./task_data/sst2/label_names_sentidict.txt"
    label2synonym = load_label(label_path)

    prompt = " It was" # My overall feeling was that the movie was

    '''
    load train examples
    '''
    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        train_path = path.replace("dev", "train")
        with open(train_path) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label = int(row['label'])
                summary = row['text']
                premise = f'{summary}{prompt}'
                # print("premise")
                options = []
                for h in hypotheses:
                    o = {}
                    o['premise'] = premise
                    o['hypothesis'] = h.lower()
                    o['uncond_premise'] = prompt
                    o['uncond_hypothesis'] = h.lower()
                    options.append(o)
                similarity = float(row["similarity"]) if "similarity" in row else None
                train_examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    examples = []
    # print(prompt, hypotheses)
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader):
            label = int(row['label'])
            summary = row['text']
            premise = f'{summary}{prompt}'
            # print("premise")
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples



# def load_examples_agn(path):
#     topics = [' World', ' Sports', ' Business', ' Science']
#     examples = []
#     with open(path) as fp:
#         reader = csv.DictReader(fp)
#         for row in reader:
#             label = int(row['label'])
#             # title = row['Title']
#             summary = row['text']
#             premise = f"{summary}\n topic:"
#             options = []
#             for h in topics:
#                 o = {}
#                 o['premise'] = premise
#                 o['hypothesis'] = h.lower()
#                 o['uncond_premise'] = '\n topic:'
#                 o['uncond_hypothesis'] = h.lower()
#                 options.append(o)
#             label = label
#             similarity = float(row["similarity"]) if "similarity" in row else None
#             examples.append({'options': options, 'label': label, "sim": similarity})
#     return examples


def load_examples_yahoo(path, args):
    topics = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
    label_list = [" society", " science", " health", " education", " computer", " sports", " business", " entertainment", " family", " politics"]

    label2synonym = {}
    label_path = "./task_data/yahoo/label.json"
    with open(label_path, "r") as f:
        topic2synonym = json.load(f)

    for i, (k, v) in enumerate(topic2synonym.items()):
        label2synonym[topics.index(k)] = [" "+item for item in v]

    # label2synonym = {k: [v] for k, v in enumerate(label_list)}
    dataset = load_dataset("yahoo_answers_topics")

    # prompt = " The text topic is about"
    prompt = " topic:"

    '''
    load train examples
    '''
    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        for row in dataset["train"].shuffle():
            label = row["topic"]
            title = row['question_title']
            summary = row['question_content']
            answer = row['best_answer']
            premise = f"title: {title} content: {summary} answer: {answer}{prompt}"
            options = []
            for h in label_list:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            label = label
            train_examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    examples = []
    for row in dataset["test"].shuffle():
        label = row["topic"]
        title = row['question_title']
        summary = row['question_content']
        answer = row['best_answer']
        # premise = f"title: {title} content: {summary}{prompt}"
        premise = f"title: {title} content: {summary} answer: {answer}{prompt}"
        options = []
        for h in label_list:
            o = {}
            o['premise'] = icl_str + premise
            # print(premise)
            # print("------")
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = prompt
            o['uncond_hypothesis'] = h
            options.append(o)
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples



def load_examples_agn(path, args):
    # topics = [' politics', ' sports', ' business', ' technology']
    topics = [' world', ' sports', ' business', ' science']
    label_path = "./task_data/agn/label_names_kb.txt"
    # label_path = "/gscratch/zlab/swj0419/knnlm/data/label_word/datasets/agnews/label_names.txt"
    label2synonym = load_label(label_path)
    # prompt = " The text topic is about"
    prompt = " topic:"
    '''
    load train examples
    '''
    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        train_path = path.replace("dev", "train")
        with open(train_path) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label = int(row['Class Index'])-1
                title = row['Title']
                summary = row['Description']
                premise = f"{title}\n {summary}{prompt}"
                options = []
                for h in topics:
                    o = {}
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = prompt
                    o['uncond_hypothesis'] = h
                    options.append(o)
                label = label
                train_examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': topics})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['Class Index'])-1
            title = row['Title']
            summary = row['Description']
            premise = f"{title} \n {summary}{prompt}"
            options = []
            for h in topics:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            label = label
            examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': topics})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples




def proc_passage(s):
    s = s.replace("``", '"')
    s = s.replace("''", '"')
    return s



def proc_question(s):
    s = s[0].upper() + s[1:]
    s = s.replace(' i ', ' I ')
    s = s + '?'
    return s

def load_examples_boolq(path, args):
    label2synonym = {0: [' true', ' yes', ' accurate', ' correct', ' faithful'], 1: [' false', ' no', ' incorrect', ' wrong', ' untrue', ' unfaithful']}
    hypotheses = [' yes', ' no']
    prompt = " true or false?\n answer:"

    train_data = []
    train_path = path
    with open(train_path) as f:
        for line in f:
            train_data += [json.loads(line)]

    icl_str = ""
    if args.k_shot > 0:
        train_examples = []
        for d in train_data:
            p = f' title: {d["title"]}\n question: {proc_question(d["question"])}{prompt}'
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = p
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            label = 1 if not d['answer'] else 0
            train_examples.append(
                {'options': options, 'label': label, 'label2synonym': label2synonym, 'label_list': hypotheses})
        icl_str = construct_icl_examples(train_examples, k=args.k_shot)


    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        options = []
        p = f' title: {d["title"]}\n question: {proc_question(d["question"])}{prompt}'
        for h in hypotheses:
            o = {}
            o['premise'] = icl_str + p
            o['knn_premise'] = p
            o['hypothesis'] = h
            o['uncond_premise'] = prompt
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 1 if not d['answer'] else 0
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
    print("examples: ", examples[0]['options'][0]['premise'])
    return examples

def load_examples_trec(path):
    label2template = [(0, 'DESC', ' description'),
                      (1, 'ENTY', ' entity'),
                      (2, 'LOC', ' location'),
                      (3, 'NUM', ' number'),
                      (4, 'ABBR', ' acronym'),
                      (5, 'HUM', ' person')]
    label2synonym = {i[0]: [i[2]] for i in label2template}
    hypotheses = [i[2] for i in label2template]



    # get index of the label string
    examples = []
    
    # params
    with open(path) as f:
        for line in f:
            label = line[:line.index(' ')].split(':')[0]
            question = detokenizer(line[line.index(' ') + 1:]).strip()

            ex = {}
            options = []
            for label_idx, label_surface_form, h in label2template:
                opt = {}
                opt['premise'] = f' {question} The answer to this question will be'
                opt['knn_premise'] = f' {question} The answer to this question will be'
                opt['hypothesis'] = f'{h}'
                opt['uncond_premise'] = ' The answer to this question will be'
                opt['uncond_hypothesis'] = f'{h}'
                options.append(opt)
                if label_surface_form == label:
                    ex['label'] = label_idx
            ex['options'] = options
            ex['label2synonym'] = label2synonym
            ex['label_list'] = hypotheses

            examples.append(ex)

    return examples

