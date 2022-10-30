import random
import sys
sys.path.append("/gscratch/zlab/swj0419/knnlm/src/knnlm_gpt/data")
import numpy as np
import torch
import logging
from data_loaders_final import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_examples(dataset_name, split, stem, n_shot, variant, args):
    if dataset_name == 'copa':
        examples = load_examples_copa(f'{stem}/copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'copa-rev':
        examples = load_examples_copa_rev(f'{stem}/copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'storycloze':
        examples = load_examples_storycloze(f'{stem}/{split}.tsv')
        closed_label_space = False
    elif dataset_name == 'hellaswag':
        examples = load_examples_hellaswag(f'{stem}/dev.jsonl')
        closed_label_space = False
    elif dataset_name == 'race-m' or \
         dataset_name == 'race-h':
        version = 'high' if dataset_name == 'race-h' else 'middle'
        examples = load_examples_race(stem, split, version)
        closed_label_space = False
    elif dataset_name == 'arc-easy' or \
         dataset_name == 'arc-challenge':
        examples = load_examples_arc(f'{stem}/{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'obqa':
        examples = load_examples_obqa(f'{stem}/{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'cqa':
        if args.split == 'test':
            raise NotImplementedError("CSQA does not release test answers directly, please do not spam their leaderboard either :)")
        else:
            examples = load_examples_cqa(f'{stem}/{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'boolq':
        examples = load_examples_boolq(f'{stem}/dev.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'rte':
        examples = load_examples_rte(f'{stem}/val.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'cb':
        examples = load_examples_cb(f'{stem}/dev.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'sst2':
        examples = load_examples_sst2(f'{stem}/{split}.tsv', args)
        closed_label_space = True
    elif dataset_name == 'sst-5':
        examples = load_examples_sst5(f'{stem}/{split}.tsv')
        closed_label_space = True
    elif dataset_name == 'agn':
        print("test data: ", f'{stem}/{split}.csv')
        examples = load_examples_agn(f'{stem}/{split}.csv', args)
        closed_label_space = True
    elif dataset_name == 'dbpedia':
        examples = load_examples_dbpedia(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'yahoo':
        examples = load_examples_yahoo(None, args)
        closed_label_space = True
    elif dataset_name == 'yelp':
        examples = load_examples_yelp(f'{stem}/{split}.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'yelp_full':
        examples = load_examples_yelp_full(f'{stem}/{split}.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'rotten_tomatoes':
        examples = load_examples_rotten_tomatoes(f'{stem}/{split}.jsonl', args)
        closed_label_space = True

    elif dataset_name == 'financial_phrasebank':
        examples = load_examples_fb(f'{stem}/{split}.jsonl', args)
        closed_label_space = True
    elif dataset_name == 'hyp':
        examples = load_examples_hyp(f'{stem}/{split}.csv', args)
        closed_label_space = True
    elif dataset_name == 'imdb':
        examples = load_examples_imdb(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'cr':
        examples = load_examples_cr(f'{stem}/{split}.csv', args)
        closed_label_space = True
    elif dataset_name == 'mr':
        examples = load_examples_mr(f'{stem}/{split}.csv', args)
        closed_label_space = True
    elif dataset_name == 'amazon':
        examples = load_examples_amazon(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'trec':
        split = 'train' if split == 'dev' else split
        examples = load_examples_trec(f'{stem}/{split}.txt')
        closed_label_space = True
    elif dataset_name == 'lama':
        examples = load_examples_lama(f'{stem}/{split}.jsonl', args)
        closed_label_space = False
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return examples, closed_label_space



def load_test_data(args):
    examples, closed_label_space = get_examples(args.dataset_name, args.split, args.dataset_dir, 0, args.variant, args)
    random.seed(0)
    if args.n_sample:
        if args.n_sample > len(examples):
            args.n_sample = len(examples)
        random.shuffle(examples)
        index_value = random.sample(list(range(0, len(examples))), args.n_sample)
        examples_sample = []
        for i in index_value:
            examples_sample.append(examples[i])
        examples = examples_sample
    return examples, closed_label_space








