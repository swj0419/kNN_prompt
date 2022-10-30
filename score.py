import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import time
import numpy as np
# import statistics
import os
from collections import defaultdict
from utils import get_key, gpt3, cross_entropy_list_gpt3, log_softmax
from pdb import set_trace as bp
from scipy.special import softmax
from IPython import embed
from ipdb import set_trace as bp

Acc = []
'''
test demo
'''
all_scores = []

all_scores_if_zero = [] # [[0, 0, 1], [1, 0, 1], ...]

class EvaluatingWrapper():
    def __init__(self, model, encoder, knn_model, knn_tokenizer, examples, knn_dstore, args):
        self.model = model
        self.encoder = encoder
        self.vocab = encoder.vocab_size
        self.knn_model = knn_model
        self.knn_tokenizer = knn_tokenizer
        self.examples = examples
        self.label2synonym = examples[0]["label2synonym"]
        self.label2synonym_id = self.init_label2word_id(self.label2synonym)

        self.labels = examples[0]["label_list"]
        self.labels_id = self.encoder(self.labels)["input_ids"]

        self.label2word = {i: [v] for i, v in enumerate(examples[0]["label_list"])}
        self.label2word_id = self.init_label2word_id(self.label2word)
        print("label2word: ", self.label2word)

        self.knn_dstore = knn_dstore
        self.args = args
        self.hist_path = None
        self.max_len = 0
        self.lambdas = [0, 0.3]


    def init_label2word_id(self, label2synonym):
        label2synonym_id = {}
        for k, v in label2synonym.items():
            synonym_id = []
            for word in v:
                if len(self.encoder(word)["input_ids"]) == 1: # check single word
                    synonym_id.append(self.encoder(word)["input_ids"]) # change later
            label2synonym_id[k] = torch.LongTensor(synonym_id).cuda()
        return label2synonym_id

    def save_score(self, klambda2result, klambda2predictions_list, scoring_func):
        # save scores
        results_path = f'{self.args.output_dir}/{scoring_func}_{self.args.split}.accs'
        with open(results_path, 'w') as out:
            out.write(json.dumps(klambda2result))

        # save predicted labels
        preds_path = f'{self.args.output_dir}/{scoring_func}_{self.args.split}.preds'
        with open(preds_path, 'w') as out:
            out.write(json.dumps(klambda2predictions_list))

    def print_overview(self):
        # print the first example to make sure the format is ok
        print('=' * 50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(self.examples)))
        ex = self.examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(self.encoder.decode(self.encoder.encode(opt['premise'])) + '<BREAK>' + self.encoder.decode(
            self.encoder.encode(opt['hypothesis'])))
        print('UNCONDITIONAL:')
        print(self.encoder.decode(self.encoder.encode(opt['uncond_premise'])) + '<BREAK>' + self.encoder.decode(
            self.encoder.encode(opt['uncond_hypothesis'])))
        print('=' * 50)


    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, knn_lambda):
        # bp()
        if self.args.scoring.startswith("log_softmax"):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - knn_lambda)
            coeffs[1] = np.log(knn_lambda)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
        else:
            curr_prob = knn_lambda*knn_p + (1-knn_lambda)*vocab_p
        return curr_prob

    def logmeanexp(self, x, dim=0):
        mmax, _ = torch.max(x, dim=dim, keepdim=True)
        return (torch.squeeze(mmax, dim=dim) +
                torch.log(torch.mean(torch.exp((x - mmax)), dim=dim)))
        # return (torch.squeeze(mmax, dim=dim) +
        #         torch.mean(torch.exp((x - mmax)), dim=dim))

    def get_knn_scores(self, outputs):
        def dist_func(d, k, q):
            qsize = q.shape
            if self.args.sim_func == 'l2':
                knns_vecs = torch.from_numpy(self.knn_dstore.keys[k]).cuda().view(qsize[0], self.args.k, -1)
                # bp()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.args.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2  # negative distance, higher better
            elif self.args.sim_func == 'do_not_recomp_l2':
                return -1 * d
            elif self.args.sim_func == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.knn_dstore.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        queries = outputs['hidden_states'][-1][:, -1, :]
        dists, knns = self.knn_dstore.get_knns(queries)  # smaller --> better

        knn_ids = self.knn_dstore.vals[knns]

        dists = torch.from_numpy(dists).cuda()
        dists = dist_func(dists, knns, queries)

        knn_ids = knn_ids[0]
        probs = torch.softmax(dists / self.args.knn_temp, dim=-1)
        probs = probs.detach().cpu().numpy()

        # bp()
        # assert len(knn_ids[0])==1 and probs.shape[0]==1 and len(knn_ids[0])==len(probs[0])
        probs = probs.squeeze()
        knn_ids = knn_ids.squeeze()
        full_knn_scores = defaultdict(int)
        for vocab, p in zip(knn_ids, probs):
            full_knn_scores[vocab.item()] += p

        # extract score for synonyms
        # print("full_knn_scores: ", full_knn_scores)

        # swj: apply softmax
        label2knn_prob_np = np.zeros((self.vocab, ))
        for label, prob in full_knn_scores.items(): 
            label2knn_prob_np[label] = prob
        # label2knn_prob_np = softmax(label2knn_prob_np)
        return label2knn_prob_np

    def cal_result(self, cond_ce, options, uncond_ce, domain_cond_ce):
        '''son
        prediction
        '''
        ## get average CE by token
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]

        # calculate dcpmi
        dcpmi = [ce_0/ce_1 for ce_0, ce_1 in zip(domain_cond_ce, cond_ce)]
        pmi = [ce_0/ce_1 for ce_0, ce_1 in zip(uncond_ce, cond_ce)]

        ## make predictions based on different scores
        lm_pred = cond_ce.index(min(cond_ce))
        lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
        lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
        dcpmi_pred = dcpmi.index(max(dcpmi))
        pmi_pred = pmi.index(max(pmi))
        pred = {
            'lm': lm_pred,
            'tok_mean': lm_avg_pred,
            'dcpmi': dcpmi_pred,
            'pmi': pmi_pred,
            'domain_cond': lm_domain_cond_pred,
        }
        return pred


    def cal_score(self, labels, predictions_list):
        # get predictions into list by scoring key
        predictions_dict = {key: list(map(lambda v: v[key], predictions_list)) for key in
                            predictions_list[0].keys()}

        # calculate accuracies
        results = {key: round((sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key], labels)))) / len(labels)) * 100, 2) for key in predictions_dict.keys()}

        # save labels for later
        predictions_dict['labels'] = labels
        return results, predictions_dict

    def compute_LM_prob4tokens(self, outputs):
        # logits = outputs.logits[:, :-1].contiguous()
        logits = outputs.logits[:, :].contiguous()
        if self.args.scoring.startswith("logsoftmax_"):
            last_token_softmax = torch.log_softmax(logits[:, -1, :], dim=-1)
        else:
            last_token_softmax = torch.softmax(logits[:, -1, :], dim=-1)
        # label2LM_prob = {}
        last_token_softmax = last_token_softmax.squeeze()
        # bp()
        # for label, s_id in self.label2word_id.items():  # swj: label2word_id,  self.label2synonym_id
        #     s_id = s_id.squeeze()
        #     s_index2prob = last_token_softmax[s_id]
        #     label2LM_prob[label] = s_index2prob
        return last_token_softmax


    def eval_one_ex(self, input_texts, knn_input_texts):
        inputs = self.encoder.encode_plus(input_texts, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        knn_inputs = self.encoder.encode_plus(knn_input_texts, return_tensors="pt").to("cuda")
        knn_input_ids = knn_inputs["input_ids"]

        if len(input_ids[0]) > 1024:
            input_ids = input_ids[0][-1024:]
            input_ids = input_ids.unsqueeze(0)

        if len(knn_input_ids[0]) > 1024:
            knn_input_ids = knn_input_ids[0][-1024:]
            knn_input_ids = knn_input_ids.unsqueeze(0)


        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            label2LM_prob = self.compute_LM_prob4tokens(outputs) # vocab, 1
            label2LM_prob = label2LM_prob.cpu().numpy()
            knn_outputs = self.knn_model(knn_input_ids, output_hidden_states=True)
            label2knn_prob = self.get_knn_scores(knn_outputs)  # vocab, 1
            # embed()
        return label2LM_prob, label2knn_prob
 

    def compute_acc(self, klambda2all_pred, all_label):
        klambda2all_acc = {}
        for klambda, all_pred in klambda2all_pred.items():
            klambda2all_acc[klambda] = round(sum(1 for x, y in zip(all_label, all_pred) if x==y)/len(all_label), 4)
            # swj: error analysis
            # if klambda == 0:
            #     for i, (x, y) in enumerate(zip(all_label, all_pred)):
            #         if x != y:
            #             text = self.examples[i]["options"][0]["premise"]
            #             print(f"{text} label: {x} pred: {y}")
        return klambda2all_acc

    def vocab2label(self, final_prob, label2synonym_id):
        label2knn_prob = np.zeros((len(label2synonym_id), ))
        for label, s_ids in label2synonym_id.items():
            s_ids = s_ids.squeeze(dim=-1)
            for s_id in s_ids:
                prob = final_prob[s_id.item()]
                label2knn_prob[label] += prob
        label2knn_prob = torch.FloatTensor(label2knn_prob)
        return label2knn_prob


    def score(self):
        def apply_pmi(label2LM_prob, label2knn_prob, domain_label2LM_prob, domain_label2knn_prob):
            '''
            interpolate token level (both knn and lm are same) -> calibration token level -> compute label prob
            '''
            # LM score
            for klambda in self.lambdas: # lambdas=[0, 0.3]
                # final_prob: 500x1, after interpolation
                final_prob = self.combine_knn_and_vocab_probs(label2knn_prob, label2LM_prob, klambda)
                final_prob_domain = self.combine_knn_and_vocab_probs(domain_label2knn_prob, domain_label2LM_prob, klambda)
                # bp()
                ## standard
                # LM
                label2prob = self.vocab2label(final_prob, self.label2word_id)
                klambda2all_pred_standard[klambda].append(torch.argmax(label2prob).item())

                # PMI
                final_prob_pmi = np.log(final_prob+1e-10) - np.log(final_prob_domain+1e-10)
                label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2word_id)
                klambda2all_pred_pmi_standard[klambda].append(torch.argmax(label2prob_pmi).item())

                # fuzzy
                # LM score
                label2prob = self.vocab2label(final_prob, self.label2synonym_id)
                klambda2all_pred[klambda].append(torch.argmax(label2prob).item())

                # PMI
                # final_prob_pmi = final_prob/final_prob_domain
                label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2synonym_id)
                klambda2all_pred_pmi[klambda].append(torch.argmax(label2prob_pmi).item())
        
        klambda2all_pred_standard = defaultdict(list) #{0.1: [1,2,3], }
        klambda2all_pred_pmi_standard = defaultdict(list) #{0.1: [1,2,3], }
        klambda2all_pred = defaultdict(list) #{0.1: [1,2,3], }
        klambda2all_pred_pmi = defaultdict(list) #{0.1: [1,2,3], }

        all_label = []

        # compute domain prior
        domain_text = self.examples[0]["options"][0]["uncond_premise"]
        domain_label2LM_prob, domain_label2knn_prob = self.eval_one_ex(domain_text, domain_text)

        for ex in tqdm(self.examples):
            # if ex["options"][0]["premise"] != "sometimes , it needs few minutes to open an application . It was":
            #     continue
            all_label.append(ex["label"])
            input_text = ex["options"][0]["premise"]
            knn_input_text = ex["options"][0]["knn_premise"]
            label2LM_prob, label2knn_prob = self.eval_one_ex(input_text, knn_input_text) # {0.1: [0.8,0.2], 0.2: []}
            apply_pmi(label2LM_prob, label2knn_prob, domain_label2LM_prob, domain_label2knn_prob)



        print("=============standard========================")
        klambda2all_acc = self.compute_acc(klambda2all_pred_standard, all_label)
        klambda2all_acc_pmi = self.compute_acc(klambda2all_pred_pmi_standard, all_label)
        print(f"standard LM (lm/pmi) acc: {klambda2all_acc[0]}/{klambda2all_acc_pmi[0]}")

        # LM acc
        print("=============kNN-LM========================")
        klambda2all_acc = self.compute_acc(klambda2all_pred, all_label)
        # PMI acc
        klambda2all_acc_pmi = self.compute_acc(klambda2all_pred_pmi, all_label)
        self.save_score(klambda2all_acc_pmi, klambda2all_pred, "pmi")
        print(f"kNN-LM with interpolation lambda 0.3 (lm/pmi) acc: {klambda2all_acc[0.3]}/{klambda2all_acc_pmi[0.3]}")