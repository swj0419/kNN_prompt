import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import sys
import openai
import time
import os


def log_softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)



def get_key(source, target):
    return '{}'.format(json.dumps({'source': source, 'target': target}))


def gpt3(prompt, max_len, model_name, temp=0, num_log_probs=100, echo=False, n=None):
    print('calling API')
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo,
                                                stop='\n',
                                                n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(1)
    return response

def cross_entropy_list_gpt3(inputs, targets, model_name, batch=None, cache=None, calculate=False):
    '''
    get a list of -log P(target|inp) for
    the inputs and targets in inputs, targets
    using gpt3
    '''
    assert (len(inputs) == len(targets))

    ### This block at the top handles caching/batching
    ## basically, first log all computations not in the cache
    ## if calculate is False, return dummy values (just
    ## logging computations to do later)
    ## if calculate is True, do all computations that are not done
    ## then return results for this batch
    ###############################
    ## if we are caching results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all needed
    # calculations to the batch with calculate = False
    # then run with calculate=True to work through all cached calculations
    if cache is not None:
        # log calculations we have not done yet
        for inp, targ in zip(inputs, targets):
            if get_key(inp, targ) not in cache:
                cache[get_key(inp, targ)] = {'source': inp, 'target': targ, 'result': None}

        # if not calculating, return dummy values
        if not calculate:
            return [1.] * len(inputs), [1.] * len(inputs), None

        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]

        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]

            ce_list, t_len_list, result_list = cross_entropy_list_gpt3(sources_todo, targets_todo, model_name,
                                                                       cache=None, batch=batch)
            for source, target, ce, t_len, result in zip(sources_todo, targets_todo, ce_list, t_len_list, result_list):
                cache[get_key(source, target)]['ce'] = ce
                cache[get_key(source, target)]['result'] = result
                cache[get_key(source, target)]['t_len'] = t_len
        ## return results for thie example
        output = ([cache[get_key(inp, targ)]['ce'] for inp, targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['t_len'] for inp, targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['result'] for inp, targ in zip(inputs, targets)])
        return output
    ###############################

    ### batching ####
    if batch is not None:
        result = {'choices': []}
        ce_list = []
        len_list = []
        while len(inputs) > 0:
            ce_out, len_out, result_out = cross_entropy_list_gpt3(inputs[:batch], targets[:batch], model_name,
                                                                  cache=None, batch=None)
            inputs, targets = inputs[batch:], targets[batch:]

            ce_list = ce_list + ce_out
            len_list = len_list + len_out
            result['choices'] = result['choices'] + result_out

            return ce_list, len_list, result['choices']
            #########

    #####
    ## calculating cross-entropy
    #####
    data = [inp + targ for inp, targ in zip(inputs, targets)]
    result = gpt3(data, 0, model_name, echo=True, num_log_probs=1)

    # with open(out_file, 'a') as out:
    #    out.write(f'{json.dumps(result)}\n')
    ce_list = []
    t_lens = []
    for inp, out in zip(inputs, result['choices']):
        # get the beginning of the target from the response (based on tokenization)
        i = 0
        while out['logprobs']['text_offset'][i] < len(inp):
            i += 1
        t_lens.append(len(out['logprobs']['text_offset']) - i)
        # sum of log probs over the target tokens
        ce = -sum(out['logprobs']["token_logprobs"][i:])
        ce_list.append(ce)
    return ce_list, t_lens, result['choices']

