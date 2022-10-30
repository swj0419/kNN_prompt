import numpy as np
import faiss
import time
import torch
from datasets import load_dataset, concatenate_datasets
import logging
from tqdm import tqdm
import glob
import re
import pickle

logger = logging.getLogger(__name__)


def keep_1_column(example):
    new_example = {"text": example["text"]}
    return new_example


def load_data(args):
    interval = 100/args.num_gpus
    print("args.column_names: ", args.column_names)
    dataset_list = []
    if args.raw_file.startswith("wikitext"):
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=f'train[{int(interval*args.subset_index)}%:{int(interval*args.subset_index+interval)}%]')
        dataset_list.append(dataset.map(keep_1_column, ""))
    elif args.raw_file.startswith("cc_news"):
        dataset = load_dataset("cc_news", split=f'train[{int(interval*args.subset_index)}%:{int(interval*args.subset_index+interval)}%]')
        dataset_list.append(dataset.map(keep_1_column))
    elif args.raw_file.startswith("wikipedia"):
        dataset = load_dataset('wikipedia', "20200501.en", split='train')
    elif args.raw_file.startswith("imdb"):
        dataset = load_dataset("imdb", split="train")
    elif "tapt" in args.raw_file:
        dataset = load_dataset(
            "csv", data_files=[args.raw_file], split=f"train", delimiter=",", column_names=["text"])
    else:
        dataset = load_dataset(
            "csv", data_files=[args.raw_file], split=f"train", delimiter=",", column_names=["title", "text"] # change to ["title", "text"] ["title", "text", "id", "label"] when for imdb training data
        ) # didn't handle over

    #
    # '''
    # amazon: 1.66G
    # imdb: 127.06MB
    # wikitext103: 522.23 MiB
    # cc_news: 1.88G
    # '''
    # dataset_list = []
    # print("args.raw_file: ", args.raw_file.split(","))
    # for file in args.raw_file.split(","):
    #     if file.startswith("wikitext"):
    #         dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    #
    #     elif file.startswith("cc_news"):
    #         dataset = load_dataset("cc_news", split="train[40%:60%]") # 100
    #         dataset = dataset.remove_columns( ['title', 'domain', 'date', 'description', 'url', 'image_url'])
    #
    #
    #     elif file.startswith("amazon"):
    #         dataset = load_dataset("amazon_polarity", split="train[0%:30%]") # 40
    #         dataset = dataset.rename_column("content", "text")
    #         dataset= dataset.remove_columns(['label', 'title'])
    #
    #     elif file.startswith("imdb"):
    #         dataset = load_dataset("imdb", split="train")
    #         dataset = dataset.remove_columns(['label'])
    #
    #     dataset = dataset.map(keep_1_column)
    #     dataset_list.append(dataset)
    #
    # # concat
    # print("dataset_list: ", dataset_list)
    # dataset = dataset_list[0]
    # for d in dataset_list[1:]:
    #     dataset = concatenate_datasets([dataset, d])
    # sample
    print("dataset: ", len(dataset))
    # remove short and empty example
    dataset = dataset.filter(lambda example: len(example["text"].split()) > 8)
    print("filtered dataset: ", len(dataset))
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    return dataloader

def save_dstore(args, model, tokenizer, dataloader):
    print("args.dstore_dir: ", args.dstore_dir)
    dstore_keys = np.memmap(f"{args.dstore_dir}/keys_{args.subset_index}.npy", dtype=np.float32, mode='w+',
                            shape=(args.dstore_size, args.dimension))
    dstore_vals = np.memmap(f"{args.dstore_dir}/vals_{args.subset_index}.npy", dtype=np.int32, mode='w+', shape=(args.dstore_size, 1))

    # key_texts_len = 50
    # dstore_key_texts = np.memmap(f"{args.dstore_dir}/key_texts_{args.subset_index}.npy", dtype=np.int32, mode='w+',
    #                         shape=(args.dstore_size, key_texts_len))
    dstore_idx = 0
    dstore_key_texts = []
    index2text_id = {}
    all_texts = []
    for batch in tqdm(iter(dataloader)):
        batch = batch["text"]
        inputs = tokenizer(batch, padding=True, return_length=True, return_tensors="pt", truncation=True).to("cuda")
        assert (inputs['length'] > 1).all()
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            output_hidden_states=True)
            # We pick the hidden state at the last layer as the key
            keys = outputs['hidden_states'][-1].cpu().numpy().astype(np.float32)
            vals = inputs["input_ids"].cpu().numpy().astype(np.float32)
            bsz, seq_len, dim = keys.shape

            for i in range(bsz):
                len_i = inputs['length'][i]
                # print(dstore_idx, len_i)
                # print("keys shape: ", keys.shape)
                # print("dstore_keys shape: ", len_i-1)
                dstore_keys[dstore_idx:(dstore_idx+len_i-1)] = keys[i, 0:(len_i-1)] # exclude last key
                dstore_vals[dstore_idx:(dstore_idx+len_i-1)] = np.expand_dims(vals[i, 1:len_i], axis=1) # exclude first value
                all_texts.append(batch[i])
                for tmp_i in range(dstore_idx, dstore_idx+len_i-1):
                    index2text_id[tmp_i] = len(all_texts)-1
                # dstore_key_texts[dstore_idx:(dstore_idx+len_i-1)] = inputs['input_ids'][i, :max(key_texts_len, len_i)]
                # print(dstore_vals[dstore_idx:(dstore_idx+len_i-1)])
                dstore_idx += len_i-1
            # dump size
            with open(f"{args.dstore_dir}/size_{args.subset_index}.txt", "w") as f:
                f.write(f"{dstore_idx}")

    print("dstore_idx", dstore_idx, "final shape", args.dimension)
    print("Keys", dstore_keys.shape, dstore_keys.dtype)
    print("Vals", dstore_vals.shape, dstore_vals.dtype)

    with open(f'{args.dstore_dir}/index2text_id.pk', 'wb') as f:
        pickle.dump(index2text_id, f)

    with open(f'{args.dstore_dir}/all_texts.pk', 'wb') as f:
        pickle.dump(all_texts, f)

    # dump size
    with open(f"{args.dstore_dir}/size_{args.subset_index}.txt", "w") as f:
        f.write(f"{dstore_idx}")

    with open(f"{args.dstore_dir}/size_{args.subset_index}.txt", "w") as f:
        f.write(f"{dstore_idx}")


def combine_memmap(output_dir, dimension):
    # large memmap for saving file
    final_size = 0
    index2size = {}
    for file in glob.glob(f"{output_dir}/size_*"):
        index = re.findall("\d+", file.rsplit("/", 1)[1])[0]
        with open(file) as f:
            size = int(f.readline())+1 # index to size
            final_size += size
        index2size[index] = size

    final_dstore_keys = np.memmap(f"{output_dir}/keys.npy", dtype=np.float32, mode='w+', shape=(final_size, dimension))
    final_dstore_vals = np.memmap(f"{output_dir}/vals.npy", dtype=np.int32, mode='w+', shape=(final_size, 1))

    # save key/value
    cur_index = 0
    for index, size in index2size.items():
        tmp_dstore_keys = np.memmap(f"{output_dir}/keys_{index}.npy", dtype=np.float32, mode='r', shape=(size, dimension))
        tmp_dstore_vals = np.memmap(f"{output_dir}/vals_{index}.npy", dtype=np.float32, mode='r', shape=(size, 1))
        print("loaded files")
        final_dstore_vals[cur_index:(cur_index+size), :] = tmp_dstore_vals
        print("copy vals")
        final_dstore_keys[cur_index:(cur_index+size), :] = tmp_dstore_keys

        cur_index += size
        print("current size: ", cur_index)
    print("final_size: ", cur_index)


    # dump size
    with open(f"{output_dir}/size.txt", "w") as f:
        f.write(f"{cur_index}")
    print("done")
    return final_dstore_keys, final_dstore_vals


def combine_memmap_concat(output_dir, dimension):
    # large memmap for saving file
    final_size = 0
    index2size = {}
    for file in glob.glob(f"{output_dir}/size_*"):
        index = re.findall("\d+", file.rsplit("/", 1)[1])[0]
        with open(file) as f:
            size = int(f.readline()) + 1  # index to size
            final_size += size
        index2size[index] = size

    final_dstore_keys = np.memmap(f"{output_dir}/keys.npy", dtype=np.float32, mode='w+', shape=(final_size, dimension))
    final_dstore_vals = np.memmap(f"{output_dir}/vals.npy", dtype=np.int32, mode='w+', shape=(final_size, 1))

    # save key/value
    cur_index = 0
    for index, size in index2size.items():
        tmp_dstore_keys = np.memmap(f"{output_dir}/keys_{index}.npy", dtype=np.float32, mode='r',
                                    shape=(size, dimension))
        tmp_dstore_vals = np.memmap(f"{output_dir}/vals_{index}.npy", dtype=np.float32, mode='r', shape=(size, 1))

        print("loaded files")
        final_dstore_vals[cur_index:(cur_index + size), :] = tmp_dstore_vals
        print("copy vals")
        final_dstore_keys[cur_index:(cur_index + size), :] = tmp_dstore_keys

        cur_index += size
        print("current size: ", cur_index)
    print("final_size: ", cur_index)

    # dump size
    with open(f"{output_dir}/size.txt", "w") as f:
        f.write(f"{cur_index}")
    print("done")
    return final_dstore_keys, final_dstore_vals