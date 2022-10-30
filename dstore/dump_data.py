import sys
sys.path.append("../")
import logging
import subprocess
from save_dstore import combine_memmap

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cmd(subset_index, num_gpus):
    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "True",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/wikitext-103-v1_test",
    #         "--raw_file", "wikitext-103-v1", # small_external_512
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         # "--dstore_size", 10**20
    #         ]
    # #
    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-s-tmp/amazon-review",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/LM-BFF/amazon-review/index/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    command = ["python", "../main.py",
            "--model", "gpt2",
            "--dimension", "768",
            "--save_knnlm", "1",
            "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-s-tmp/imdb-hg",
            "--raw_file", "imdb",
            "--subset_index", f"{subset_index}",
            "--num_gpus", f"{num_gpus}",
            "--dstore_size", f"{10**10}"
            ]


    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/amazon-review",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/LM-BFF/amazon-review/index/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}"
    #         ]
    #
    # command = ["python", "../main.py",
    #         "--model", "gpt2-xl",
    #         "--dimension", "1600",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-xl/amazon-review",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/LM-BFF/amazon-review/index/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}",
    #         "--batch_size", "8"
    #         ]


    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/cc_news",
    #         "--raw_file", "cc_news",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/wikitext-103-v1",
    #         "--raw_file", "wikitext-103-v1",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/imdb-tlm-small",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/TLM/imdb-tlm/small_external_512.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]
    #
    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/ag-tlm-small",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/TLM/ag-tlm/small_external.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/ag-tlm-small",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/TLM/ag-tlm/small_external.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/sst-2_train",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/surface-form/sst-2/train.tsv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**8}"
    #         ]
    # column_names = ["title", "text", "id", "label"]
    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/imdb_train",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/TLM/imdb-tlm/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}",
    #         "--batch_size", "8"
    #         # "--column_names", f"{column_names}"
    #         ]
    # swj: general corpus
    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dataset_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/wikitext_ccnews_amazon_imdb",
    #         "--raw_file", "wikitext-103-v1,cc_news,amazon,imdb_30_20",
    #         "--subset_index", "0",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}",
    #         "--batch_size", "16"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dataset_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/wikitext_ccnews_amazon_imdb",
    #         "--raw_file", "wikitext-103-v1,cc_news,amazon,imdb",
    #         "--subset_index", "0",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}",
    #         "--batch_size", "16"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dataset_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/cc_news_40-60",
    #         "--raw_file", "cc_news_40-60",
    #         "--subset_index", "0",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**9}",
    #         "--batch_size", "16"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dataset_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/amazon_polarity",
    #         "--raw_file", "amazon_polarity",
    #         "--subset_index", "0",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**8}",
    #         "--batch_size", "16"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2-large",
    #         "--dimension", "1280",
    #         "--save_knnlm", "1",
    #         "--dataset_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/imdb_hg",
    #         "--raw_file", "imdb_hg",
    #         "--subset_index", "0",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**8}",
    #         "--batch_size", "16"
    #         ]


    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/imdb_sst2_mr",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/tapt/imdb_sst2_mr/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/agnews_train",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/LM-BFF/agnews/index/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    # command = ["python", "../main.py",
    #         "--model", "gpt2",
    #         "--dimension", "768",
    #         "--save_knnlm", "1",
    #         "--dstore_dir", "/gscratch/zlab/swj0419/knnlm/data/checkpoints/hyp_train",
    #         "--raw_file", "/gscratch/zlab/swj0419/knnlm/data/TLM/hyp-tlm/train.csv",
    #         "--subset_index", f"{subset_index}",
    #         "--num_gpus", f"{num_gpus}",
    #         "--dstore_size", f"{10**10}"
    #         ]

    return command



def run_dump_phrases(num_gpus):
    for subset_index in range(num_gpus):
        cmd_run = get_cmd(subset_index, num_gpus)
        print(" ".join(cmd_run))
        # subprocess.Popen(cmd_run)

# from sentence_transformers import SentenceTransformer, util
# from tqdm import tqdm
# import numpy
#
# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
# embeddings = model.encode(sentences)
# all_emb = []
# for d in tqdm(dataset["train"]):
#     all_emb.append(model.encode(d["text"]))
#     # if "digital camera" in d['text']:
#     #     print(d['text'])
# all_score = []
# query = model.encode("the movie would seem less of a trifle if ms . sugarman followed through on her defiance of the saccharine . It was.")
# for e in all_emb:
#     all_score.append(util.pytorch_cos_sim(e, query))
# sort_index = numpy.argsort(all_score)[::-1][:5]

if __name__ == '__main__':
    num_gpus = 1
    run_dump_phrases(num_gpus)

    # output_dir = "/gscratch/zlab/swj0419/knnlm/data/checkpoints/gpt2-large/wikitext_ccnews_amazon_imdb"
    # dimension = 760
    # print("output_dir: ", output_dir)
    # combine_memmap(output_dir, dimension)
