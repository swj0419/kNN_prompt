from options import Options
from transformers import AutoModelForCausalLM, AutoTokenizer
from dstore.save_dstore import load_data, save_dstore
from dstore.dstore import KNN_Dstore
from data.test_data import load_test_data
from score import EvaluatingWrapper
import logging
from pathlib import Path
import os
import shutil
import random
import torch
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to("cuda")
    model = model.eval()
    return model, tokenizer

def copy_file(args):
    shutil.copyfile("/gscratch/zlab/swj0419/knnlm/src/knnlm_gpt/score_sfc.py", f"{args.output_dir}/score_sfc.py")
    shutil.copyfile("/gscratch/zlab/swj0419/knnlm/src/knnlm_gpt/dstore/build_dstore.py", f"{args.output_dir}/build_dstore.py")
    shutil.copyfile("/gscratch/zlab/swj0419/knnlm/src/knnlm_gpt/data/data_loaders.py", f"{args.output_dir}/data_loaders.py")


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    args = options.parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.subset_index}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if "/" in args.model:
        model_name = args.model.rsplit("/", 1)[1]
    else:
        model_name = args.model
    args.output_dir = f"{args.dataset_dir}/{args.split}_{model_name}_knn-{args.knn_model}/kshot_{args.k_shot}/sample{args.n_sample}/dstore_{args.raw_file}/k{args.k}_knntemp{args.knn_temp}"
    print("output_dir: ", args.output_dir)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    copy_file(args)

    model, tokenizer = load_model(args.model)
    knn_model, knn_tokenizer = load_model(args.knn_model)

    if args.save_knnlm:
        Path(args.dstore_dir).mkdir(exist_ok=True, parents=True)
        dataloader = load_data(args)
        save_dstore(args, model, tokenizer, dataloader)
    else:
        knn_dstore = KNN_Dstore(args)
        examples, closed_label_space = load_test_data(args)
        eval_wrapper = EvaluatingWrapper(model=model, encoder=tokenizer, knn_model=knn_model, knn_tokenizer=knn_tokenizer, examples=examples, knn_dstore=knn_dstore, args=args)
        eval_wrapper.score()




