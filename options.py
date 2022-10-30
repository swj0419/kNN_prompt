import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        self.add_model()
        self.add_knn_options()
        self.eval_data()


    def add_knn_options(self):
        self.parser.add_argument("--raw_file", type=str, default=None)
        self.parser.add_argument("--subset_index", type=int, default=0)

        self.parser.add_argument('--dstore_size', type=int, default=10**9, help='number of items saved in the datastore memmap')
        self.parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
        self.parser.add_argument("--no_load_keys", type=bool, default=False)
        self.parser.add_argument("--dstore_dir", type=str, default=None, help="dir to save key val")
        self.parser.add_argument("--indexfile", type=str, default="checkpoints/knn.index")
        self.parser.add_argument("--save_knnlm", type=int, default=0)
        self.parser.add_argument("--sim_func", type=str, default="do_not_recomp_l2", help="do_not_recomp_l2, l2, dot")
        self.parser.add_argument("--knn_temp", type=float, default=1)
        self.parser.add_argument("--knn_score_factor", type=float, default=1)

        self.parser.add_argument("--knn_dist_half", type=bool, default=True)

        self.parser.add_argument('--k', type=int, default=300)
        self.parser.add_argument("--scoring", type=str, default="softmax")
        # self.parser.add_argument('--knn_lambda', type=float, default=0.3)


    def initialize_parser(self):
        self.parser.add_argument("--num_gpus", type=int, default=10)
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        self.parser.add_argument("--verbose_logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                        "A number of warnings are expected for a normal SQuAD evaluation.",)
        self.parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
        self.parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html",)
        self.parser.add_argument("--load_cache", type=int, default=1)


    def add_model(self):
        self.parser.add_argument("--model", type=str, default=None)
        self.parser.add_argument("--knn_model", type=str, default="gpt2")
        self.parser.add_argument("--batch_size", type=int, default=16)

    def eval_data(self):
        self.parser.add_argument("--dataset_name", type=str, default=None)
        self.parser.add_argument("--column_names", type=list, default= ["title", "text", "id", "label"])
        self.parser.add_argument("--dataset_dir", type=str, default=None)
        self.parser.add_argument("--split", type=str, default="test")
        self.parser.add_argument("--n_sample", type=int, default=100)
        self.parser.add_argument("--variant", type=int, default=0)
        self.parser.add_argument("--k_shot", type=int, default=0)