import numpy as np
import faiss
import time
import logging
logger = logging.getLogger(__name__)


class KNN_Dstore(object):
    def __init__(self, args):
        self.dimension = args.dimension
        self.k = args.k
        print("save_knnlm: ", args.save_knnlm)
        if args.save_knnlm is 1:
            self.dstore_size = args.dstore_size # 103225485
        else:
            with open(f"{args.dstore_dir}/size_0.txt") as f:
                size = int(f.readline()) + 1  # index to size
                self.dstore_size = size

        self.index = self.setup_faiss(args)



    def setup_faiss(self, args):
        print("dstore_size: ", self.dstore_size)
        if not args.no_load_keys:
            self.keys = np.memmap(f"{args.dstore_dir}/keys_0.npy", dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f"{args.dstore_dir}/vals_0.npy", dtype=np.int32, mode='r', shape=(self.dstore_size, 1))

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR) # faiss index of key
        logger.info('Reading datastore took {} s'.format(time.time() - start))
        return index

    def get_knns(self, queries):
        # queries: num_examples, dimension
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns



