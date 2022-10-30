# kNN_prompt
# knnlm
I include an example script of running kNN-LM on RTE dataset. The datastore is built upon wikitext-103.

## write datastore file
The first step is to save the key (embedding of the leftward context) and value (next token) file in the datastore. You can change the knn-model through --model and make sure --dimension matches the dimension of last hidden representation of the knn-model. 
```
python main.py --model gpt2 \
               --dimension 768 \
               --save_knnlm 1 \
               --dstore_dir ./wikitext-103 \
                --raw_file wikitext-103-v1 \
                --subset_index 0  \
                --num_gpus 1 \
                --dstore_size 10000000000
```

## build faiss index
build faiss index from the key file
```
python ./dstore/build_dstore.py \
--dstore_mmap ./wikitext-103 \
--ncentroids 4096 \
--dstore_size 114418175 \
--faiss_index ./wikitext-103/knn.index \
--num_keys_to_add_at_a_time 500000 \
--starting_point 0 \
--dimension 768 \
--cuda 1
```

## Model inference using kNN-LM
Hyperparameters for tuning: knn_temp (temperature for knn distribution) and k (number of k nearest neighbors). k_shot refers to the number of in-context-learning examples. I use n_sample to subsample the test data for fast inference.
```
raw_file=wikitext-103
knn_model=gpt2-large
dataset=rte
k=1600
knn_temp=3

 python main.py \
    --model gpt2-large \
    --knn_model ${knn_model} \
    --n_sample 1000000 \
    --dimension 1280 \
    --raw_file ${raw_file} \
    --dstore_dir /gscratch/zlab/swj0419/knnlm/data/checkpoints/${knn_model}/${raw_file} \
    --indexfile /gscratch/zlab/swj0419/knnlm/data/checkpoints/${knn_model}/${raw_file}/knn.index \
    --dataset_dir /gscratch/zlab/swj0419/knnlm/data/final/$dataset \
    --k $k \
    --dataset_name $dataset \
    --batch_size 5 \
    --knn_temp $knn_tmp \
    --k_shot 0 \
    --sim_func do_not_recomp_l2 \
    --scoring softmax
```

TBC
