# Augmentations on ExpressGNN

In this folder, we provide the baseline code and rule sets for our experiments with ExpressGNN. The original and augmented rule sets can be found in the respective dataset directories inside the `data` folder. The original rules for each dataset have been obtained by selecting the top 5-10 rules per relation (in the rule head) based on PCA score from the RNNLogic rules for that dataset. The original rules are in files of the form expressgnn_rules_DATASET.txt and augmented rules are present in files with the name expressgnn_rules_aug.txt.

The code present here is a modification of the code repository of the following paper:
Yuyu Zhang, Xinshi Chen, Yuan Yang, Arun Ramamurthy, Bo Li, Yuan Qi, Le Song, "Efficient Probabilistic Logic Reasoning with Graph Neural Networks", ICLR 2020.
This code is available at:
https://github.com/expressGNN/ExpressGNN

## Running Training
The following command is an example of running ExpressGNN on any DATASET on GPU without augmentation:
```
python -m main.train -data_root data/DATASET -rule_filename expressgnn_rules_DATASET.txt -slice_dim 16 -batchsize 16 -use_gcn 1 -num_hops 1 -embedding_size 128 -gcn_free_size 127 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -exp_name DATASET -device cuda
```
Here DATASET can be one of "WN18RR", "UMLS" and "Kinship".

The following command is an example of running ExpressGNN on any DATASET on GPU with augmentation:
```
python -m main.train -data_root data/DATASET -rule_filename expressgnn_rules_aug.txt -slice_dim 16 -batchsize 16 -use_gcn 1 -num_hops 1 -embedding_size 128 -gcn_free_size 127 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -exp_name DATASET -device cuda
```

## Requirements
- python 3.7
- pytorch 1.1
- scikit-learn
- networkx
- tqdm
