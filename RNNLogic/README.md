# Augmentations on RNNLogic+

In this folder, we provide the baseline code and rule sets for our experiments with RNNLogic+. The `data` folder containing the rulesets, datasets and RotatE checkpoints used in our experiments can be downloaded from Google Drive: [Link](https://drive.google.com/file/d/11IyXV0lpWi8z_jVPaxNsAdGX_2FkdTGd/view?usp=share_link). The original and augmented rule sets can be found in the respective dataset directories inside the `data` folder.

Most of the code is taken from the code repository of the following paper:
Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, Jian Tang, "RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs", ICLR 2021.
This code is available at:
https://github.com/DeepGraphLearning/RNNLogic

## Step 1: Abduction and Rule Inversion

In the first step, we perform abduction and rule_inversion on the seed set of rules from random walk or RNNLogic. 

To do that, go to the folder `src`, and run:
`python modify_rules.py DATASET RULE_FILE AUG_FILE`
where DATASET is one of "wn18rr" for WN18RR, "fb15k" for FB15k-237, "umls" for UMLS and "kinship" for Kinship. 
If augmentation needs to be done for custom rule set, the rule file can be provided as the additional command line argument RULE_FILE after the DATASET. 
This will create a file "DATASET_rules_invabd.txt" in the data directory of the corresponding dataset. If the augmented rules are to be dumped to another file (with a custom RULE_FILE), it can be provided as the additional command line argument AUG_FILE.

## Step 2: Filtering

In this step, we first filter the rules obtained after abduction and inversion according to the PCA metric.

To do that, go to the folder `src`, and run:
`python score_rules.py DATASET IN_FILE OUT_FILE DEVICE BATCHES`
where:
DATASET is one of "wn18rr", "fb15k", "umls" and "kinship". 
IN_FILE is the rule file to be filtered.
OUT_FILE is the filtered rule file. 
DEVICE can be cpu or the cuda on which the code is to be run. 
BATCHES is the number of batches to be used to compute the PCA scores for WN18RR or FB15k-237.
This creates an additional file "DATASET_rule_scores.txt" where the rule scores are present. 

## Step 3: Random Walk-based Rule Augmentation

In this step, we discover rules through Random Walks from each entity in the dataset, and filter them using the PCA metric.

To do that, go to the folder `src`, and run:
`python rule_discovery.py DATASET OUT_FILE RULE_FILE FIN_FILE NUM_WALKS DEVICE BATCHES`
where:
DATASET is one of "wn18rr", "fb15k", "umls" and "kinship". 
OUT_FILE is rule file to which rules discovered through Random Walks are dumped. 
RULE_FILE is the rule file with which these random walk rules have to be merged. 
FIN_FILE is the final rule file after augmentation.
NUM_WALKS is the number of random walks to run from each entity in the dataset. 
DEVICE can be cpu or the cuda on which the code is to be run. 
BATCHES is the number of batches to be used to compute the PCA scores for filtering in WN18RR or FB15k-237.

## Step 4: Running RNNLogic+

Next, we are ready to run RNNLogic+. To do that, please first edit the config file for each dataset in the folder `config`, and then go to folder `src`. RotatE can be enabled/disabled inside the config as well (set entity_feature to bias to disable RotatE). 

If you would like to use single-GPU training, please run:

`python run_predictorplus.py --config ../config/DATASET_predictorplus.yaml` 

If you would like to use multi-GPU training, please run:

`python -m torch.distributed.launch --nproc_per_node=NUM_GPUS run_predictorplus.py --config ../config/DATASET_predictorplus.yaml`


## Requirements
- Python 3.7.13
- Easydict 1.9
- PyYAML 6.0
- Torch 1.10.2
- Torch-Scatter 2.0.9
- NumPy

