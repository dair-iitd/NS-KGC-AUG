import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
import torch
import random

from data import KnowledgeGraph

THRESHOLD = 0.01
MIN_GRND = 10


def pca_confidence(graph, rules, OUT_FILE, DEVICE, BATCHING, BATCHES=1):
    # Formula = Number of Positive Examples satisfied by Rule / Number of Examples satisfied by Rule for (h, r) present in KG 
    num_heads = graph.entity_size
    curr_rel = -1
    gt = None
    curr_heads = None
    agg_score = 0
    writer = open(OUT_FILE, 'w')
    if (BATCHING):
        pos_grnd_vals = {}
        tot_grnd_vals = {}
        bs = (num_heads//BATCHES) + 1
        for epoch in range(BATCHES):
            heads = torch.arange(bs*epoch, min((epoch+1)*bs, num_heads)).to(torch.device(DEVICE))
            print(f'Heads in batch: {list(heads.size())[0]}')
            for rule in rules:
                rule_head = rule[0]
                rule_body = rule[1]
                if (curr_rel != rule_head):
                    print(f'Cache Refreshed: {rule_head}')
                    curr_rel = rule_head
                    gt = graph.grounding(heads, curr_rel, [curr_rel], None).float()
                    ind = []
                    for head in range(list(heads.size())[0]):
                        if (gt[head].sum().item() > 0):
                            ind.append(head)
                    gt = gt[ind]
                    curr_heads = heads[ind]
                counts = graph.grounding(curr_heads, rule_head, rule_body, None).float()
                support = torch.clamp(counts, max=1.0)
                support_cnt = gt * support
                pos_grnd = support_cnt.sum().item()
                tot_grnd = support.sum().item()
                if (epoch == 0):
                    pos_grnd_vals[rule] = pos_grnd
                    tot_grnd_vals[rule] = tot_grnd
                else:
                    pos_grnd_vals[rule] += pos_grnd
                    tot_grnd_vals[rule] += tot_grnd
        for rule in pos_grnd_vals.keys():
            if (tot_grnd_vals[rule]==0):
                rule_score = 0
            else:
                rule_score = pos_grnd_vals[rule]/tot_grnd_vals[rule]
            print(f'Score for {rule}: {rule_score}')
            agg_score += rule_score
            writer.write(f'{str(rule[0]) + " " + " ".join([str(_) for _ in list(rule[1])])} {rule_score} {tot_grnd_vals[rule]}\n')
    else:
        heads = torch.arange(num_heads).to(torch.device(DEVICE))
        for rule in rules:
            rule_head = rule[0]
            rule_body = rule[1]
            if (curr_rel != rule_head):
                print(f'Cache Refreshed: {rule_head}')
                curr_rel = rule_head
                gt = graph.grounding(heads, curr_rel, [curr_rel], None).float()
                ind = []
                for head in range(num_heads):
                    if (gt[head].sum().item() > 0):
                        ind.append(head)
                gt = gt[ind]
                curr_heads = heads[ind]
            counts = graph.grounding(curr_heads, rule_head, rule_body, None).float()
            support = torch.clamp(counts, max=1.0)
            support_cnt = gt * support
            pos_grnd = support_cnt.sum().item()
            tot_grnd = support.sum().item()
            if (tot_grnd==0):
                rule_score = 0
            else:
                rule_score = pos_grnd/tot_grnd
            print(f'Score for {rule}: {rule_score}')
            agg_score += rule_score
            writer.write(f'{str(rule_head) + " " + " ".join([str(_) for _ in rule_body])} {rule_score} {tot_grnd}\n')
    print(f'Avg Score: {agg_score/len(rules)}')
    writer.write(f'Average: {agg_score/len(rules)}\n')
    

def filter_score(graph, RULE_SCORE_FILE, IN_FILE):
    prev_avg = 0
    prev_rules = 0
    curr_avg = 0
    curr_rules = 0
    rule2score = {}
    rule2grnd = {}
    with open(RULE_SCORE_FILE, 'r') as fi:
        for line in fi:
            rule = line.strip().split()
            if (rule[0] == 'Average:'):
                continue
            score = float(rule[-2])
            grnd = float(rule[-1])
            rule2score[tuple(rule[:-2])] = score
            rule2grnd[tuple(rule[:-2])] = grnd
    with open(IN_FILE, 'r') as fi:
        for line in fi:
            rule = line.strip().split()
            score = rule2score[tuple(rule)]
            if (score >= THRESHOLD or rule2grnd[tuple(rule)] >= MIN_GRND):
                curr_avg += score
                curr_rules += 1
            prev_avg += score
            prev_rules += 1
    print(f'Prev Num Rules: {prev_rules}')
    print(f'Prev Avg Score: {prev_avg/prev_rules}')
    print(f'Filtered Num Rules: {curr_rules}')
    print(f'Filtered Avg Score: {curr_avg/curr_rules}')
    print(f'Ratio: {curr_rules/prev_rules}')            

def main():
    DATASET = sys.argv[1]
    if (DATASET == 'wn18rr'):
        BATCHING = True
        BATCHES = int(sys.argv[5])
        NUM_RELATIONS = 22
    elif (DATASET == 'fb15k'):
        BATCHING = True
        BATCHES = int(sys.argv[5])
        NUM_RELATIONS = 474
    elif (DATASET == 'umls'):
        BATCHING = False
        BATCHES = 1
        NUM_RELATIONS = 92
    elif (DATASET == 'kinship'):
        BATCHING = False
        BATCHES = 1
        NUM_RELATIONS = 50
    else:
        sys.exit('Invalid Dataset!')
    IN_FILE = sys.argv[2]
    RULE_OUT_FILE = sys.argv[3]
    DEVICE = sys.argv[4]
    DATA = f'../data/{DATASET}'
    OUT_FILE = f'{DATASET}_rule_scores.txt'
    RULE_SCORE_FILE = f'{DATASET}_rule_scores.txt'
    graph = KnowledgeGraph(DATA)
    rules = set()
    with open(IN_FILE, 'r') as fi:
        for line in fi:
            rule = line.strip().split()
            rule = [int(_) for _ in rule]
            rule_ = tuple([rule[0], tuple(rule[1:])])
            rules.add(rule_)
    rules = list(rules)
    rules.sort()
    print(f'Num Rules: {len(rules)}')
    pca_confidence(graph, rules, OUT_FILE, DEVICE, BATCHING, BATCHES)
    filter_score(graph, RULE_SCORE_FILE, IN_FILE)


if __name__ == '__main__':
    main()