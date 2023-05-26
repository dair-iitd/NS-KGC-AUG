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

random.seed(42)

THRESHOLD = 0.1
DEPTHS = 4

def pca_confidence(graph, rules, DATASET, OUT_FILE, DEVICE, BATCHES):
    # Formula = Number of Positive Examples satisfied by Rule / Number of Examples satisfied by Rule for (h, r) present in KG 
    final_rules = set()
    num_heads = graph.entity_size
    curr_rel = -1
    gt = None
    curr_heads = None
    agg_score = 0
    if (DATASET in ['wn18rr', 'FB15k-237']):
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
            if (rule_score >= THRESHOLD):
                final_rules.add(rule)
                agg_score += rule_score
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
            if (rule_score >= THRESHOLD):
                final_rules.add(rule)
                agg_score += rule_score
    with open(OUT_FILE, 'w') as fi:
        for rule in final_rules:
            fi.write(str(rule[0]) + " " + " ".join([str(_) for _ in list(rule[1])]) + '\n')
    print(f'Avg Score: {agg_score/len(final_rules)}')
    print(f'Rules Discovered: {len(final_rules)}')

def walk(graph, start, node, depth, NUM_RELATIONS):
    relation = random.randrange(NUM_RELATIONS)
    if (graph.encode_hr(node, relation) not in graph.hr2o):
        return False, []
    candidates = graph.hr2o[graph.encode_hr(node, relation)]
    if (len(candidates) == 0):
        return False, []
    if (depth == DEPTHS - 1):
        rules = []
        for candidate in candidates:
            for head_relation in range(NUM_RELATIONS):
                if (graph.encode_hr(start, head_relation) not in graph.hr2o):
                    continue
                if (candidate in graph.hr2o[graph.encode_hr(start, head_relation)]):
                    rules.append([head_relation, [relation]])
        if (len(rules) == 0):
            return False, []
        else:
            return True, rules
    else:
        candidate = candidates[random.randrange(len(candidates))]
        found, rules = walk(graph, start, candidate, depth+1, NUM_RELATIONS)
        for i in range(len(rules)):
            rules[i][1] = [relation] + rules[i][1]
        if (depth > 0):
            for candidate in candidates:
                for head_relation in range(NUM_RELATIONS):
                    if (graph.encode_hr(start, head_relation) not in graph.hr2o):
                        continue
                    if (candidate in graph.hr2o[graph.encode_hr(start, head_relation)]):
                        rules.append([head_relation, [relation]])
        if (len(rules) > 0):
            return True, rules
        else:
            return False, []


def random_walk(graph, DATASET, NUM_RELATIONS, NUM_WALKS, OUT_FILE, DEVICE, BATCHES):
    rules_total = set()
    for entity in range(graph.entity_size):
        for _ in range(NUM_WALKS):
            found, rules = walk(graph, entity, entity, 0, NUM_RELATIONS)
            if (found):
                for rule in rules:
                    rules_total.add(tuple([rule[0], tuple(rule[1])]))
        print(f'Done with entity {entity}')
    print(f'Initial Rules Discovered: {len(rules_total)}')
    rules_total = list(rules_total)
    rules_total.sort()
    pca_confidence(graph, rules_total, DATASET, OUT_FILE, DEVICE, BATCHES)

def merge_rules(OUT_FILE, RULE_FILE, FIN_FILE):
    rules = set()
    with open(OUT_FILE, 'r') as fi:
        for line in fi:
            rules.add(tuple(line.strip().split()))
    temp = len(rules)
    print(f'New Rules: {len(rules)}')
    with open(RULE_FILE, 'r') as fi:
        for line in fi:
            rules.add(tuple(line.strip().split()))
    print(f'Orig Rules - New Rules: {len(rules) - temp}')
    print(f'Final Rules: {len(rules)}')
    with open(FIN_FILE, 'w') as fi:
        for rule in rules:
            fi.write(" ".join(rule) + '\n')


def main():
    DATASET = sys.argv[1]
    if (DATASET == 'wn18rr'):
        BATCHES = int(sys.argv[7])
        NUM_RELATIONS = 22
    elif (DATASET == 'fb15k'):
        BATCHES = int(sys.argv[7])
        NUM_RELATIONS = 474
    elif (DATASET == 'umls'):
        BATCHES = 1
        NUM_RELATIONS = 92
    elif (DATASET == 'kinship'):
        BATCHES = 1
        NUM_RELATIONS = 50
    else:
        sys.exit('Invalid Dataset!')
    DATA = f'../data/{DATASET}'
    OUT_FILE = sys.argv[2]
    RULE_FILE = sys.argv[3]
    FIN_FILE = sys.argv[4]
    NUM_WALKS = int(sys.argv[5])
    DEVICE = sys.argv[6]
    graph = KnowledgeGraph(DATA)
    random_walk(graph, DATASET, NUM_RELATIONS, NUM_WALKS, OUT_FILE, DEVICE, BATCHES)
    merge_rules(OUT_FILE, RULE_FILE, FIN_FILE)


if __name__ == '__main__':
    main()