import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    if (dataset == 'wn18rr'):
        NUM_RELATIONS = 11
    elif (dataset == 'fb15k'):
        NUM_RELATIONS = 237
    elif (dataset == 'umls'):
        NUM_RELATIONS = 46
    elif (dataset == 'kinship'):
        NUM_RELATIONS = 25
    else:
        sys.exit('Invalid Dataset!')
    orig_rules = set()
    if (len(sys.argv) > 2):
        rule_file = sys.argv[2]
    else:
        rule_file = f'../data/{dataset}/{dataset}_rules.txt'
    with open(rule_file, 'r') as fi:
        for line in fi:
            rule = line.strip().split()
            rule = tuple([int(_) for _ in rule])
            orig_rules.add(rule)
    new_rules = orig_rules.copy()
    for rule in orig_rules:
        for i in range(1, len(rule)):
            new_rule = []
            new_rule.append((NUM_RELATIONS+rule[i])%(2*NUM_RELATIONS))
            for k in range(i+1,len(rule)):
                new_rule.append(rule[k])
            new_rule.append((NUM_RELATIONS + rule[0])%(2*NUM_RELATIONS))
            for k in range(1,i):
                new_rule.append(rule[k])
            new_rules.add(tuple(new_rule))
    new_rules = sorted(new_rules)
    with open(f'../data/{dataset}/{dataset}_rules_abd.txt', 'w') as fi:
        for i in new_rules:
            output = " ".join([str(i) for i in list(i)])
            fi.write(output+'\n')
    inv_rules = orig_rules.copy()
    for rule in orig_rules:
        new_rule = []
        for relation in rule[1:]:
            new_rule.append((NUM_RELATIONS+relation)%(2*NUM_RELATIONS))
        new_rule.reverse()
        new_rule = [(NUM_RELATIONS+rule[0])%(2*NUM_RELATIONS)] + new_rule
        inv_rules.add(tuple(new_rule))
    with open(f'../data/{dataset}/{dataset}_rules_inv.txt', 'w') as fi:
        for i in inv_rules:
            output = " ".join([str(i) for i in list(i)])
            fi.write(output+'\n')
    fin_rules = inv_rules.copy()
    for rule in inv_rules:
        for i in range(1, len(rule)):
            new_rule = []
            new_rule.append((NUM_RELATIONS+rule[i])%(2*NUM_RELATIONS))
            for k in range(i+1,len(rule)):
                new_rule.append(rule[k])
            new_rule.append((NUM_RELATIONS + rule[0])%(2*NUM_RELATIONS))
            for k in range(1,i):
                new_rule.append(rule[k])
            fin_rules.add(tuple(new_rule))
    fin_rules = sorted(fin_rules)
    if (len(sys.argv) > 3):
        out_file = sys.argv[3]
    else:
        out_file = f'../data/{dataset}/{dataset}_rules_invabd.txt'
    with open(out_file, 'w') as fi:
        for i in fin_rules:
            output = " ".join([str(i) for i in list(i)])
            fi.write(output+'\n')
