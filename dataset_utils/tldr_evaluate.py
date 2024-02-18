import sys
sys.path.insert(0, '/Users/zhaoshengming/Code_RAG_Benchmark')
import re
import json
import numpy as np
import editdistance
from sacrebleu.metrics import BLEU
from collections import defaultdict, Counter
from dataset_configs import TldrLoader
from generator.generate import generate_config


def clean_results(results):
    new_results = []
    for result in results:
        if 'cmd_name' in result.keys() and result['cmd_name'] == 'xkcdpass': continue
        if 'oracle_cmd' in result.keys() and result['oracle_cmd'] == 'xkcdpass': continue
        new_results.append(result)
    return new_results


"""match if exactly same"""
VAR_STR = "[[VAR]]"
def clean_command(s):
    s = s.replace("sudo", "").strip()
    s =  s.replace("`", "").replace('"', "").replace("'", "")
    #  '>', '|', '+'
    s = s.replace("|", " ").replace(">", " ").replace("<", " ")
    s = " ".join(s.split())
    return s

def anonymize_command(s):
    s = s.replace("={", " {")
    var_to_pc_holder = defaultdict(lambda: len(var_to_pc_holder))
    for var in re.findall("{{(.*?)}}", s):
        _ = var_to_pc_holder[var]
    for var, id in var_to_pc_holder.items():
        var_str = "{{%s}}" % var
        s = s.replace(var_str, f"{VAR_STR}_{id}")
    # s = re.sub("{{.*?}}", VAR_STR, s)
    return s

def clean_anonymize_command(s):
    return anonymize_command(clean_command(s))

def calc_template_matching(gold, pred):
    ag = clean_anonymize_command(gold)
    ap = clean_anonymize_command(pred)
    m = {'template_matching': int(ag == ap)}
    ag = ' '.join(ag.split()[1:])
    ap = ' '.join(ap.split()[1:])
    m['no_cmd_template_matching'] = int(ag == ap)
    return m


"""calc tokenF1"""
def get_bag_of_words(cmd):
    cmd = clean_anonymize_command(cmd)
    tokens = cmd.strip().split()
    return tokens

def token_prf(tok_gold, tok_pred, match_blank=False):
    if match_blank and len(tok_gold) == 0: # do not generate anything
        if len(tok_pred) == 0:
            m = {'r': 1, 'p': 1, 'f1': 1}
        else:
            m = {'r': 0, 'p': 0, 'f1': 0}
    else:
        tok_gold_dict = Counter(tok_gold)
        tok_pred_dict = Counter(tok_pred)
        tokens = set([*tok_gold_dict] + [*tok_pred_dict])
        hit = 0
        for token in tokens:
            hit += min(tok_gold_dict.get(token, 0), tok_pred_dict.get(token, 0))
        p = hit / (sum(tok_pred_dict.values()) + 1e-10)
        r = hit / (sum(tok_gold_dict.values()) + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        m = {'r': r, 'p': p, 'f1': f1}
    return m

def measure_bag_of_word(gold, pred):
    tok_gold = get_bag_of_words(gold)
    tok_pred = get_bag_of_words(pred)
    m = token_prf(tok_gold, tok_pred) # whole sentence
    gold_cmd = tok_gold[0] if len(tok_gold) else "NONE_GOLD"
    pred_cmd = tok_pred[0] if len(tok_pred) else "NONE_PRED"
    m = {**m, 'cmd_acc': int(gold_cmd == pred_cmd)}

    # without cmd
    no_cmd_m = token_prf(tok_gold[1:], tok_pred[1:], match_blank=True)
    no_cmd_m = {f"no_cmd_{k}": v for k, v in no_cmd_m.items()}
    m = {**m, **no_cmd_m}
    return m


"""edit distance"""
def calc_edit_distance(gold, pred):
    ag = clean_anonymize_command(gold)
    ap = clean_anonymize_command(pred)
    ag_toks = ag.split()
    ap_toks = ap.split()
    m = {'edit_distance': editdistance.eval(ag_toks, ap_toks)}
    return m


"""calc BLEU"""
def charBLEU(gold_list, pred_list):
    bleu = BLEU(tokenize='char')
    pred_list = [clean_anonymize_command(pred) for pred in pred_list]
    gold_list = [clean_anonymize_command(gold) for gold in gold_list]
    bleu_score = bleu.corpus_score(pred_list, [gold_list]).score
    return bleu_score


"""cmd consistency"""
def calc_cmd_consistency(ret_cmds, pred):
    tok_pred = get_bag_of_words(pred)
    pred_cmd = tok_pred[0] if len(tok_pred) else "NONE_PRED"
    pred_cmd = pred_cmd.split('-')[0].split('_')[0].split('.')[0]
    _ret_cmds = list()
    for cmd in ret_cmds:
        _ret_cmds.append(cmd.split('_')[0].split('-')[0].split('.')[0])
    _ret_cmds = list(set(_ret_cmds))

    m = {'cmd_consistency': int(pred_cmd in _ret_cmds)}
    if pred_cmd not in _ret_cmds: print(_ret_cmds, pred_cmd)
    return m


def tldr_evaluate(args):
    tldr_loader = TldrLoader()
    qs_list = tldr_loader.load_qs_list(args.dataset_type)
    oracle_list = tldr_loader.load_oracle_list(args.dataset_type)
    _qs_list, _oracle_list = list(), list()
    for (qs, oracle) in zip(qs_list, oracle_list):
        assert qs['qs_id'] == oracle['qs_id']
        if oracle['doc_keys'][0] == 'xkcdpass': continue
        _qs_list.append(qs)
        _oracle_list.append(oracle)
    qs_list, oracle_list = _qs_list, _oracle_list
    gene_results = json.load(open(args.save_file, 'r'))
    assert len(gene_results) == len(qs_list) == len(oracle_list)

    metric_list = defaultdict(list)
    gold_list, pred_list = list(), list()
    for idx, qs in enumerate(qs_list):
        gold = oracle_list[idx]['output']
        pred = gene_results[idx]['output']
        gold_list.append(gold)
        pred_list.append(pred)
        for k, v in calc_template_matching(gold, pred).items():
            metric_list[k].append(v)
        for k, v in measure_bag_of_word(gold, pred).items():
            metric_list[k].append(v)
        for k, v in calc_edit_distance(gold, pred).items():
            metric_list[k].append(v)
        # cmd consistency
        ret_cmds = gene_results[idx]['ret_cmd']
        if len(ret_cmds) > 0:
            for k, v in calc_cmd_consistency(ret_cmds, pred).items():
                metric_list[k].append(v)

    for k, v in metric_list.items():
        metric_list[k] = np.mean(v)

    metric_list['bleu_char'] = charBLEU(gold_list, pred_list)

    return metric_list


if __name__ == '__main__':
    in_program_call = '--dataset tldr --top_k 1 --k_line 5 --retriever unrelated --dataset_type dev'
    args = generate_config()
    args.dataset_type = 'dev'

    metric_list = tldr_evaluate(args)

    print(f'exact match: {metric_list["template_matching"]*100: .2f}%')
    print(f'cmd acc: {metric_list["cmd_acc"]*100: .2f}%')
    print(f'tokenF1: {metric_list["f1"]*100: .2f}%')
    print(f'charBleu: {metric_list["bleu_char"]: .2f}')
    if 'cmd_consistency' in metric_list.keys():
        print(f'cmd consistency: {metric_list["cmd_consistency"]*100: .2f}%')