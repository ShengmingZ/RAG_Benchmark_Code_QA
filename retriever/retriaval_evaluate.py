import json

def conala_eval(gold, pred, top_k=None):
    if top_k is None: top_k = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]

    def calc_recall(src, pred, top_k, print_result=True):
        recall_n = {x: 0 for x in top_k}
        precision_n = {x: 0 for x in top_k}

        for s, p in zip(src, pred):
            # cmd_name = s['cmd_name']
            oracle_man = s
            pred_man = p

            for tk in recall_n.keys():
                cur_result_vids = pred_man[:tk]
                cur_hit = sum([x in cur_result_vids for x in oracle_man])
                # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
                recall_n[tk] += cur_hit / (len(oracle_man)) if len(oracle_man) else 1
                precision_n[tk] += cur_hit / tk
        recall_n = {k: v / len(pred) for k, v in recall_n.items()}
        precision_n = {k: v / len(pred) for k, v in precision_n.items()}

        if print_result:
            for k in sorted(recall_n.keys()):
                print(f"{recall_n[k] :.3f}", end="\t")
            print()
            for k in sorted(precision_n.keys()):
                print(f"{precision_n[k] :.3f}", end="\t")
            print()
            for k in sorted(recall_n.keys()):
                print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
            print()

        return {'recall': recall_n, 'precision': precision_n}

    metrics = calc_recall(gold, pred, top_k)
    print(metrics)


# calc hit rate
def tldr_eval(src, pred, top_k):

    def calc_hit(src, pred, top_k):
        # top_k = TOP_K if top_k is None else top_k
        hit_n = {x: 0 for x in top_k}
        assert len(src) == len(pred), (len(src), len(pred))
        for s, p in zip(src, pred):
            cmd_name = s
            pred_man = p
            for tk in hit_n.keys():
                cur_result_vids = pred_man[:tk]
                cur_hit = any([cmd_name in x for x in cur_result_vids])
                hit_n[tk] += cur_hit
        hit_n = {k: v / len(pred) for k, v in hit_n.items()}
        for k in sorted(hit_n.keys()):
            print(f"{hit_n[k] :.3f}", end="\t")
        print()
        return hit_n

    return calc_hit(src, pred, top_k)