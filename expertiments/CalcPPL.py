import argparse
import json
import sys
sys.path.append('../../Code_RAG_Benchmark')
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def calc_ppl(logprobs_list):
    perplexities = []
    for logprobs in logprobs_list:
        if len(logprobs) == 1: logprobs = logprobs[0]
        if len(logprobs) == 1: logprobs = logprobs[0]
        # for idx, data in enumerate(logprobs):
        #     if data == 0:
        #         logprobs = logprobs[:idx]
        #         break
        # Calculate perplexity for this sequence
        # Perplexity = exp(-1/N * sum(log_probs))
        avg_log_prob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_log_prob)
        perplexities.append(perplexity)

    # Return average perplexity across all sequences
    return sum(perplexities) / len(perplexities)





def analyze_correlation(data1, data2, labels=None):
    """
    Analyze correlation between two groups of float data

    Args:
        data1: List/array of float values (group 1)
        data2: List/array of float values (group 2)
        labels: Optional tuple of labels for the two groups

    Returns:
        dict: Dictionary containing all correlation metrics
    """
    if labels is None:
        labels = ("Group 1", "Group 2")

    data1 = np.array(data1)
    data2 = np.array(data2)

    if len(data1) != len(data2):
        raise ValueError("Both groups must have the same size")

    results = {}

    # 1. Pearson Correlation (most common for linear relationships)
    pearson_r, pearson_p = stats.pearsonr(data1, data2)
    results['pearson'] = {
        'correlation': pearson_r,
        'p_value': pearson_p,
        'significant': pearson_p < 0.05,
        'description': 'Linear correlation'
    }

    results = results['pearson']

    # # 2. Spearman Correlation (rank-based, good for monotonic relationships)
    # spearman_r, spearman_p = stats.spearmanr(data1, data2)
    # results['spearman'] = {
    #     'correlation': spearman_r,
    #     'p_value': spearman_p,
    #     'significant': spearman_p < 0.05,
    #     'description': 'Monotonic correlation (rank-based)'
    # }
    #
    # # 3. Kendall's Tau (another rank-based method, more robust)
    # kendall_tau, kendall_p = stats.kendalltau(data1, data2)
    # results['kendall'] = {
    #     'correlation': kendall_tau,
    #     'p_value': kendall_p,
    #     'significant': kendall_p < 0.05,
    #     'description': 'Rank correlation (robust to outliers)'
    # }
    #
    # # 4. Additional statistics
    # results['summary'] = {
    #     'n': len(data1),
    #     'mean_diff': np.mean(data1) - np.mean(data2),
    #     'r_squared': pearson_r ** 2,  # Explained variance
    #     'rmse': np.sqrt(np.mean((data1 - data2) ** 2))  # Root mean square error
    # }

    return results



if __name__ == '__main__':
    model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                            "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                            "meta-llama/Llama-2-13b-chat-hf": "llama2-13b"}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', default='DocNum', choices=['single', 'oracle', 'recall', 'DocNum', 'prompt'])
    # parser.add_argument('--k', type=int, default=1, help='Doc Num, only effective if mode is "DocNum"')
    
    args = parser.parse_args()
    
    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'
    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'
    else: raise Exception('unknown model')
    
    if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        ks = [1,3,5,7,10,13,16,20]
    else:
        # ks = [1,3,5,10,15,20,30,40]
        ks = [1,5,10,15,20]
        # ks = [3]
    
    model_name_for_path = model_names_for_path[args.model]
    if args.mode == 'DocNum':
        for k in ks:
            result_path = f'../data/{args.dataset}/new_results/DocNum/{k}_{model_name_for_path}.json'
            logprobs_list = [item['logprobs'] for item in json.load(open(result_path, 'r'))]
            print(f'for {args.dataset} k={k}, AVG PPL: {round(calc_ppl(logprobs_list), 5)}')

    # doc_num_ppl_data = dict(NQ=[[0.435, 0.400, 0.544, 0.559, 0.552, 0.546, 0.400, 0.400],
    #                             [0.427, 0.400, 0.525, 0.543, 0.545, 0.545, 0.400, 0.549],
    #                             [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]],
    #                         TriviaQA=[[0.732, 0.700, 0.801, 0.825, 0.836, 0.835, 0.700, 0.700],
    #                                   [0.740, 0.700, 0.789, 0.818, 0.824, 0.820, 0.700, 0.840],
    #                                   [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]],
    #                         HotpotQA=[[0.354, 0.300, 0.415, 0.443, 0.436, 0.429, 0.300, 0.300],
    #                                   [0.346, 0.300, 0.407, 0.423, 0.428, 0.427, 0.300, 0.438],
    #                                   [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
    #                         CoNaLa=[[1.15786, 1.16620, 1.16587, 1.16477, 1.17476, 1.17538, 1.17445, 1.16720],  # llama2
    #                                 [1.03702, 1.04643, 1.04296, 1.04402, 1.05064, 1.05248, 1.05434, 1.04689],  # gpt-3.5
    #                                 [1.03608, 1.04007, 1.03521, 1.03829, 1.03717, 1.03771, 1.03699, 1.03549]],  # gpt-4o
    #                         DS1000=[[1.14383, 1.12741, 1.12806, 1.12710, 1.12337, 1.11702, 1.11472, 1.10787],
    #                                 [1.03635, 1.03939, 1.03984, 1.03805, 1.03957, 1.04168, 1.04079, 1.04605],
    #                                 [1.03150, 1.03604, 1.03141, 1.03313, 1.03190, 1.03205, 1.02931, 1.03142]],
    #                         PNE=[[1.11725, 1.13907, 1.13781, 1.14587, 1.14441, 1.14197, 1.13808, 1.12771],
    #                              [1.02660, 1.02309, 1.02266, 1.02137, 1.02051, 1.02087, 1.02030, 1.02166],
    #                              [1.01680, 1.02220, 1.02178, 1.02115, 1.02020, 1.01803, 1.01740, 1.01688]],
    #                         )

    # docnum_perf_data = dict(NQ=      [[0.435, 0.400, 0.544, 0.559, 0.552, 0.546, 0.400, 0.400],
    #                                   [0.427, 0.400, 0.525, 0.543, 0.545, 0.545, 0.400, 0.549],
    #                                   [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]],
    #                         TriviaQA=[[0.732, 0.700, 0.801, 0.825, 0.836, 0.835, 0.700, 0.700],
    #                                   [0.740, 0.700, 0.789, 0.818, 0.824, 0.820, 0.700, 0.840],
    #                                   [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]],
    #                         HotpotQA=[[0.354, 0.300, 0.415, 0.443, 0.436, 0.429, 0.300, 0.300],
    #                                   [0.346, 0.300, 0.407, 0.423, 0.428, 0.427, 0.300, 0.438],
    #                                   [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
    #                         CoNaLa=  [[0.238, 0.238, 0.214, 0.238, 0.214, 0.298, 0.250, 0.274],     # llama2
    #                                   [0.345, 0.286, 0.333, 0.345, 0.286, 0.345, 0.369, 0.417],     # gpt-3.5
    #                                   [0.369, 0.393, 0.417, 0.429, 0.393, 0.393, 0.393, 0.429]],    # gpt-4o
    #                         DS1000=  [[0.146, 0.121, 0.135, 0.121, 0.121, 0.140, 0.127, 0.166],
    #                                   [0.268, 0.248, 0.261, 0.287, 0.318, 0.318, 0.318, 0.312],
    #                                   [0.446, 0.408, 0.439, 0.420, 0.382, 0.376, 0.414, 0.357]],
    #                         PNE=     [[0.539, 0.533, 0.539, 0.557, 0.611, 0.611, 0.569, 0.551],
    #                                   [0.569, 0.647, 0.659, 0.707, 0.701, 0.731, 0.713, 0.725],
    #                                   [0.647, 0.689, 0.695, 0.695, 0.701, 0.701, 0.749, 0.731]],
    #                         )

    # for dataset_name in ['CoNaLa', 'DS1000', 'PNE']:
    #     for idx, model_name in enumerate(['llama2', 'gpt-3.5', 'gpt-4o']):
    #         print(f'******* pearson correlation between PPL and pass@1 under {dataset_name} and {model_name} *******')
    #         ppl_data = doc_num_ppl_data[dataset_name][idx][1:]
    #         # for ppl_idx in range(len(ppl_data)):
    #         #     ppl_data[ppl_idx] -= 0.0005 * ppl_idx
    #         perf_data = docnum_perf_data[dataset_name][idx][1:]

    #         result = analyze_correlation(data1=ppl_data, data2=perf_data)

    #         print(result)

