import argparse
import json
import sys
sys.path.append('..')
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency



def pred_distribution_analysis(pred_list_a, pred_list_b, alpha=0.05):
    """
    assert pred_list_a and pred_list_b are a list of prediction, with True/False values
    :param pred_list_a:
    :param pred_list_b:
    :return:
    """
    assert len(pred_list_a) == len(pred_list_b)
    n = len(pred_list_a)
    only_a_correct_count = 0
    only_b_correct_count = 0
    mutual_correct_count = 0
    both_wrong_count = 0
    for i in range(len(pred_list_a)):
        if pred_list_a[i] and pred_list_b[i]:
            mutual_correct_count += 1
        elif pred_list_a[i] and not pred_list_b[i]:
            only_a_correct_count += 1
            # print(qid_list[i])
        elif not pred_list_a[i] and pred_list_b[i]:
            only_b_correct_count += 1
        else:
            both_wrong_count += 1
    print("Percentage of only a correctly solve: ", round(only_a_correct_count/len(pred_list_a), 3))
    print('Percentage of only b correctly solve: ', round(only_b_correct_count/len(pred_list_b), 3))
    print('Percentage of mutual correctly solve: ', round(mutual_correct_count/len(pred_list_a), 3))
    # print('percentage of only correct in k=1 samples: ', round(only_a_correct_count/len(pred_list_a)*100, 3))


    contingency_table = np.array([
        [mutual_correct_count, only_a_correct_count],
        [only_b_correct_count, both_wrong_count]
    ])

    # result = mcnemar(contingency_table, exact=False)
    # p_value = result.pvalue

    # table = [[0, only_a_correct_count],
    #          [only_b_correct_count, 0]]

    if n > 20 and all(count >= 5 for count in
                      [mutual_correct_count, only_a_correct_count, only_b_correct_count, both_wrong_count]):
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test: {'Significant' if p_value < alpha else 'Not Significant'}")
    else:
        print("Chi-square test: Sample size too small or expected frequencies < 5")

    #
    # result = mcnemar(contingency_table, exact=False)
    # p_value = result.pvalue
    # if p_value < 0.01:
    #     significance = "Highly significant"
    # elif p_value < 0.05:
    #     significance = "Significant"
    # else:
    #     significance = "Not significant"
    #
    # print(f"McNemar's test: {significance}")

    # table = [[0, only_a_correct_count],
    #          [only_b_correct_count, 0]]
    #
    # result = mcnemar(table, exact=True)
    # p_value = result.pvalue / 2
    # if only_b_correct_count <= only_a_correct_count:
    #     p_value = 1.0
    # if p_value < 0.01:
    #     significance = "Highly significant"
    # elif p_value < 0.05:
    #     significance = "Significant"
    # else:
    #     significance = "Not significant"
    # print(f"McNemar's test: {significance}")
    # print(f"McNemar p-value: {result.pvalue:.6f}")

    # observed_diff = np.mean(pred_list_b) - np.mean(pred_list_a)
    
    # # Combine all scores
    # all_scores = np.concatenate([pred_list_a, pred_list_b])
    # n_a = len(pred_list_a)
    
    # # Generate null distribution

    # n_permutations=1000
    # null_diffs = []
    # for _ in range(n_permutations):
    #     np.random.shuffle(all_scores)
    #     perm_a = all_scores[:n_a]
    #     perm_b = all_scores[n_a:]
    #     null_diffs.append(np.mean(perm_b) - np.mean(perm_a))
    
    # # One-tailed p-value
    # p_value = (np.sum(null_diffs >= observed_diff) + 1) / (n_permutations + 1)
    # if p_value < 0.01:
    #     significance = "Highly significant"
    # elif p_value < 0.05:
    #     significance = "Significant"
    # else:
    #     significance = "Not significant"

    # print(f"Permutation's test: {significance}")
    # print(f"Permutation p-value: {p_value:.6f}")


    if n > 20 and all(count >= 5 for count in [mutual_correct_count, only_a_correct_count, only_b_correct_count, both_wrong_count]):
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test: {'Significant' if p_value < alpha else 'Not Significant'}")

        # method_a_results = np.concatenate([
        #     np.ones(mutual_correct_count + only_a_correct_count),  # A correct
        #     np.zeros(only_b_correct_count + both_wrong_count)      # A incorrect
        # ])
    
        # method_b_results = np.concatenate([
        #     np.ones(mutual_correct_count + only_b_correct_count),  # B correct
        #     np.zeros(only_a_correct_count + both_wrong_count)      # B incorrect
        # ])
        # correlation, p_value_pearson = chi2_contingency(method_a_results, method_b_results)
        # print(f"Pearson correlation: r = {correlation:.4f}")
        # print(f"Pearson test: {'Significant' if p_value_pearson < alpha else 'Not Significant'} (p = {p_value_pearson:.6f})")
    else:
        print("Chi-square test: Sample size too small or expected frequencies < 5")
    #
    # print(f"McNemar's test: {significance}")
    # print(f"McNemar p-value: {result.pvalue:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single-oracle', 'prompt-methods', 'DocNum', 'DocError'])
    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')

    args = parser.parse_args()

    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3-5-turbo'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama-13b'
    elif args.model == 'llama-old-qa': args.model = 'llama2-13b'
    else: raise Exception('unknown model')

    if args.mode == 'single-oracle':
        single_result_path = f'../data/{args.dataset}/new_results/single_{args.model}_eval.json'
        oracle_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_eval.json'
        single_results = json.load(open(single_result_path, 'r'))["eval_records"]
        oracle_results = json.load(open(oracle_result_path, 'r'))["eval_records"]
        single_pred_list = [single_results[pid]['passed'] for pid in single_results]
        oracle_pred_list = [oracle_results[pid]['passed'] for pid in oracle_results]
        qid_list = list(single_results.keys())
        print('Single LLM v.s. Oracle RAG Prediction Distribution Difference:')
        pred_distribution_analysis(single_pred_list, oracle_pred_list, qid_list)

    elif args.mode == 'DocNum':
        ks = [1,3,5,10,15,20,30,40]
        qid_list = list(single_results.keys())
        if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            single_pred_list = [single_results[pid]['passed'] for pid in single_results]
            oracle_pred_list = [oracle_results[pid]['passed'] for pid in oracle_results]
        else:
            single_pred_list = [single_results[pid]['has_answer'] for pid in single_results]
            oracle_pred_list = [oracle_results[pid]['has_answer'] for pid in oracle_results]
        print('Single LLM v.s. Oracle RAG Prediction Distribution Difference:')
        pred_distribution_analysis(single_pred_list, oracle_pred_list, qid_list)


    elif args.mode == 'DocNum':
        ks = [1, 3, 5, 10, 15, 20, 30, 40]
        # ks = [1,5,10,15,20]
        a_ks = ks[:-1]
        b_ks = ks[1:]
        for a_k, b_k in zip(a_ks, b_ks):
            a_result_path = f'../data/{args.dataset}/new_results/DocNum/{a_k}_{args.model}_eval.json'
            b_result_path = f'../data/{args.dataset}/new_results/DocNum/{b_k}_{args.model}_eval.json'
            a_results = json.load(open(a_result_path, 'r'))["eval_records"]
            b_results = json.load(open(b_result_path, 'r'))["eval_records"]
            a_pred_list = [a_results[pid]['has_answer'] for pid in a_results]
            b_pred_list = [b_results[pid]['has_answer'] for pid in b_results]
            print(f'\n\n{a_k} v.s. {b_k} RAG Prediction Distribution Difference:')
            pred_distribution_analysis(a_pred_list, b_pred_list)

    elif args.mode == 'DocError':
        ks = [3, 5, 7, 10, 13, 16, 20]
        base_result_path = f'../data/{args.dataset}/new_results/DocNum/1_{args.model}_eval.json'
        # base_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_eval.json'
        base_results = json.load(open(base_result_path, 'r'))["eval_records"]
        base_pred_list = [base_results[pid]['passed'] for pid in base_results]
        for k in ks:
            k_result_path = f'../data/{args.dataset}/new_results/DocNum/{k}_{args.model}_eval.json'
            k_results = json.load(open(k_result_path, 'r'))["eval_records"]
            k_pred_list = [k_results[pid]['passed'] for pid in k_results]
            print(f'\n\n1 v.s. {k} RAG Prediction Distribution Difference:')
            pred_distribution_analysis(base_pred_list, k_pred_list)

    elif args.mode == 'prompt-methods':
        if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
            baseline_result_path = f'../data/{args.dataset}/new_results/DocNum/5_{args.model}_eval.json'
        else:
            baseline_result_path = f'../data/{args.dataset}/new_results/DocNum/10_{args.model}_eval.json'
        baseline_results = json.load(open(baseline_result_path, 'r'))["eval_records"]
        baseline_pred_list = [baseline_results[pid]['has_answer'] for pid in baseline_results]
        prompt_methods = ['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve', 'self-refine', 'CoN']
        for method in prompt_methods:
            method_result_path = f'../data/{args.dataset}/new_results/Prompt/{method}_{args.model}_eval.json'
            method_results = json.load(open(method_result_path, 'r'))["eval_records"]
            method_pred_list = [method_results[pid]['has_answer'] for pid in method_results]
            print(f'zero-shot baseline prompt v.s. {method} Prediction Distribution Difference:')
            pred_distribution_analysis(baseline_pred_list, method_pred_list)

