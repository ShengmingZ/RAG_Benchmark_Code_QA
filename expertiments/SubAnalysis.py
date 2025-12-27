# def calculate_average_performance(domain, p1, p2, p3):
#     if domain == 'NLP':
#         return round((p1 + p2 + p3) / 3, 3)
#     elif domain == 'SE':
#         return round((p1*84 + p2*157 + p3*167)/408, 3)
#
#
# if __name__ == '__main__':
#     print(calculate_average_performance('SE', 0.274,
# 0.166,
# 0.551,
#
#
# ))


import numpy as np
from typing import List, Tuple
import argparse


def bootstrap_accuracy(
        predictions: List[bool],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate bootstrap statistics for a list of boolean predictions.

    Args:
        predictions: List of True/False values (True = correct, False = incorrect)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI (default 0.95 for 95% CI)

    Returns:
        mean: Bootstrap mean accuracy
        std: Bootstrap standard deviation
        ci: Confidence interval (lower, upper)
    """
    predictions = np.array(predictions, dtype=float)
    n_samples = len(predictions)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(predictions, size=n_samples, replace=True)
        bootstrap_scores.append(np.mean(sample))

    bootstrap_scores = np.array(bootstrap_scores)

    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)

    # Confidence interval using percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return mean, std, (ci_lower, ci_upper)


# Example usage
if __name__ == "__main__":
    # Example prediction list
    # model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
    #                         "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
    #                         "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
    #                         "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    # parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    # parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'recall', 'DocNum', 'prompt'])
    # parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')
    # parser.add_argument('--k', type=int, default=1, help='Doc Num, only effective if mode is "DocNum"')
    # parser.add_argument('--prompt', type=str, default='zero-shot',
    #                     choices=['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve',
    #                              'self-refine', 'CoN'])
    #
    # args = parser.parse_args()
    #
    # if args.model == 'openai-new':
    #     args.model = 'gpt-4o-mini'
    # elif args.model == 'openai-old':
    #     args.model = 'gpt-3.5-turbo-0125'
    # elif args.model == 'llama-new':
    #     args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    # elif args.model == 'llama-old-code':
    #     args.model = 'codellama/CodeLlama-13b-Instruct-hf'
    # elif args.model == 'llama-old-qa':
    #     args.model = 'meta-llama/Llama-2-13b-chat-hf'
    # else:
    #     raise Exception('unknown model')
    #
    # model_name_for_path = model_names_for_path[args.model]
    # if args.mode == 'single':
    #     result_path = f'../data/{args.dataset}/new_results/single_{model_name_for_path}.json'
    # elif args.mode == 'oracle' or args.mode == 'recall' and args.recall == 1:
    #     result_path = f'../data/{args.dataset}/new_results/oracle_{model_name_for_path}.json'
    # elif args.mode == 'recall':
    #     if args.recall == 0: args.recall = int(args.recall)
    #     result_path = f'../data/{args.dataset}/new_results/recall-{args.recall}_{model_name_for_path}.json'
    # elif args.mode == 'DocNum':
    #     result_path = f'../data/{args.dataset}/new_results/DocNum/{args.k}_{model_name_for_path}.json'
    # elif args.mode == 'prompt':
    #     result_path = f'../data/{args.dataset}/new_results/Prompt/{args.prompt}_{model_name_for_path}.json'
    #
    # import json
    # eval_results = json.load(open(result_path.replace('.json', '_eval.json')))["eval_records"]
    # predictions = [eval_results[key]['passed'] for key in eval_results]
    #
    # mean, std, ci = bootstrap_accuracy(predictions, n_bootstrap=1000)
    #
    # print(f"Sample size: {len(predictions)}")
    # print(f"Raw accuracy: {sum(predictions) / len(predictions):.1%}")
    # print(f"Bootstrap mean: {mean:.1%}")
    # print(f"Bootstrap std: {std:.1%}")
    # print(f"95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
    # print(f"\nFor paper: {mean:.1%} ± {std:.1%}")


    file1 = '/home/zhaoshengming/RAG_Benchmark_Code_QA/data/DS1000/new_results/DocNum/5_gpt-4o-mini_eval.json'
    file2 = '/home/zhaoshengming/RAG_Benchmark_Code_QA/data/DS1000/new_results/Prompt/CoT_gpt-4o-mini_eval.json'
    import json
    prediction_list1 = json.load(open(file1))["eval_records"]
    prediction_list2 = json.load(open(file2))["eval_records"]

    for key in prediction_list1:
        if prediction_list1[key]['passed'] and not prediction_list2[key]['passed']:
            print(key)
