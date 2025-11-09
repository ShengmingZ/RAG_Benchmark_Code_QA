import argparse
import json
import sys
import math
sys.path.append('..')


def calc_ppl(logprobs_list):
    perplexities = []
    for logprobs in logprobs_list:
        # Calculate perplexity for this sequence
        # Perplexity = exp(-1/N * sum(log_probs))
        avg_log_prob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_log_prob)
        perplexities.append(perplexity)

    # Return average perplexity across all sequences
    return sum(perplexities) / len(perplexities)



if __name__ == '__main__':
    model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                            "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                            "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'DocNum'])
    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')

    args = parser.parse_args()

    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama-13b'
    elif args.model == 'llama-old-qa': args.model = 'llama2-13b'
    else: raise Exception('unknown model')

    if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        ks = [1, 3, 5, 7, 10, 13, 16, 20]
    else:
        ks = [1, 3, 5, 10, 15, 20, 30, 40]
        if args.dataset == 'llama2-13b': ks = [1, 3, 5, 10, 15, 20]

    model_name_for_path = model_names_for_path[args.model]
    if args.mode == 'DocNum':
        for k in ks:
            result_path = f'../data/{args.dataset}/new_results/DocNum/{k}_{model_name_for_path}.json'
            logprobs_list = [item['logprobs'] for item in json.load(open(result_path, 'r'))]
            print(f'for {args.dataset} k={k}, AVG PPL: {round(calc_ppl(logprobs_list), 5)}')