import argparse
import sys
sys.path.append('..')
from generator.pred_eval import pred_eval_new


if __name__ == '__main__':
    model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                            "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                            "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--retriever', required=True, choices=['openai-embedding', 'BM25', 'miniLM'])
    parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'recall', 'DocNum', 'prompt', 'realistic_recall'])
    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')
    parser.add_argument('--k', type=int, default=1, help='Doc Num, only effective if mode is "DocNum"')
    parser.add_argument('--prompt', type=str, default='zero-shot', choices=['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve', 'self-refine', 'CoN'])

    args = parser.parse_args()

    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'
    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'
    else: raise Exception('unknown model')

    model_name_for_path = model_names_for_path[args.model]
    if args.recall == 0: args.recall = int(args.recall)
    if args.mode == 'single':
        result_path = f'../data/{args.dataset}/new_results/single_{model_name_for_path}.json'
    elif args.mode == 'oracle' or args.mode == 'recall' and args.recall == 1:
        result_path = f'../data/{args.dataset}/new_results/oracle_{model_name_for_path}.json'
    elif args.mode == 'recall':
        if args.recall == 0: args.recall = int(args.recall)
        result_path = f'../data/{args.dataset}/new_results/recall-{args.recall}_{model_name_for_path}.json'
    elif args.mode == 'DocNum':
        result_path = f'../data/{args.dataset}/new_results/DocNum/{args.k}_{model_name_for_path}.json'
    elif args.mode == 'prompt':
        result_path = f'../data/{args.dataset}/new_results/Prompt/{args.prompt}_{model_name_for_path}.json'
    elif args.mode == 'realistic_recall':
        result_path = f'../data/{args.dataset}/new_results/realistic_recall-{args.recall}_{model_name_for_path}.json'

    pred_eval_new(model=args.model, dataset=args.dataset, result_path=result_path, prompt_method=args.prompt)

