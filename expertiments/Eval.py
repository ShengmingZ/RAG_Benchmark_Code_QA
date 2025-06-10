import argparse
import sys
sys.path.append('../../Code_RAG_Benchmark')
from generator.pred_eval import pred_eval_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single', 'oracle', 'both'])

    args = parser.parse_args()

    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'
    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'
    else: raise Exception('unknown model')

    result_path = f'../data/{args.dataset}/new_results/single_{args.model}.json'

    pred_eval_new(dataset=args.dataset, result_path=result_path)

