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

    result_path = f'../data/{args.dataset}/new_results/single_{args.model}.json'

    pred_eval_new(dataset=args.dataset, result_path=result_path)

