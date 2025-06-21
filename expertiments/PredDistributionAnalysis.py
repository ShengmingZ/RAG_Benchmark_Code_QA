import argparse
import json
import sys
sys.path.append('../../Code_RAG_Benchmark')




def pred_distribution_analysis(pred_list_a, pred_list_b):
    """
    assert pred_list_a and pred_list_b are a list of prediction, with True/False values
    :param pred_list_a:
    :param pred_list_b:
    :return:
    """
    assert len(pred_list_a) == len(pred_list_b)
    only_a_correct_count = 0
    only_b_correct_count = 0
    mutual_correct_count = 0
    for i in range(len(pred_list_a)):
        if pred_list_a[i] and pred_list_b[i]:
            mutual_correct_count += 1
        elif pred_list_a[i] and not pred_list_b[i]:
            only_a_correct_count += 1
        elif not pred_list_a[i] and pred_list_b[i]:
            only_b_correct_count += 1
    print("Percentage of only a correctly solve: ", only_a_correct_count/len(pred_list_a))
    print('Percentage of only b correctly solve: ', only_b_correct_count/len(pred_list_b))
    print('Percentage of mutual correctly solve: ', mutual_correct_count/len(pred_list_a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')
    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')
    parser.add_argument('--mode', required=True, choices=['single-oracle'])
    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')

    args = parser.parse_args()

    if args.model == 'openai-new': args.model = 'gpt-4o-mini'
    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'
    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'
    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'
    else: raise Exception('unknown model')

    if args.mode == 'single-oracle':
        single_result_path = f'../data/{args.dataset}/new_results/single_{args.model}_evals.json'
        oracle_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_evals.json'
        single_results = json.load(open(single_result_path, 'r'))["eval_records"]
        oracle_results = json.load(open(oracle_result_path, 'r'))["eval_records"]
        single_pred_list = [single_results[pid]['passed'] for pid in single_results]
        oracle_pred_list = [oracle_results[pid]['passed'] for pid in oracle_results]
        print('Single LLM v.s. Oracle RAG Prediction Distribution Difference:')
        pred_distribution_analysis(single_pred_list, oracle_pred_list)
