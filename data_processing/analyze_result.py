import json
import numpy as np
import sys, platform
from scipy.spatial.distance import hamming
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from generator.generate_utils import generate_config
from dataset_utils.conala_utils import ConalaLoader


def perplexity_analysis(results):
    perplexity = 0
    for result in results:
        logprobs = result['logprobs'][0]    # todo: only for n=1
        if len(logprobs) == 1: logprobs = logprobs[0]   # for llama
        perplexity += np.exp(-sum(logprobs)/len(logprobs))
    perplexity /= len(results)

    print(perplexity)
    return perplexity



def code_error_type(eval_datas):
    syntax_error_count = 0



def retrieval_vs_eval(eval_datas):
    # todo: for code, detect if api appears in LLM
    # todo: for qa: how to detect?
    retrieval_in_eval_count = 0
    retrieval_not_in_eval_count = 0


def eval_vs_eval(dataset, eval_datas1, eval_datas2):
    data1_true_data2_true_count = 0
    data1_false_data2_true_count = 0
    data1_true_data2_false_count = 0
    data1_false_data2_false_count = 0
    eval_records1 = eval_datas1['eval_records']
    eval_records2 = eval_datas2['eval_records']
    if dataset == 'conala':
        for key in eval_records1.keys():
            eval_records1[key] = eval_records1[key]['passed']
            eval_records2[key] = eval_records2[key]['passed']
    elif dataset == 'DS1000':
        for key in eval_records1.keys():
            eval_records1[key] = eval_records1[key][0]
            eval_records2[key] = eval_records2[key][0]
    elif dataset == 'pandas_numpy_eval':
        for key in eval_records1.keys():
            if 'passed' in eval_records1[key]: eval_records1[key] = True
            else: eval_records1[key] = False
            if 'passed' in eval_records2[key]: eval_records2[key] = True
            else: eval_records2[key] = False


    evals1, evals2 = [], []
    evals1_wo_false, evals2_wo_false = [], []
    for key in eval_records1.keys():
        if eval_records1[key] == eval_records2[key]:
            if eval_records1[key] is True:
                data1_true_data2_true_count += 1
            else:
                data1_false_data2_false_count += 1
        else:
            if eval_records1[key] is True:
                data1_true_data2_false_count += 1
            else:
                data1_false_data2_true_count += 1

        if eval_records1[key] is True or eval_records2[key] is True:
            evals1_wo_false.append(eval_records1[key])
            evals2_wo_false.append(eval_records2[key])
        evals1.append(eval_records1[key])
        evals2.append(eval_records2[key])
    hamming_dist = hamming(evals1, evals2)
    hamming_dist_wo_false = hamming(evals1_wo_false, evals2_wo_false)

    print('data1 True. data2 True', data1_true_data2_true_count / len(eval_records1))
    print('data1 True, data2 False', data1_true_data2_false_count / len(eval_records1))
    print('data1 False, data2 True', data1_false_data2_true_count / len(eval_records1))
    print('data1 False, data2 False', data1_false_data2_false_count / len(eval_records1))
    print('hamming dist: ', hamming_dist)
    print('hamming dist without both False: ', hamming_dist_wo_false)



def ret_eval_vs_eval(eval_datas):
    retrieval_records = eval_datas['retrieval_records']
    ret_eval_records = eval_datas['ret_eval_records']
    eval_records = eval_datas['eval_records']
    output_records = eval_datas['output_records']
    qs_list = ConalaLoader().load_qs_list()

    ret_eval_true_eval_false_count = 0
    ret_eval_true_eval_true_count = 0
    ret_eval_false_eval_true_count = 0
    ret_eval_false_eval_false_count = 0
    # todo: ret eval part true; eval part true
    for qs in qs_list:
        qs_id = qs['qs_id']
        eval = eval_records[qs_id]
        retrieval = retrieval_records[qs_id]
        ret_eval = ret_eval_records[qs_id]
        output = output_records[qs_id]

        if eval['passed'] is True:
            if True in ret_eval:
                ret_eval_true_eval_true_count += 1
            else:
                ret_eval_false_eval_true_count += 1
        else:
            if True in ret_eval:
                ret_eval_true_eval_false_count += 1
            else:
                ret_eval_false_eval_false_count += 1

    print('ret eval True, eval True percent', ret_eval_true_eval_true_count / len(qs_list))
    print('ret eval True, eval False percent', ret_eval_true_eval_false_count / len(qs_list))
    print('ret eval False, eval True percent', ret_eval_false_eval_true_count / len(qs_list))
    print('ret eval False, eval False percent', ret_eval_false_eval_false_count / len(qs_list))


if __name__ == '__main__':
    # in_program_call = ('--action eval_pred --model codellama-13b-instruct --temperature 0.0 --dataset conala --retriever openai-embedding '
    #                    '--analysis_type retrieval_recall --n 1 --ret_acc 1.0')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas = json.load(open(eval_file))
    #
    # in_program_call = ('--action eval_pred --model codellama-13b-instruct --temperature 0.0 --dataset conala --retriever openai-embedding '
    #                    '--analysis_type retrieval_doc_type --n 1 --ret_doc_type none')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas2 = json.load(open(eval_file))

    # ret_eval_vs_eval(eval_datas)

    # eval_vs_eval(args.dataset, eval_datas, eval_datas2)


    in_program_call = ('--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset conala --retriever openai-embedding '
                       '--analysis_type retrieval_doc_type --n 1 --ret_doc_type none')
    args = generate_config(in_program_call)
    results = json.load(open(args.result_save_file))[:1]
    perplexity_analysis(results)
