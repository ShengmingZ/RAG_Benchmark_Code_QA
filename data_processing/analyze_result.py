import json
import ast
from scipy.stats import pearsonr
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
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.DS1000_utils import DS1000Loader
from data_processing import results




def calc_perplexity(results):
    perplexity = 0
    for result in results:
        logprobs = result['logprobs'][0]    # todo: only for n=1
        if len(logprobs) == 1: logprobs = logprobs[0]   # for llama
        perplexity += np.exp(-sum(logprobs)/len(logprobs))
    perplexity /= len(results)

    # print('perplexity: ', round(perplexity,3))
    return perplexity


def count_semantic_error(dataset, eval_datas):
    if dataset == 'conala':
        loader = ConalaLoader()
    elif dataset == 'DS1000':
        loader = DS1000Loader()
    elif dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()

    oracle_docs_in_output_dict = dict()
    oracle_list = loader.load_oracle_list()
    output_records = eval_datas['output_records']
    for qs_id in output_records.keys():
        oracle_docs_in_output = list()
        for item in oracle_list:
            if item['qs_id'] == qs_id: oracle = item
        oracle_funcs = [item.rsplit('.', 1)[-1] for item in oracle['oracle_docs']]
        for func in oracle_funcs:
            if func in output_records[qs_id][0]:
                oracle_docs_in_output.append(True)
            else:
                oracle_docs_in_output.append(False)
        oracle_docs_in_output_dict[qs_id] = oracle_docs_in_output

    semantic_partial_correct_count = 0
    semantic_all_correct_count = 0
    semantic_error_count = 0  # whether use the desired functions
    for qs_id in oracle_docs_in_output_dict.keys():
        if True in oracle_docs_in_output_dict[qs_id]: semantic_partial_correct_count += 1
        if False not in oracle_docs_in_output_dict[qs_id]: semantic_all_correct_count += 1
        if True not in oracle_docs_in_output_dict[qs_id]: semantic_error_count += 1

    # print('semantic partial correct percentage: ', round(semantic_partial_correct_count/len(output_records),3))
    # print('semantic correct percentage: ', round(semantic_all_correct_count/len(output_records),3))
    # print('semantic error percentage: ', round(semantic_error_count/len(output_records),3))
    return semantic_error_count/len(output_records)



def count_syntax_error(dataset, eval_datas):
    syntax_error_count = 0
    output_records = eval_datas['output_records']
    programs = list()
    if dataset == 'conala':
        loader = ConalaLoader()
        unittests = json.load(open(loader.unittest_file, 'r'))
        for qs_id in output_records.keys():
            unittest = unittests[qs_id]
            program = f"{unittest['prompt']}{output_records[qs_id][0]}{unittest['suffix']}"
            programs.append(program)
    elif dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        unittests = json.load(open(loader.data_file, 'r'))
        for qs_id in output_records.keys():
            for item in unittests:
                if item['task_id'] == qs_id: unittest = item
            program = unittest['prompt'] + output_records[qs_id][0]
            programs.append(program)
    elif dataset == 'DS1000':
        loader = DS1000Loader()
        unittests = json.load(open(loader.sampled_data_file, 'r'))
        for qs_id in output_records.keys():
            for item in unittests:
                if item['qs_id'] == qs_id: unittest = item
            program = unittest['code_context'].replace('[insert]', output_records[qs_id][0])
            programs.append(program)

    for program in programs:
        try:
            ast.parse(program)
        except:
            syntax_error_count += 1

    # print('Syntax error percentage: ', round(syntax_error_count / len(output_records),3))
    return syntax_error_count / len(output_records)



def calc_retrieval_consistency(eval_datas):
    # todo: for qa: detect if answer in the corpus
    retrieval_in_output_count = 0
    retrieval_not_in_output_count = 0
    retrieval_consistency = 0
    output_records = eval_datas['output_records']
    retrieval_records = eval_datas['retrieval_records']
    retrieval_in_output_dict = dict()

    if next(iter(retrieval_records.values())) is None: return None

    for key in retrieval_records.keys():
        retrieval_in_output = list()
        retrieved_funcs = [item.rsplit('.', 1)[-1] for item in retrieval_records[key]]
        output = output_records[key][0]     # todo: now only for n=1
        for func in retrieved_funcs:
            if func in output:
                retrieval_in_output.append(True)
            else:
                retrieval_in_output.append(False)
        retrieval_in_output_dict[key] = retrieval_in_output


    for key in retrieval_in_output_dict.keys():
        retrieval_consistency += retrieval_in_output_dict[key].count(True)
        if True in retrieval_in_output_dict[key]: retrieval_in_output_count += 1
        else: retrieval_not_in_output_count += 1

    # print('retrieval consistency: ', round(retrieval_consistency/len(output_records),3))
    # print('retrieval in output count: ', retrieval_in_output_count)
    # print('retrieval not in output count: ', retrieval_not_in_output_count)
    return retrieval_consistency/len(output_records)


def retrieval_consistency_vs_eval(dataset, eval_datas):
    retrieval_in_output_dict = calc_retrieval_consistency(eval_datas)
    eval_records = eval_datas['eval_records']
    if dataset == 'conala':
        for key in eval_records.keys():
            eval_records[key] = eval_records[key]['passed']
    elif dataset == 'DS1000':
        for key in eval_records.keys():
            eval_records[key] = eval_records[key][0]
    elif dataset == 'pandas_numpy_eval':
        for key in eval_records.keys():
            if 'passed' in eval_records[key]: eval_records[key] = True
            else: eval_records[key] = False

    retrieval_consist_eval_true_count = 0
    retrieval_not_consist_eval_true_count = 0
    retrieval_consist_eval_false_count = 0
    retrieval_not_consist_eval_false_count = 0
    for key in retrieval_in_output_dict.keys():
        if True in retrieval_in_output_dict[key]:
            if eval_records[key] is True:
                retrieval_consist_eval_true_count += 1
            else:
                retrieval_consist_eval_false_count += 1
        else:
            if eval_records[key] is True:
                retrieval_not_consist_eval_true_count += 1
            else:
                retrieval_not_consist_eval_false_count += 1

    print('retrieval consist eval True percentage = ', retrieval_consist_eval_true_count / len(eval_records))
    print('retrieval not consist eval True percentage = ', retrieval_not_consist_eval_true_count / len(eval_records))
    print('retrieval consist eval False percentage = ', retrieval_consist_eval_false_count / len(eval_records))
    print('retrieval not consist eval False percentage = ', retrieval_not_consist_eval_false_count / len(eval_records))



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


def analyze_results_for_code(dataset, eval_datas):
    ret_consistency = calc_retrieval_consistency(eval_datas)
    syntax_error = count_syntax_error(dataset, eval_datas)
    semantic_error = count_semantic_error(dataset, eval_datas)
    return ret_consistency, syntax_error, semantic_error


def calc_pearson_r():
    p_score_dict = dict()
    p_score_dict['llama'], p_score_dict['gpt'] = dict(), dict()
    dataset_names = ['NQ', 'TriviaQA', 'hotpotQA', 'conala', 'DS1000', 'pandas_numpy_eval']
    qa_dataset_names, code_dataset_names = dataset_names[:3], dataset_names[3:]
    ret_recalls = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    """perplexity vs performance retrieval recall analysis """
    # model_names = ['gpt', 'llama', 'gpt', 'llama']
    # metric_names = ['recall', 'recall', 'pass@1', 'pass@1']
    # dataset_names_list = [make_graph.qa_dataset_names, make_graph.qa_dataset_names, make_graph.code_dataset_names, make_graph.code_dataset_names]
    # datas = [results.qa_ret_recall_gpt_n_1, results.qa_ret_recall_llama_n_1, results.code_ret_recall_gpt_n_1, results.code_ret_recall_llama_n_1]
    # for model_name, metric_name, dataset_names, data in zip(model_names, metric_names, dataset_names_list, datas):
    #     for dataset_name in dataset_names:
    #         perf_list = [data[dataset_name][ret_recall][metric_name] for ret_recall in make_graph.ret_recalls]
    #         perplexity_list = [data[dataset_name][ret_recall]['perplexity'] for ret_recall in make_graph.ret_recalls]
    #         p_score, _ = pearsonr(perf_list, perplexity_list)
    #         p_score_dict[model_name][dataset_name] = p_score
    # print('p_score of perplexity and performance, retrieval recall analysis', p_score_dict)


    """syntax error vs performance for retrieval recall analysis"""
    model_names = ['gpt', 'llama']
    metric_names = ['pass@1', 'pass@1']
    dataset_names_list = [code_dataset_names, code_dataset_names]
    datas = [results.code_ret_recall_gpt_n_1, results.code_ret_recall_llama_n_1]
    for model_name, metric_name, dataset_names, data in zip(model_names, metric_names, dataset_names_list, datas):
        for dataset_name in dataset_names:
            perf_list = [data[dataset_name][ret_recall][metric_name] for ret_recall in ret_recalls]
            # perf_list = make_graph.ret_recalls
            syntax_error_percent_list = [data[dataset_name][ret_recall]['semantic_error_percent'] for ret_recall in ret_recalls]
            p_score, _ = pearsonr(perf_list, syntax_error_percent_list)
            p_score_dict[model_name][dataset_name] = round(p_score,3)
    print('p_score of syntax error percent and performance, retrieval recall analysis: \n', p_score_dict)


    return p_score_dict


if __name__ == '__main__':
    calc_pearson_r()

    """
    in_program_call = ('--action eval_pred --model codellama-13b-instruct --temperature 0.0 --dataset conala --retriever openai-embedding '
                       '--analysis_type retrieval_recall --n 1 --ret_acc 1.0')
    args = generate_config(in_program_call)
    eval_file = args.result_save_file.replace('.json', '_eval.json')
    eval_datas = json.load(open(eval_file))

    in_program_call = ('--action eval_pred --model codellama-13b-instruct --temperature 0.0 --dataset conala --retriever openai-embedding '
                       '--analysis_type retrieval_doc_type --n 1 --ret_doc_type none')
    args = generate_config(in_program_call)
    eval_file = args.result_save_file.replace('.json', '_eval.json')
    eval_datas2 = json.load(open(eval_file))

    ret_eval_vs_eval(eval_datas)

    eval_vs_eval(args.dataset, eval_datas, eval_datas2)
    """


    # for ret_acc in [1.0, 0.8, 0.6, 0.4, 0.2, 0]:
    #     in_program_call = (
    #         '--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset pandas_numpy_eval --retriever openai-embedding '
    #         f'--analysis_type retrieval_recall --n 1 --ret_acc {ret_acc}')
    #     args = generate_config(in_program_call)
    #     results = json.load(open(args.result_save_file))
    #     eval_file = args.result_save_file.replace('.json', '_eval.json')
    #     eval_datas = json.load(open(eval_file))
    #
    #     perplexity = calc_perplexity(results)
    #
    #     retrieval_consistency = calc_retrieval_consistency(eval_datas)
    #     # retrieval_consistency_vs_eval(args.dataset, eval_datas)
    #
    #     syntax_error_count = count_syntax_error(args.dataset, eval_datas)
    #     semantic_error_count = count_semantic_error(args.dataset, eval_datas)
    #
    #     print(dict(perplexity=round(perplexity,3), retrieval_consistency=round(retrieval_consistency,3), syntax_error_percent=round(syntax_error_count,3), semantic_error_percent=round(semantic_error_count,3)))

