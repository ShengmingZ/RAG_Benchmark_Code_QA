import json
import ast
from scipy.stats import pearsonr, wilcoxon
import numpy as np
import sys, platform
from scipy.spatial.distance import hamming
system = platform.system()
if system == 'Darwin':
    root_path = '/'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/RAG_Benchmark_Code_QA'
sys.path.insert(0, root_path)
from generator_deprecated.generate_utils import generate_config
from dataset_utils.conala_utils import ConalaLoader
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.DS1000_utils import DS1000Loader
from data_processing import results
from scipy import stats




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



def eval_vs_eval(dataset, baseline_datas, compared_datas):
    # data1_true_data2_true_count = 0
    only_baseline_false_count = 0
    only_baseline_true_count = 0
    # data1_false_data2_false_count = 0
    all_false_count = 0
    eval_records1 = baseline_datas['eval_records']
    eval_records2 = compared_datas['eval_records']
    baseline_predictions = []
    compared_predictions = []
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
    elif dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA':
        for key in eval_records1.keys():
            eval_records1[key] = eval_records1[key]['has_answer']
            eval_records2[key] = eval_records2[key]['has_answer']


    evals1, evals2 = [], []
    evals1_wo_false, evals2_wo_false = [], []
    for key in eval_records1.keys():
        baseline_predictions.append(eval_records1[key])
        compared_predictions.append(eval_records2[key])
        if eval_records1[key] == eval_records2[key]:
            if eval_records1[key] is True:
                # data1_true_data2_true_count += 1
                ...
            else:
                # data1_false_data2_false_count += 1
                all_false_count += 1

        else:
            if eval_records1[key] is True:
                only_baseline_true_count += 1
            else:
                only_baseline_false_count += 1

        if eval_records1[key] is True or eval_records2[key] is True:
            evals1_wo_false.append(eval_records1[key])
            evals2_wo_false.append(eval_records2[key])
        evals1.append(eval_records1[key])
        evals2.append(eval_records2[key])
    hamming_dist = hamming(evals1, evals2)
    hamming_dist_wo_false = hamming(evals1_wo_false, evals2_wo_false)

    # print('data1 True. data2 True', data1_true_data2_true_count / len(eval_records1))
    # print('data1 True, data2 False', data1_true_data2_false_count / len(eval_records1))
    # print('data1 False, data2 True', data1_false_data2_true_count / len(eval_records1))
    # print('data1 False, data2 False', data1_false_data2_false_count / len(eval_records1))
    # print('hamming dist: ', hamming_dist)
    # print('hamming dist without both False: ', hamming_dist_wo_false)

    # calculate McNemar's test
    baseline = np.array(baseline_predictions, dtype=bool)
    technique = np.array(compared_predictions, dtype=bool)

    # Fisher's Exact Test (for the 2x2 table)
    # We'll create the 2x2 table of discordant pairs
    b = np.sum((technique == 1) & (baseline == 0))  # Technique solves, Baseline fails
    c = np.sum((technique == 0) & (baseline == 1))  # Technique fails, Baseline solves

    # Exact binomial test on discordant pairs
    from scipy.stats import binom_test
    mcnemar_p = binom_test(b, n=b + c, p=0.5)

    # Fisher's exact test on discordant pairs
    _, fisher_p = stats.fisher_exact([[b, c], [c, b]])

    return hamming_dist, only_baseline_false_count/len(eval_records1), [mcnemar_p, fisher_p], baseline_predictions, compared_predictions



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



def count_if_llm_refuse_to_answer(args):
    gene_results = json.load(open(args.result_save_file))
    refuse_answer_count = 0
    if args.dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
        for result in gene_results:
            output = result['outputs'][0]
            if '<answer>' not in output:
                refuse_answer_count += 1
                print(output)
    else:
        for result in gene_results:
            output = result['outputs'][0]
            if '<code>' not in output:
                refuse_answer_count += 1
                print(output)
    print(refuse_answer_count)
    print(refuse_answer_count / len(gene_results))


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
    ret_doc_types = ['oracle', 'distracting', 'random', 'irrelevant_diff', 'irrelevant_dummy']
    qa_gpt_topks = ['top_1', 'top_20', 'top_40', 'top_60', 'top_80']
    qa_llama_topks = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
    code_gpt_topks = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
    code_llama_topks = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']

    # """perplexity vs performance retrieval recall analysis """
    # model_names = ['gpt', 'llama', 'gpt', 'llama']
    # metric_names = ['recall', 'recall', 'pass@1', 'pass@1']
    # dataset_names_list = [qa_dataset_names, qa_dataset_names, code_dataset_names, code_dataset_names]
    # datas = [results.qa_ret_recall_gpt_n_1, results.qa_ret_recall_llama_n_1, results.code_ret_recall_gpt_n_1, results.code_ret_recall_llama_n_1]
    # for model_name, metric_name, dataset_names, data in zip(model_names, metric_names, dataset_names_list, datas):
    #     for dataset_name in dataset_names:
    #         # perf_list = [data[dataset_name][ret_recall][metric_name] for ret_recall in ret_recalls]
    #         perf_list = ret_recalls
    #         perplexity_list = [data[dataset_name][ret_recall]['perplexity'] for ret_recall in ret_recalls]
    #         p_score, _ = pearsonr(perf_list, perplexity_list)
    #         p_score_dict[model_name][dataset_name] = round(p_score, 3)
    # print('p_score of perplexity and performance, retrieval recall analysis', p_score_dict)


    # """perplexity vs performance retrieval doc type analysis"""
    # model_names = ['gpt', 'llama', 'gpt', 'llama']
    # metric_names = ['recall', 'recall', 'pass@1', 'pass@1']
    # dataset_names_list = [qa_dataset_names, qa_dataset_names, code_dataset_names, code_dataset_names]
    # datas = [results.qa_ret_doc_type_gpt_n_1, results.qa_ret_doc_type_llama_n_1, results.code_ret_doc_type_gpt_n_1, results.code_ret_doc_type_llama_n_1]
    # for model_name, metric_name, dataset_names, data in zip(model_names, metric_names, dataset_names_list, datas):
    #     for dataset_name in dataset_names:
    #         perf_list = [data[dataset_name][doc_type][metric_name] for doc_type in ret_doc_types]
    #         perplexity_list = [data[dataset_name][doc_type]['perplexity'] for doc_type in ret_doc_types]
    #         p_score, _ = pearsonr(perf_list, perplexity_list)
    #         p_score_dict[model_name][dataset_name] = round(p_score, 3)
    # print('p_score of perplexity and performance, retrieval doc type analysis', p_score_dict)


    # """syntax error vs performance for retrieval recall analysis"""
    # model_names = ['gpt', 'llama']
    # metric_names = ['pass@1', 'pass@1']
    # dataset_names_list = [code_dataset_names, code_dataset_names]
    # datas = [results.code_ret_recall_gpt_n_1, results.code_ret_recall_llama_n_1]
    # for model_name, metric_name, dataset_names, data in zip(model_names, metric_names, dataset_names_list, datas):
    #     for dataset_name in dataset_names:
    #         perf_list = [data[dataset_name][ret_recall][metric_name] for ret_recall in ret_recalls]
    #         # perf_list = ret_recalls
    #         syntax_error_percent_list = [data[dataset_name][ret_recall]['semantic_error_percent'] for ret_recall in ret_recalls]
    #         p_score, _ = pearsonr(perf_list, syntax_error_percent_list)
    #         p_score_dict[model_name][dataset_name] = round(p_score,3)
    # print('p_score of syntax error percent and performance, retrieval recall analysis: \n', p_score_dict)


    """topk vs perplexity for retrieval doc selection analysis"""
    model_names = ['gpt', 'llama', 'gpt', 'llama']
    metric_names = ['recall', 'recall', 'pass@1', 'pass@1']
    dataset_names_list = [qa_dataset_names, qa_dataset_names, code_dataset_names, code_dataset_names]
    datas = [results.qa_ret_doc_selection_topk_gpt_n_1, results.qa_ret_doc_selection_topk_llama_n_1, results.code_ret_doc_selection_topk_gpt_n_1, results.code_ret_doc_selection_topk_llama_n_1]
    topks_list = [qa_gpt_topks, qa_llama_topks, code_gpt_topks, code_llama_topks]
    for model_name, metric_name, dataset_names, data, topks in zip(model_names, metric_names, dataset_names_list, datas, topks_list):
        for dataset_name in dataset_names:
            # perf_list = [data[dataset_name][topk][metric_name] for topk in topks]
            perf_list = [int(topk.replace('top_','')) for topk in topks]
            perplexity_list = [data[dataset_name][topk]['perplexity'] for topk in topks]
            p_score, _ = pearsonr(perf_list, perplexity_list)
            p_score_dict[model_name][dataset_name] = round(p_score, 3)
    print('p_score of perplexity and performance, retrieval doc type analysis', p_score_dict)



    return p_score_dict


def wilcoxon_test(dataset, eval_datas1, eval_datas2):
    eval_records1 = eval_datas1['eval_records']
    eval_records2 = eval_datas2['eval_records']

    preds1, preds2 = [], []
    if dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA':
        for key in eval_records1.keys():
            preds1.append(eval_records1[key]['f1'])
            preds2.append(eval_records2[key]['f1'])
    else:
        ...

    _, p_value = wilcoxon(preds1, preds2)

    print('p-value: ', p_value)


# def has_answer(dataset, eval_datas):
#     assert dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA'
#     eval_records = eval_datas['eval_records']
#     for key in eval_records.keys():
#         eval_records[key] = eval_records[key]['has_answer']
#
#     for eval_records in eval_records.values():


def calc_avg_syntax_error():
    # result_to_be_processed = results.code_ret_recall_llama_n_1
    result_to_be_processed = results.code_ret_doc_selection_topk_gpt_n_1
    syntax_errors = dict()
    for model_name in result_to_be_processed.keys():
        for ret_recall in result_to_be_processed[model_name].keys():
            if ret_recall not in syntax_errors.keys(): syntax_errors[ret_recall] = 0
            syntax_errors[ret_recall] += result_to_be_processed[model_name][ret_recall]['syntax_error_percent']
    for ret_recall in syntax_errors.keys():
        print(ret_recall, round(syntax_errors[ret_recall]/3, 3))


def get_avg_result_pl_analysis():
    datasets = ['conala', 'DS1000', 'pandas_numpy_eval']
    # dataset = datasets[0]
    types = ['oracle', 'distracting', 'retrieved_top', 'none', 'irrelevant']
    type = types[4]
    print(type)
    result = results.code_pl_analysis_gpt_n_1
    avg_result = dict()
    for dataset in datasets:
        for key in result[dataset][type].keys():
            if result[dataset][type][key] is None: continue
            if key not in avg_result: avg_result[key] = 0
            avg_result[key] += result[dataset][type][key]['pass@1']
    # baseline_result = avg_result[type]/3
    baseline_result = 0.477
    for key in avg_result.keys():
        avg_result[key]/=3
        avg_result[key] = dict(avg_result=round(avg_result[key],3), avg_improve=round((avg_result[key]-baseline_result)/baseline_result, 3))
    for key in avg_result.keys():
        print(key, avg_result[key])




if __name__ == '__main__':
    # get_avg_result_pl_analysis()

    # calc_avg_syntax_error()

    """count has answer"""
    # in_program_call = (
    #     '--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset NQ --retriever openai-embedding '
    #                        f'--analysis_type retrieval_doc_type --n 1 --ret_doc_type oracle')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas = json.load(open(eval_file))
    #
    # in_program_call = ('--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset NQ --retriever openai-embedding '
    #                    f'--analysis_type prompt_length --n 1 --pl_analysis oracle_4000')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas2 = json.load(open(eval_file))
    #
    # wilcoxon_test(args.dataset, eval_datas, eval_datas2)



    # datasets = ['NQ', 'TriviaQA', 'hotpotQA']
    # for dataset in datasets:
    # in_program_call = (
    #     f'--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset NQ --retriever openai-embedding '
    #     f'--analysis_type prompt_length --n 1 --pl_analysis random_4000')
    # args = generate_config(in_program_call)
    # count_if_llm_refuse_to_answer(args)


    # calc_pearson_r()


    """compare 2 prediction distributions"""
    # datasets = ['NQ', 'TriviaQA', 'hotpotQA']
    datasets = ['conala', 'DS1000', 'pandas_numpy_eval']
    evals = ['3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
    if datasets == ['NQ', 'TriviaQA', 'hotpotQA']: model = 'llama2-13b-chat'
    else: model = 'codellama-13b-instruct'
    # model = 'gpt-3.5-turbo-0125'
    for dataset in datasets:
        predictions_list = []
        print(dataset)
        for eval in evals:
            in_program_call = (f'--action eval_pred --model {model} --temperature 0.0 --dataset {dataset} --retriever openai-embedding '
                               f'--analysis_type retrieval_doc_selection --n 1 --doc_selection_type top_10')
            args = generate_config(in_program_call)
            eval_file = args.result_save_file.replace('.json', '_eval.json')
            eval_datas = json.load(open(eval_file)) # baseline

            if eval == 'self-consistency':
                in_program_call = (
                    f'--action eval_pred --model {model} --temperature 1.0 --dataset {dataset} --retriever openai-embedding '
                    f'--analysis_type prompt_method --n 10 --prompt_type {eval}')
            else:
                in_program_call = (f'--action eval_pred --model {model} --temperature 0.0 --dataset {dataset} --retriever openai-embedding '
                                   f'--analysis_type prompt_method --n 1 --prompt_type {eval}')
            args = generate_config(in_program_call)
            eval_file = args.result_save_file.replace('.json', '_eval.json')
            eval_datas2 = json.load(open(eval_file))    # prompt method

            # ret_eval_vs_eval(eval_datas)

            hamming_dist, percentage_only_correct_data2, p_value, baseline_preds, prediction_preds = eval_vs_eval(args.dataset, eval_datas, eval_datas2)
            if dataset not in ['NQ', 'TriviaQA', 'hotpotQA'] or args.prompt_type == 'RaR':
                predictions_list.append(prediction_preds)
            print(f"0shot False but {eval} True percent: {round(percentage_only_correct_data2,4)}  |  p-value: {p_value}")
            # print(f"RAG false LLM true percent: {round(eval1_false_eval2_true, 3)}")

        add_up_unique_samples_count = 0
        for i, baseline_pred in enumerate(baseline_preds):
            if baseline_pred is False:
                for j in range(len(predictions_list)):
                    if predictions_list[j][i] is True:
                        add_up_unique_samples_count += 1
                        break

        print('add up percentage: ', add_up_unique_samples_count / len(baseline_preds))

    """wilcoxon test"""
    # in_program_call = (
    #     '--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset NQ --retriever openai-embedding '
    #                        f'--analysis_type retrieval_doc_type --n 1 --ret_doc_type oracle')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas = json.load(open(eval_file))
    #
    # in_program_call = ('--action eval_pred --model gpt-3.5-turbo-0125 --temperature 0.0 --dataset NQ --retriever openai-embedding '
    #                    f'--analysis_type prompt_length --n 1 --pl_analysis oracle_4000')
    # args = generate_config(in_program_call)
    # eval_file = args.result_save_file.replace('.json', '_eval.json')
    # eval_datas2 = json.load(open(eval_file))
    #
    # wilcoxon_test(args.dataset, eval_datas, eval_datas2)


    """calc syntax, semantic error"""
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



    """perplexity cliffs_delta"""
    # data_list = []



    # sample_indices = np.random.choice(2000, 200, replace=False)
    #
    # # Sample datasets at those indices
    # sampled_datasets = [
    #     [dataset[idx] for idx in sample_indices]
    #     for dataset in data_list
    # ]
    # data_list = sampled_datasets

    # results = []
    #
    # import numpy as np
    # from scipy import stats
    #
    # # Iterate through consecutive pairs of data
    # for i in range(len(data_list) - 1):
    #     # Perform Mann-Whitney U test
    #     statistic, p_value = stats.mannwhitneyu(
    #         data_list[i],
    #         data_list[i + 1],
    #         alternative='two-sided'  # two-sided test for difference in distribution
    #     )
    #
    #     # Determine significance (typically using alpha = 0.05)
    #     significance = "Significant" if p_value < 0.05 else "Not Significant"
    #
    #
    #     # cliff delta
    #     data1 = np.asarray(data_list[i])
    #     data2 = np.asarray(data_list[i+1])
    #     greater = 0
    #     lesser = 0
    #     for x in data1:
    #         for y in data2:
    #             if x > y:
    #                 greater += 1
    #             elif x < y:
    #                 lesser += 1
    #     total_comparisons = len(data1) * len(data2)
    #     delta = (greater - lesser) / total_comparisons
    #
    #     # Interpret the magnitude of the effect
    #     if abs(delta) < 0.147:
    #         interpretation = "negligible"
    #     elif abs(delta) < 0.33:
    #         interpretation = "small"
    #     elif abs(delta) < 0.474:
    #         interpretation = "medium"
    #     else:
    #         interpretation = "large"
    #
    #
    #
    #     results.append({
    #         'Pair': f'{i*0.2} vs {(i + 1)*0.2}',
    #         'cliff_delta': delta,
    #         'effect_size_interpretation': interpretation,
    #         'Statistic': statistic,
    #         'P-value': p_value,
    #         'Significance': significance,
    #         'sample_size': (len(data_list[i]), len(data_list[i+1])),
    #         'mean_perplexity': (np.mean(data_list[i]), np.mean(data_list[i+1])),
    #         'median_perplexity': (np.median(data_list[i]), np.median(data_list[i+1]))
    #     })
    #
    # print("Test Results:")
    # for result in results:
    #     print(f"Pair {result['Pair']}:")
    #     print(f"  Cliff delta: {result['cliff_delta']}")
    #     print(f"  Effect size interpretation: {result['effect_size_interpretation']}")
    #     print(f"  Statistic: {result['Statistic']}")
    #     print(f"  P-value: {result['P-value']:.4f}")
    #     print(f"  Significance: {result['Significance']}")
    #     print(f"  Mean: {result['mean_perplexity']}")
    #     print(f"  Median: {result['median_perplexity']}")

