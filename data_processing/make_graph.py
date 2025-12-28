import matplotlib.pyplot as plt
from matplotlib import gridspec

import results
import numpy as np


dataset_names = ['NQ', 'TriviaQA', 'hotpotQA', 'conala', 'DS1000', 'pandas_numpy_eval']
auth_dataset_names = ['NQ', 'TriviaQA', 'HotpotQA', 'CoNaLa', 'DS1000', 'PNE']
qa_dataset_names, code_dataset_names = dataset_names[:3], dataset_names[3:]
auth_qa_dataset_names, auth_code_dataset_names = auth_dataset_names[:3], auth_dataset_names[3:]
# retriever_names = ['BM25', 'miniLM', 'openai-embedding', 'contriever', 'codeT5']
# top_ks = [1, 3, 5, 10, 20, 50, 100]
ret_recalls = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# # ret_doc_types = ["oracle", "retrieved", "distracting", "random", "irrelevant_dummy", "irrelevant_diff", "none"]
# ret_doc_types = ["oracle", "distracting", "random", "irrelevant_dummy", "irrelevant_diff"]
# qa_gpt_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20', 'top_40', 'top_60', 'top_80']
# qa_llama_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
# code_gpt_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
# code_llama_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']


# percentage_of_mistakes_as_documents_added_gpt = [
#     [0, 3.6, 4.3, 5.1, 5.8, 6.4, 7.1],
#     [0, 2.4, 2.9, 3.4, 3, 3.4, 4],
#     [0, 4.4, 5.2, 5.9, 6.1, 7.6, 7.9],
#     [0, 3.5, 4.1, 4.8, 5.0, 5.8, 6.3],
#     [0, 4.8, 6, 7.1],
#     [0, 5.1, 6.4, 8.3],
#     [0, 4.2, 5.4, 5.4],
#     [0, 4.7, 5.9, 6.9]
# ]
#
# percentage_of_mistakes_as_documents_added_llama = [
#     [0, 5.4, 6.8, 9],
#     [0, 3, 3.2, 4.3],
#     [0, 4, 5.9, 7.4],
#     [0, 4.1, 5.3, 6.9],
#     [0, 4.8, 8.3, 9.5],
#     [0, 1.9, 2.5, 4.5],
#     [0, 3.6, 4.2, 6.6],
#     [0, 3.4, 5, 6.9]
# ]

# percentage_of_mistakes_as_documents_added_gpt = [
#     [3.6, 4.3, 5.1, 5.8, 6.4, 7.1],
#     [2.4, 2.9, 3.4, 3, 3.4, 4],
#     [4.4, 5.2, 5.9, 6.1, 7.6, 7.9],
#     [3.5, 4.1, 4.8, 5.0, 5.8, 6.3],
#     [4.8, 6, 7.1],
#     [5.1, 6.4, 8.3],
#     [4.2, 5.4, 5.4],
#     [4.7, 5.9, 6.9]
# ]
#
# percentage_of_mistakes_as_documents_added_llama = [
#     [5.4, 6.8, 9],
#     [3, 3.2, 4.3],
#     [4, 5.9, 7.4],
#     [4.1, 5.3, 6.9],
#     [4.8, 8.3, 9.5],
#     [1.9, 2.5, 4.5],
#     [3.6, 4.2, 6.6],
#     [3.4, 5, 6.9]
# ]


# percentage_of_correct_prompt_method_gpt_qa = {
#     'NQ': [5.4, 18.3, 5.3, 5.4, 7, 6.5, 6.9, 7.7],
#     'TriviaQA': [3.5, 9.1, 4.1, 3.8, 5, 4.2, 5.1, 4.3],
#     'hotpotQA': [5.4, 17.8, 8.7, 8.4, 10.1, 12.2, 6.9, 8.1]
# }
#
# percentage_of_correct_prompt_method_gpt_code = {
#     'conala': [11.9, 3.6, 7.1, 4.8, 8.3, 0, 11.9, 0],
#     'DS1000': [7, 7, 8.9, 8.9, 6.4, 6.4, 10.2, 5.1],
#     'pandas_numpy_eval': [9, 9.6, 11.4, 9.6, 8.4, 7.8, 10.2, 6],
# }
#
# percentage_of_correct_prompt_method_llama_qa = {
#     'NQ': [4, 9.3, 4.1, 3.6, 5.4, 5.7, 4.5, 4.9],
#     'TriviaQA': [2.1, 4.9, 1.6, 1.7, 2.5, 3.5, 2.9, 3.1],
#     'hotpotQA': [3.7, 9.2, 6, 6.1, 7, 5.1, 5, 5.2]
# }
#
# percentage_of_correct_prompt_method_llama_code = {
#     'conala': [8.3, 2.4, 7.1, 8.3, 13.1, 3.6, 7.1, 2.4],
#     'DS1000': [17.2, 6.4, 14.6, 11.5, 13.4, 5.7, 12.7, 2.5],
#     'pandas_numpy_eval': [8.4, 7.8, 7.8, 6.6, 9, 5.4, 7.8, 6],
# }

# use this over the corresponding performance, get the percentage of instance only correct in prompt method over the whole correct ones


# def make_doc_selection_percentage_of_mistakes():
#     graph_name = 'select_topk_percentage_of_mistakes.pdf'
#
#     plt.style.use('ggplot')
#     fig = plt.figure(figsize=(18, 4))
#     gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.75, 1])
#     # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[0])
#     ax3 = plt.subplot(gs[3])
#     ax4 = plt.subplot(gs[2])
#     # colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(qa_dataset_names)))
#     # colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))
#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
#     colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
#     markers = ['D', 'o', '^', '*']
#
#     qa_dataset_names.append('avg of QA')
#     code_dataset_names.append('avg of code')
#     auth_qa_dataset_names.append('avg of QA tasks')
#     auth_code_dataset_names.append('avg of code tasks')
#
#     x = [0, 5, 10, 20, 30, 40]
#     axs = [ax1, ax2, ax3, ax4]
#     code_llama_datas = percentage_of_mistakes_as_documents_added_llama[4:]
#     qa_llama_datas = percentage_of_mistakes_as_documents_added_llama[:4]
#     code_gpt_datas = percentage_of_mistakes_as_documents_added_gpt[4:]
#     qa_gpt_datas = percentage_of_mistakes_as_documents_added_gpt[:4]
#     perf_datas_list = [code_llama_datas, qa_llama_datas, code_gpt_datas, qa_gpt_datas]
#     topk_list = [code_llama_doc_selection_types[2:], qa_llama_doc_selection_types[2:], code_gpt_doc_selection_types[2:], qa_gpt_doc_selection_types[2:]]
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
#         if ax_idx % 2 == 0:
#             dataset_names = code_dataset_names
#             auth_dataset_names = auth_code_dataset_names
#             tmp_colors = colors[4:]
#         else:
#             dataset_names = qa_dataset_names
#             auth_dataset_names = auth_qa_dataset_names
#             tmp_colors = colors[:4]
#         for idx, dataset_name in enumerate(dataset_names):
#             if ax_idx == 3:
#                 ax.plot(x, perf_datas[idx], marker=markers[idx],
#                         markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
#             else:
#                 ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx],
#                         markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
#         ax.set_ylabel('Instances Percentage', fontsize=16)
#         ax.set_xlabel('Top K Documents', fontsize=16)
#         if ax_idx == 3:
#             ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
#             ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
#         else:
#             ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
#             ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
#         # if ax_idx in [1, 3]:
#         #     ax.set_yticks([0.3, 0.5, 0.7, 0.9])
#         #     ax.set_yticklabels([0.3, 0.5, 0.7, 0.9], fontsize=12)
#         # elif ax_idx == 0:
#         #     ax.set_yticks([0.1, 0.3, 0.5, 0.7])
#         #     ax.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize=12)
#         # elif ax_idx == 2:
#         #     ax.set_yticks([0.2, 0.4, 0.6, 0.8])
#         #     ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=12)
#         ax.set_yticks([0, 2, 4, 6, 8, 10])
#         ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontsize=16)
#         if ax_idx == 0:
#             ax.set_title('b. CodeLlama-13B, Code Tasks', fontsize=14)
#         elif ax_idx == 1:
#             ax.set_title('a. Llama2-13B, QA Tasks', fontsize=14)
#         elif ax_idx == 2:
#             ax.set_title('d. GPT-3.5, Code Tasks', fontsize=14)
#         else:
#             ax.set_title('c. GPT-3.5, QA Tasks', fontsize=14)
#
#     ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
#     ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
#     handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
#     fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()


# def make_doc_selection_topk_syntax_semantic_error():
#     graph_name = 'select_topk_syntax_error.pdf'
#     llama_syntax_errors = []
#     gpt_syntax_errors = []
#     llama_semantic_errors = []
#     gpt_semantic_errors = []
#     for dataset_name in code_dataset_names:
#         llama_syntax_errors.append(
#             [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type]['syntax_error_percent'] for doc_selection_type in code_llama_doc_selection_types])
#         gpt_syntax_errors.append(
#             [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type]['syntax_error_percent'] for doc_selection_type in code_gpt_doc_selection_types])
#         llama_semantic_errors.append(
#             [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type]['semantic_error_percent'] for doc_selection_type in code_llama_doc_selection_types])
#         gpt_semantic_errors.append(
#             [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type]['semantic_error_percent'] for doc_selection_type in code_gpt_doc_selection_types])
#
#     def get_avg_data(perf_datas, doc_selection_types, dataset_names):
#         avg_perf_datas = [0] * len(doc_selection_types)
#         for data in perf_datas:
#             avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
#         avg_perf_datas = [item / len(dataset_names) for item in avg_perf_datas]
#         return avg_perf_datas
#
#     datas_list = [llama_syntax_errors, gpt_syntax_errors, llama_semantic_errors, gpt_semantic_errors]
#     topk_list = [code_llama_doc_selection_types, code_gpt_doc_selection_types, code_llama_doc_selection_types, code_gpt_doc_selection_types]
#     for datas, topk in zip(datas_list, topk_list):
#         datas.append(get_avg_data(datas, topk, code_dataset_names))
#     qa_dataset_names.append('avg syntax error')
#     code_dataset_names.append('avg semantic error')
#
#     plt.style.use('ggplot')
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
#     colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(code_dataset_names)))
#     colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))
#
#     axs = [ax1, ax2, ax3, ax4]
#     error_datas_list = [llama_syntax_errors, gpt_syntax_errors, llama_semantic_errors, gpt_semantic_errors]
#     topk_list = [code_llama_doc_selection_types, code_gpt_doc_selection_types, code_llama_doc_selection_types, code_gpt_doc_selection_types]
#     for ax_idx, (ax, datas) in enumerate(zip(axs, error_datas_list)):
#         dataset_names = code_dataset_names
#         for idx, dataset_name in enumerate(dataset_names):
#             ax.plot(topk_list[ax_idx], datas[idx], marker='o', linestyle='-', label=dataset_name, color=colors1[idx])
#         ax.set_ylabel('error percentage')
#         ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#         if ax_idx == 0:
#             ax.set_title('syntax error, llama2-13b', fontsize=10)
#         elif ax_idx == 1:
#             ax.set_title('syntax error, gpt-3.5', fontsize=10)
#         elif ax_idx == 2:
#             ax.set_title('semantic error, llama2-13b', fontsize=10)
#         else:
#             ax.set_title('syntax error, gpt-3.5', fontsize=10)
#
#     ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
#     handles, labels = ax3_handles, ax3_labels
#     fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.1))
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()
#
def make_doc_selection_topk_analysis():
    graph_name = 'select_topk_analysis.pdf'
    qa_gpt_perf_datas = []
    qa_metric = 'has_answer'
    code_metric = 'pass@1'
    for dataset_name in qa_dataset_names:
        qa_gpt_perf_datas.append(
            [results.qa_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][qa_metric] for doc_selection_type in qa_gpt_doc_selection_types])
    code_gpt_perf_datas = []
    for dataset_name in code_dataset_names:
        code_gpt_perf_datas.append(
            [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][code_metric] for doc_selection_type in code_gpt_doc_selection_types])
    qa_llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        qa_llama_perf_datas.append(
            [results.qa_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type][qa_metric] for doc_selection_type in qa_llama_doc_selection_types])
    code_llama_perf_datas = []
    for dataset_name in code_dataset_names:
        code_llama_perf_datas.append(
            [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type][code_metric] for doc_selection_type in code_llama_doc_selection_types])

    def get_avg_data(perf_datas, doc_selection_types, dataset_names):
        avg_perf_datas = [0]*len(doc_selection_types)
        for data in perf_datas:
            avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
        avg_perf_datas = [item/len(dataset_names) for item in avg_perf_datas]
        return avg_perf_datas

    qa_gpt_perf_datas.append(get_avg_data(qa_gpt_perf_datas, qa_gpt_doc_selection_types, qa_dataset_names))
    qa_llama_perf_datas.append(get_avg_data(qa_llama_perf_datas, qa_llama_doc_selection_types, qa_dataset_names))
    code_gpt_perf_datas.append(get_avg_data(code_gpt_perf_datas, code_gpt_doc_selection_types, code_dataset_names))
    code_llama_perf_datas.append(get_avg_data(code_llama_perf_datas, code_llama_doc_selection_types, code_dataset_names))
    qa_dataset_names.append('avg of QA')
    code_dataset_names.append('avg of code generation')
    auth_qa_dataset_names.append('avg of QA tasks')
    auth_code_dataset_names.append('avg of code tasks')

    plt.style.use('ggplot')
    x = [0, 5, 10, 15, 20, 30, 40, 50]
    fig = plt.figure(figsize=(18, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.75, 1])
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[2])
    # colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(qa_dataset_names)))
    # colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
    markers = ['D', 'o', '^', '*']

    axs = [ax1, ax2, ax3, ax4]
    perf_datas_list = [code_llama_perf_datas, qa_llama_perf_datas, code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_llama_doc_selection_types, qa_llama_doc_selection_types, code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            auth_dataset_names = auth_code_dataset_names
            metric = code_metric
            tmp_colors = colors[4:]
        else:
            dataset_names = qa_dataset_names
            auth_dataset_names = auth_qa_dataset_names
            metric = qa_metric
            tmp_colors = colors[:4]
        for idx, dataset_name in enumerate(dataset_names):
            if ax_idx == 3:
                ax.plot(x, perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
            else:
                ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
        ax.set_ylabel(metric, fontsize=16)
        ax.set_xlabel('Top K Documents', fontsize=16)
        if ax_idx == 3:
            ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        else:
            ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        if ax_idx in [1,3]:
            ax.set_yticks([0.3, 0.5, 0.7, 0.9])
            ax.set_yticklabels([0.3, 0.5, 0.7, 0.9], fontsize=16)
        elif ax_idx == 0:
            ax.set_yticks([0.1, 0.3, 0.5, 0.7])
            ax.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize=16)
        elif ax_idx == 2:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=16)
        if ax_idx == 0:
            ax.set_title('b. CodeLlama-13B, Code Tasks', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('a. Llama2-13B, QA Tasks', fontsize=14)
        elif ax_idx == 2:
            ax.set_title('d. GPT-3.5, Code Tasks', fontsize=14)
        else:
            ax.set_title('c. GPT-3.5, QA Tasks', fontsize=14)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()
#
#
# def make_doc_selection_topk_perplexity():
#     graph_name = 'select_topk_perplexity.pdf'
#     qa_gpt_perf_datas = []
#     qa_metric = 'perplexity'
#     code_metric = 'perplexity'
#     for dataset_name in qa_dataset_names:
#         qa_gpt_perf_datas.append(
#             [results.qa_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][qa_metric] for doc_selection_type in qa_gpt_doc_selection_types])
#     code_gpt_perf_datas = []
#     for dataset_name in code_dataset_names:
#         code_gpt_perf_datas.append(
#             [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][code_metric] for doc_selection_type in code_gpt_doc_selection_types])
#     qa_llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         qa_llama_perf_datas.append(
#             [results.qa_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type][qa_metric] for doc_selection_type in qa_llama_doc_selection_types])
#     code_llama_perf_datas = []
#     for dataset_name in code_dataset_names:
#         code_llama_perf_datas.append(
#             [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type][code_metric] for doc_selection_type in code_llama_doc_selection_types])
#
#     def get_avg_data(perf_datas, doc_selection_types, dataset_names):
#         avg_perf_datas = [0]*len(doc_selection_types)
#         for data in perf_datas:
#             avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
#         avg_perf_datas = [item/len(dataset_names) for item in avg_perf_datas]
#         return avg_perf_datas
#
#     qa_gpt_perf_datas.append(get_avg_data(qa_gpt_perf_datas, qa_gpt_doc_selection_types, qa_dataset_names))
#     qa_llama_perf_datas.append(get_avg_data(qa_llama_perf_datas, qa_llama_doc_selection_types, qa_dataset_names))
#     code_gpt_perf_datas.append(get_avg_data(code_gpt_perf_datas, code_gpt_doc_selection_types, code_dataset_names))
#     code_llama_perf_datas.append(get_avg_data(code_llama_perf_datas, code_llama_doc_selection_types, code_dataset_names))
#     qa_dataset_names.append('avg of QA')
#     code_dataset_names.append('avg of code generation')
#     auth_qa_dataset_names.append('avg of QA tasks')
#     auth_code_dataset_names.append('avg of code tasks')
#
#     plt.style.use('ggplot')
#     fig = plt.figure(figsize=(18, 4))
#     gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.75, 1])
#     # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
#     ax1 = plt.subplot(gs[1])
#     ax2 = plt.subplot(gs[0])
#     ax3 = plt.subplot(gs[3])
#     ax4 = plt.subplot(gs[2])
#     # colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(qa_dataset_names)))
#     # colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))
#     colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
#     markers = ['D', 'o', '^', '*']
#
#     x = [0, 5, 10, 15, 20, 30, 40, 50]
#     axs = [ax1, ax2, ax3, ax4]
#     perf_datas_list = [code_llama_perf_datas, qa_llama_perf_datas, code_gpt_perf_datas, qa_gpt_perf_datas]
#     topk_list = [code_llama_doc_selection_types, qa_llama_doc_selection_types, code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
#         if ax_idx%2 == 0:
#             dataset_names = code_dataset_names
#             auth_dataset_names = auth_code_dataset_names
#             tmp_colors = colors[4:]
#             metric = code_metric
#         else:
#             dataset_names = qa_dataset_names
#             auth_dataset_names = auth_qa_dataset_names
#             tmp_colors = colors[:4]
#             metric = qa_metric
#         for idx, dataset_name in enumerate(dataset_names):
#             if ax_idx == 3:
#                 ax.plot(x, perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
#             else:
#                 ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
#         ax.set_ylabel(metric, fontsize=16)
#         ax.set_xlabel('Top K Documents', fontsize=16)
#         if ax_idx == 3:
#             ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
#             ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
#         else:
#             ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
#             ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
#         if ax_idx == 0:
#             ax.set_yticks([1.12, 1.13, 1.14, 1.15, 1.16])
#             ax.set_yticklabels([1.12, 1.13, 1.14, 1.15, 1.16], fontsize=16)
#         elif ax_idx == 1:
#             ax.set_yticks([1.05, 1.06, 1.07, 1.08, 1.09])
#             ax.set_yticklabels([1.05, 1.06, 1.07, 1.08, 1.09], fontsize=16)
#         elif ax_idx == 2:
#             ax.set_yticks([1.02, 1.03, 1.04, 1.05, 1.06])
#             ax.set_yticklabels([1.02, 1.03, 1.04, 1.05, 1.06], fontsize=16)
#         elif ax_idx == 3:
#             ax.set_yticks([1.02, 1.03, 1.04, 1.05, 1.06])
#             ax.set_yticklabels([1.02, 1.03, 1.04, 1.05, 1.06], fontsize=16)
#         if ax_idx == 0:
#             ax.set_title('b. CodeLlama-13B, Code Tasks', fontsize=14)
#         elif ax_idx == 1:
#             ax.set_title('a. Llama2-13B, QA Tasks', fontsize=14)
#         elif ax_idx == 2:
#             ax.set_title('d. GPT-3.5, Code Tasks', fontsize=14)
#         else:
#             ax.set_title('c. GPT-3.5, QA Tasks', fontsize=14)
#
#     ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
#     ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
#     handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
#     fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()
#
#
#
def make_doc_selection_topk_ret_recall():
    graph_name = 'select_topk_ret_recall.pdf'
    qa_metric = 'Retrieval Recall'
    code_metric = 'Retrieval Recall'

    recall_data = {
        'NQ':                {1: 0.4720, 3: 0.6635, 5: 0.7345, 10: 0.8155, 15: 0.8470, 20: 0.8705, 30: 0.8955, 40: 0.9115},
        'TriviaQA':          {1: 0.6275, 3: 0.8175, 5: 0.8710, 10: 0.9175, 15: 0.9370, 20: 0.9515, 30: 0.9665, 40: 0.9745},
        'hotpotQA':          {1: 0.3510, 3: 0.5320, 5: 0.5820, 10: 0.6390, 15: 0.6660, 20: 0.6880, 30: 0.7120, 40: 0.7320},
        'conala':            {1: 0.0040, 3: 0.0873, 5: 0.0992, 7: 0.1210, 10: 0.1508, 13: 0.1895, 16: 0.2054, 20: 0.2113},
        'pandas_numpy_eval': {1: 0.0993, 3: 0.1801, 5: 0.2250, 7: 0.2570, 10: 0.2974, 13: 0.3423, 16: 0.3782, 20: 0.4012},
        'DS1000':            {1: 0.0479, 3: 0.1006, 5: 0.1316, 7: 0.1642, 10: 0.2014, 13: 0.2228, 16: 0.2393, 20: 0.2478},
    }

    qa_dataset_names = ['NQ', 'TriviaQA', 'hotpotQA']
    code_dataset_names = ['conala', 'DS1000', 'pandas_numpy_eval']

    qa_ks = [1, 3, 5, 10, 15, 20, 30, 40]
    code_ks = [1, 3, 5, 7, 10, 13, 16, 20]
    code_ret_recalls = []
    qa_ret_recalls = []
    for name in code_dataset_names:
        code_ret_recalls.append([recall_data[name][k] for k in code_ks])
    for name in qa_dataset_names:
        qa_ret_recalls.append([recall_data[name][k] for k in qa_ks])


    # def get_avg_data(perf_datas, doc_selection_types, dataset_names):
    #     avg_perf_datas = [0]*len(doc_selection_types)
    #     for data in perf_datas:
    #         avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
    #     avg_perf_datas = [item/len(dataset_names) for item in avg_perf_datas]
    #     return avg_perf_datas



    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    ax2 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
    markers = ['D', 'o', '^', '*']

    # Colors and markers for different datasets
    # if task_type == 'qa':
    #     colors = ['#DC143C', '#FF8C00', '#228B22']  # Red, Orange, Green
    #     markers = ['D', 'o', '^']
    #     force_zero = True  # QA accuracy often benefits from seeing full 0-1 scale
    # else:  # code
    #     colors = ['#4169E1', '#8B4513', '#C71585']  # Blue, Brown, Purple
    #     markers = ['s', 'v', 'p']
    #     force_zero = False  # Code Pass@1 benefits from zoomed-in view

    axs = [ax1, ax2]
    datas_list = [code_ret_recalls, qa_ret_recalls]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            auth_dataset_names = auth_code_dataset_names
            tmp_colors = ['#4169E1', '#8B4513', '#C71585']
        else:
            dataset_names = qa_dataset_names
            auth_dataset_names = auth_qa_dataset_names
            tmp_colors = ['#DC143C', '#FF8C00', '#228B22']
        for idx, dataset_name in enumerate(dataset_names):
            if ax_idx == 0:
                markers = ['s', 'v', 'p']
                ax.plot(code_ks, perf_datas[idx], marker=markers[idx], markersize=8, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx], linewidth=2)
            else:
                markers = ['D', 'o', '^']
                ax.plot(qa_ks, perf_datas[idx], marker=markers[idx], markersize=8, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx], linewidth=2)
        ax.set_ylabel('Retrieval Recall', fontsize=16)
        ax.set_xlabel('k', fontsize=16)
        if ax_idx == 0:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
            ax.set_xticks(code_ks)
            ax.set_xticklabels(code_ks, fontsize=16)
        elif ax_idx == 1:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            ax.set_xticks(qa_ks)
            ax.set_xticklabels(qa_ks, fontsize=16)
        if ax_idx == 0:
            ax.set_title('b. Code Tasks', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('a. QA Tasks', fontsize=14)
        ax.grid(True, alpha=0.3)

    ax3_handles, ax3_labels = ax1.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax2.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()
#
#
# def make_ret_doc_type_perplexity():
#     graph_name = 'ret_doc_type_perplexity.pdf'
#     gpt_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         gpt_perf_datas.append(
#             [results.qa_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
#     for dataset_name in code_dataset_names:
#         gpt_perf_datas.append(
#             [results.code_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
#     llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         llama_perf_datas.append(
#             [results.qa_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
#     for dataset_name in code_dataset_names:
#         llama_perf_datas.append(
#             [results.code_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in
#              ret_doc_types])
#     qa_gpt_perf_datas, code_gpt_perf_datas = gpt_perf_datas[:3], gpt_perf_datas[3:]
#     qa_llama_perf_datas, code_llama_perf_datas = llama_perf_datas[:3], llama_perf_datas[3:]
#
#     plt.style.use('ggplot')
#     fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(12, 12))  # ax1: qa, ax2: code
#     x = len(ret_doc_types)
#     # fig.suptitle('retrieval document type analysis', fontsize=16)
#     fig.text(0.5, -0.035, '(2). Perplexity of RAG systems with Llama2-13B', ha='center', va='center', fontsize=26)
#     fig.text(0.5, 0.505, '(1). Perplexity of RAG systems with GPT-3.5', ha='center', va='center', fontsize=26)
#     fig.subplots_adjust(hspace=0.4, wspace=0.2)
#     doc_type_index = np.arange(x)
#     colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
#     # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
#     axs = [ax1, ax2, ax3]
#     yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
#         # ax.set_xlabel(qa_dataset_names[idx], fontsize=16)
#         ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('PPL', fontsize=24)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax4, ax5, ax6]
#     yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('PPL', fontsize=24)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax7, ax8, ax9]
#     yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('PPL', fontsize=24)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax10, ax11, ax12]
#     yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('PPL', fontsize=24)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#
#     handles, labels = ax1.get_legend_handles_labels()
#     # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.21))
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.6)
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()
#
#
# def make_ret_doc_type_analysis():
#     graph_name = 'ret_doc_type_analysis.pdf'
#     metric = 'has_answer'
#     gpt_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         gpt_perf_datas.append(
#             [results.qa_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type][metric] for ret_doc_type in ret_doc_types])
#     for dataset_name in code_dataset_names:
#         gpt_perf_datas.append(
#             [results.code_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['pass@1'] for ret_doc_type in ret_doc_types])
#     llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         llama_perf_datas.append(
#             [results.qa_ret_doc_type_llama_n_1[dataset_name][ret_doc_type][metric] for ret_doc_type in ret_doc_types])
#     for dataset_name in code_dataset_names:
#         llama_perf_datas.append(
#             [results.code_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['pass@1'] for ret_doc_type in ret_doc_types])
#     qa_gpt_perf_datas, code_gpt_perf_datas = gpt_perf_datas[:3], gpt_perf_datas[3:]
#     qa_llama_perf_datas, code_llama_perf_datas = llama_perf_datas[:3], llama_perf_datas[3:]
#
#     # plt.style.use('ggplot')
#     # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # ax1: qa, ax2: code
#     # bar_width = 0.8 / len(qa_gpt_perf_datas)
#     # x = len(ret_doc_types)
#     # doc_type_index = np.arange(x)
#     # colors1 = plt.cm.viridis(np.linspace(0, 0.5, len(qa_gpt_perf_datas)))
#     # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
#     # for idx, perf_data in enumerate(qa_gpt_perf_datas):
#     #     ax1.bar(doc_type_index+idx*bar_width, perf_data, width=bar_width, label=qa_dataset_names[idx], color=colors1[idx])
#     # ax1.set_xlabel('document type')
#     # ax1.set_ylabel('performance')
#     # ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     # ax1.set_xticks(doc_type_index+bar_width*(x/2-0.5))  # set place of xticks
#     # ax1.set_xticklabels(ret_doc_types, rotation=45, ha='right')
#     # ax1.set_title('Document Type: llama2-13b performance')
#     # for idx, perf_data in enumerate(code_gpt_perf_datas):
#     #     ax2.bar(doc_type_index + idx * bar_width, perf_data, width=bar_width, label=code_dataset_names[idx], color=colors2[idx])
#     # ax2.set_xlabel('document type')
#     # ax2.set_ylabel('performance')
#     # ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     # ax2.set_xticks(doc_type_index + bar_width * (x / 2 - 0.5))  # set place of xticks
#     # ax2.set_xticklabels(ret_doc_types, rotation=45, ha='right')
#     # ax2.set_title('Document Type: gpt-3.5 performance')
#     # ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
#     # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
#     # fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.05))
#     # plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     # plt.show()
#
#     plt.style.use('ggplot')
#     fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(12, 12))  # ax1: qa, ax2: code
#     # fig.suptitle('retrieval document type analysis', fontsize=16)
#     fig.text(0.5, -0.035, '(2). Correctness of RAG systems with Llama2-13B', ha='center', va='center', fontsize=26)
#     fig.text(0.5, 0.505, '(1). Correctness of RAG systems with GPT-3.5', ha='center', va='center', fontsize=26)
#     fig.subplots_adjust(hspace=0.4, wspace=0.2)
#     x = len(ret_doc_types)
#     doc_type_index = np.arange(x)
#     colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
#     # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
#     axs = [ax1, ax2, ax3]
#     yticks_list = [[0,1], [0,1], [0,1]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('Accuracy', fontsize=20)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax4, ax5, ax6]
#     yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('Pass@1', fontsize=20)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax7, ax8, ax9]
#     yticks_list = [[0, 1], [0, 1], [0, 1]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('Accuracy', fontsize=20)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#     axs = [ax10, ax11, ax12]
#     yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
#     for idx, ax in enumerate(axs):
#         ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
#         ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
#         ax.set_xticks([])
#         ax.set_ylabel('Pass@1', fontsize=20)
#         ax.set_yticks(yticks_list[idx])
#         ax.set_yticklabels(yticks_list[idx], fontsize=20)
#         ax.yaxis.set_label_coords(-0.05, 0.5)
#         ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
#
#     handles, labels = ax1.get_legend_handles_labels()
#     # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.6)
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.21))
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()



def create_dataset_subplot(ax, perf_datas, llm_names, dataset_name, # x_positions,
                           xs, ylabel, yticks, yticklabels, baseline_values=None):
    """Helper function to create individual subplots"""
    # colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1',
    #           '#228B22', '#4169E1', '#8B4513', '#C71585']
    colors = ['#DC143C', '#FF8C00', '#228B22', '#4169E1', '#8B4513', '#C71585']
    markers = ['D', 'o', '^', 's', 'v', 'p']

    for idx, (perf_data, llm_name) in enumerate(zip(perf_datas, llm_names)):
        line, = ax.plot(xs, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=llm_name, color=colors[idx])

        # Add baseline horizontal line if provided
        if baseline_values is not None and idx < len(baseline_values):
            ax.axhline(y=baseline_values[idx], color=line.get_color(), linestyle='--', linewidth=3, alpha=0.7)

    ax.set_xlabel('Retrieval Recall', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs, fontsize=20)
    ax.set_title(f'{dataset_name}', fontsize=22)
    ax.grid(True, alpha=0.3)


def make_ret_recall_analysis(dataset_perf_data, dataset_baseline_data, llm_names,
                             qa_dataset_names=['NQ', 'TriviaQA', 'HotpotQA'], code_dataset_names=['CoNaLa', 'DS1000', 'PNE'],
                             ret_recalls=[0, 0.2, 0.4, 0.6, 0.8, 1.0], output_filename='ret_recall_analysis.pdf'):
    """
    Refactored version with helper functions for better code organization
    """

    def get_adaptive_yticks(all_values, n_ticks=5):
        """Generate adaptive y-axis ticks based on data range"""
        if not all_values:
            return [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        min_val = min(all_values)
        max_val = max(all_values)

        # Add some padding (5% on each side)
        range_padding = (max_val - min_val) * 0.05
        y_min = max(0, min_val - range_padding)  # Don't go below 0
        y_max = min(1.0, max_val + range_padding)  # Don't go above 1.0 for percentages

        # Round to nice values (0.05 or 0.10 increments)
        data_range = y_max - y_min

        if data_range <= 0.3:
            step = 0.05  # Use 0.05 increments for smaller ranges
        elif data_range <= 0.5:
            step = 0.10  #
        else:
            step = 0.2

        # data_center = (y_min + y_max) / 2
        # center_steps = data_center / step
        # center_rounded = round(center_steps) * step
        #
        # # Calculate how many steps we need on each side
        # half_range = (y_max - y_min) / 2
        # steps_needed = int(half_range / step) + 1
        #
        # # Generate ticks centered around the rounded center
        # y_min_rounded = center_rounded - steps_needed * step
        # y_max_rounded = center_rounded + steps_needed * step
        #
        # # Ensure we don't go outside [0, 1] bounds
        # y_min_rounded = max(0, y_min_rounded)
        # y_max_rounded = min(1.0, y_max_rounded)

        y_min_rounded = (int(y_min / step) + 1) * step
        # For y_max, only round up if we're more than halfway to the next boundary
        # This prevents excessive rounding up (0.595 -> 0.6 instead of 0.7)
        y_max_steps = y_max / step
        if y_max_steps - int(y_max_steps) > 0.5:
            y_max_rounded = (int(y_max_steps) + 1) * step
        else:
            # If we're close to a boundary, just use that boundary
            y_max_rounded = round(y_max_steps) * step

        # Ensure we stay within logical bounds
        y_min_rounded = max(0, y_min_rounded)
        y_max_rounded = min(1.0, y_max_rounded)

        # If the range becomes too small, extend it slightly
        if y_max_rounded - y_min_rounded < 2 * step:
            if y_max_rounded + step <= 1.0:
                y_max_rounded += step
            elif y_min_rounded - step >= 0:
                y_min_rounded -= step

        # Generate ticks with the chosen step
        yticks = []
        current = y_min_rounded
        while current <= y_max_rounded and len(yticks) < 8:  # Limit to max 8 ticks
            yticks.append(current)
            current += step
            current = round(current, 3)  # Avoid floating point precision issues

        # Ensure we have at least 3 ticks for readability
        if len(yticks) < 3 and yticks[-1] + step <= 1.0:
            yticks.append(round(yticks[-1] + step, 3))

        yticklabels = [f'{tick:.2f}' for tick in yticks]

        return yticks, yticklabels

    # Configuration
    graph_name = 'ret_recall_analysis.pdf'
    qa_metric = 'Accuracy'
    code_metric = 'Pass@1'

    # Create figure
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    x_positions = range(len(ret_recalls))

    # Create subplots using helper function
    for col, dataset_name in enumerate(qa_dataset_names[:3]):  # Ensure max 3
        perf_datas = dataset_perf_data[dataset_name]
        baseline_values = dataset_baseline_data[dataset_name]

        all_values = []
        for perf_list in perf_datas:
            all_values.extend(perf_list)
        all_values.extend(baseline_values)
        yticks, yticklabels = get_adaptive_yticks(all_values)

        create_dataset_subplot(axes[0, col], perf_datas=perf_datas, baseline_values=baseline_values, llm_names=llm_names,
                               xs=ret_recalls, dataset_name=dataset_name,
                               ylabel='Accuracy', yticks=yticks, yticklabels=yticklabels)

    for col, dataset_name in enumerate(code_dataset_names[:3]):
        perf_datas = dataset_perf_data[dataset_name]
        baseline_values = dataset_baseline_data[dataset_name]

        all_values = []
        for perf_list in perf_datas:
            all_values.extend(perf_list)
        all_values.extend(baseline_values)
        yticks, yticklabels = get_adaptive_yticks(all_values)

        create_dataset_subplot(axes[1, col], perf_datas=perf_datas, baseline_values=baseline_values, llm_names=llm_names,
                               xs=ret_recalls, dataset_name=dataset_name,
                               ylabel='Pass@1', yticks=yticks, yticklabels=yticklabels)

    # Add row labels
    axes[0, 0].text(-0.27, 0.5, 'QA Tasks', transform=axes[0, 0].transAxes,
                    fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.27, 0.5, 'Code Tasks', transform=axes[1, 0].transAxes,
                    fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')

    # Create shared legend
    ax_handles, ax_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels,
               loc='lower center', ncol=3, fontsize=24,
               bbox_to_anchor=(0.5, -0.06))

    # fig.suptitle('Model Performance Comparison Across Datasets\n(Solid lines: With Retrieval, Dashed lines: No Retrieval)',
    #     fontsize=20, fontweight='bold', y=0.98)


# Alternative refactored version with helper functions
def create_subplot(ax, perf_datas, dataset_names, colors, markers, x_positions, ret_recalls, title, ylabel, yticks, yticklabels):
    """Helper function to create individual subplots"""
    for idx, (perf_data, dataset_name) in enumerate(zip(perf_datas, dataset_names)):
        ax.plot(x_positions, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=dataset_name, color=colors[idx])

    ax.set_xlabel('Retrieval Recall', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ret_recalls, fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.grid(True, alpha=0.3)


def make_ret_recall_analysis():
    """
    Refactored version with helper functions for better code organization
    """
    # Configuration
    graph_name = 'ret_recall_analysis.pdf'
    qa_metric = 'has_answer'
    code_metric = 'pass@1'

    # Colors and styling
    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1',
              '#228B22', '#4169E1', '#8B4513', '#C71585']
    markers = ['D', 'o', '^']
    qa_colors = colors[:4]
    code_colors = colors[4:]

    # Data extraction (same as above)
    gpt_perf_datas = []
    for dataset_name in qa_dataset_names:
        gpt_perf_datas.append([
            results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric]
            for ret_recall in ret_recalls
        ])
    for dataset_name in code_dataset_names:
        gpt_perf_datas.append([
            results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric]
            for ret_recall in ret_recalls
        ])

    llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        llama_perf_datas.append([
            results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric]
            for ret_recall in ret_recalls
        ])
    for dataset_name in code_dataset_names:
        llama_perf_datas.append([
            results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric]
            for ret_recall in ret_recalls
        ])

    # Create figure
    plt.style.use('ggplot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    x_positions = range(len(ret_recalls))

    # Create subplots using helper function
    create_subplot(ax1, llama_perf_datas[:3], auth_qa_dataset_names,
                   qa_colors, markers, x_positions, ret_recalls,
                   'a. Llama2-13B, QA Tasks', 'Accuracy',
                   [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    create_subplot(ax2, llama_perf_datas[3:], auth_code_dataset_names,
                   code_colors, markers, x_positions, ret_recalls,
                   'b. Llama2-13B, Code Tasks', 'Pass@1',
                   [0, 0.2, 0.4, 0.6], [0, 0.2, 0.4, 0.6])

    create_subplot(ax3, gpt_perf_datas[:3], auth_qa_dataset_names,
                   qa_colors, markers, x_positions, ret_recalls,
                   'c. GPT-3.5, QA Tasks', 'Accuracy',
                   [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    create_subplot(ax4, gpt_perf_datas[3:], auth_code_dataset_names,
                   code_colors, markers, x_positions, ret_recalls,
                   'd. GPT-3.5, Code Tasks', 'Pass@1',
                   [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8])

    # Create shared legend
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()

    fig.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels,
               loc='lower center', ncol=3, fontsize=24,
               bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    plt.savefig(f'graph/{graph_name}', bbox_inches='tight', dpi=300)
    plt.show()



# def make_ret_recall_analysis():
#     graph_name = 'ret_recall_analysis.pdf'
#     qa_metric = 'has_answer'; code_metric = 'pass@1'
#     gpt_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         gpt_perf_datas.append(
#             [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         gpt_perf_datas.append(
#             [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         llama_perf_datas.append(
#             [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         llama_perf_datas.append(
#             [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     gpt_perf_none = [results.qa_ret_doc_type_gpt_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
#     gpt_perf_none.extend([results.code_ret_doc_type_gpt_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
#     llama_perf_none = [results.qa_ret_doc_type_llama_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
#     llama_perf_none.extend([results.code_ret_doc_type_llama_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
#     x = range(len(gpt_perf_datas[0]))
#     plt.style.use('ggplot')
#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     markers = ['D', 'o', '^']
#     qa_colors, code_colors = colors[:4], colors[4:]
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # ax1: qa, ax2: code
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
#         line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#         # ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result
#     ax1.set_xlabel('Retrieval Recall', fontsize=22)
#     ax1.set_ylabel('Accuracy', fontsize=22)
#     ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#     ax1.set_xticks(x, ret_recalls)
#     ax1.set_xticklabels(ret_recalls, fontsize=20)
#     ax1.set_title('a. Llama2-13B, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
#         line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#         # ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result
#     ax2.set_xlabel('Retrieval Recall', fontsize=22)
#     ax2.set_ylabel('Pass@1', fontsize=22)
#     ax2.set_yticks([0, 0.2, 0.4, 0.6])
#     ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=20)
#     ax2.set_xticks(x, ret_recalls)
#     ax2.set_xticklabels(ret_recalls, fontsize=20)
#     ax2.set_title('b. Llama2-13B, Code Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
#         line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#         # ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result
#     ax3.set_xlabel('Retrieval Recall', fontsize=22)
#     ax3.set_ylabel('Accuracy', fontsize=22)
#     ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#     ax3.set_xticks(x, ret_recalls)
#     ax3.set_xticklabels(ret_recalls, fontsize=20)
#     ax3.set_title('c. GPT-3.5, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
#         line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#         # ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result
#     ax4.set_xlabel('Retrieval Recall', fontsize=22)
#     ax4.set_ylabel('Pass@1', fontsize=22)
#     ax4.set_yticks([0.2, 0.4, 0.6, 0.8])
#     ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=20)
#     ax4.set_xticks(x, ret_recalls)
#     ax4.set_xticklabels(ret_recalls, fontsize=20)
#     ax4.set_title('d. GPT-3.5, Code Tasks', fontsize=22)
#
#     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
#     ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.12))
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()


def make_qa_code_retriever_perf():
    graph_name = 'qa_code_perf_on_retriever.pdf'
    gpt_qa_retriever_perf_datas, llama_qa_retriever_perf_datas = dict(), dict()
    gpt_code_retriever_perf_datas, llama_code_retriever_perf_datas = dict(), dict()
    for retriever_name in retriever_names:
        if retriever_name == 'contriever':
            gpt_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3 # get mean perf of qa dataset
            llama_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
        elif retriever_name == 'codeT5':
            gpt_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
            llama_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
        else:
            gpt_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
            llama_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
            gpt_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
            llama_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3

    plt.style.use('ggplot')
    bar_width = 0.8 / 4
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 2))  # ax1: qa, ax2: code
    colors = plt.cm.viridis(np.linspace(0, 1, len(retriever_names)))
    axs = [ax1, ax2, ax3, ax4]
    retriever_data_list = [llama_qa_retriever_perf_datas, gpt_qa_retriever_perf_datas, llama_code_retriever_perf_datas, gpt_code_retriever_perf_datas]
    for ax_idx, (ax, retriever_data) in enumerate(zip(axs, retriever_data_list)):
        for idx, retriever_name in enumerate(retriever_names):
            if retriever_name in retriever_data.keys():
                ax.bar(idx*bar_width if retriever_name != 'codeT5' else (idx-1)*bar_width, retriever_data[retriever_name], width=bar_width, label=retriever_name, color=colors[idx])
        ax.set_ylabel('Performance')
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if ax_idx == 0: ax.set_title('QA Datasets, llama2-13b', fontsize=10)
        elif ax_idx == 1: ax.set_title('QA Datasets, gpt-3.5', fontsize=10)
        elif ax_idx == 2: ax.set_title('Code Datasets, llama2-13b', fontsize=10)
        else: ax.set_title('Code Datasets, gpt-3.5', fontsize=10)
    # for idx, retriever_name in enumerate(retriever_names):
    #     if retriever_name in llama_qa_retriever_perf_datas.keys():
    #         ax1.bar(idx*bar_width, llama_qa_retriever_perf_datas[retriever_name], width=bar_width, label=retriever_name, color=colors[idx])

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    handles, labels = list(dict.fromkeys(ax1_handles + ax3_handles)), list(dict.fromkeys(ax1_labels + ax3_labels))
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.15))
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()



def make_doc_num_analysis(dataset_perf_data, llm_names, qa_dataset_names=['NQ', 'TriviaQA', 'HotpotQA'], code_dataset_names=['CoNaLa', 'DS1000', 'PNE'],
                          qa_ks=[1, 3, 5, 10, 15, 20, 30, 40], code_ks=[1, 3, 5, 7, 10, 13, 16, 20], output_filename='doc_num_analysis.pdf'):
    """
    Refactored version with helper functions for better code organization
    """

    def get_adaptive_yticks(all_values, n_ticks=5):
        """Generate adaptive y-axis ticks based on data range"""
        if not all_values:
            return [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        min_val = min(all_values)
        max_val = max(all_values)

        # Add some padding (5% on each side)
        range_padding = (max_val - min_val) * 0.05
        y_min = max(0, min_val - range_padding)  # Don't go below 0
        y_max = min(1.0, max_val + range_padding)  # Don't go above 1.0 for percentages

        # Round to nice values (0.05 or 0.10 increments)
        data_range = y_max - y_min

        if data_range <= 0.3:
            step = 0.05  # Use 0.05 increments for smaller ranges
        elif data_range <= 0.5:
            step = 0.10  #
        else:
            step = 0.2

        data_center = (y_min + y_max) / 2
        center_steps = data_center / step
        center_rounded = round(center_steps) * step

        # Calculate how many steps we need on each side
        half_range = (y_max - y_min) / 2
        steps_needed = int(half_range / step) + 1

        # Generate ticks centered around the rounded center
        y_min_rounded = center_rounded - steps_needed * step
        y_max_rounded = center_rounded + steps_needed * step

        # Ensure we don't go outside [0, 1] bounds
        y_min_rounded = max(0, y_min_rounded)
        y_max_rounded = min(1.0, y_max_rounded)

        # y_min_rounded = (int(y_min / step)) * step
        # # For y_max, only round up if we're more than halfway to the next boundary
        # # This prevents excessive rounding up (0.595 -> 0.6 instead of 0.7)
        # y_max_steps = y_max / step
        # if y_max_steps - int(y_max_steps) > 0.5:
        #     y_max_rounded = (int(y_max_steps) + 1) * step
        # else:
        #     # If we're close to a boundary, just use that boundary
        #     y_max_rounded = round(y_max_steps) * step

        # Ensure we stay within logical bounds
        y_min_rounded = max(0, y_min_rounded)
        y_max_rounded = min(1.0, y_max_rounded)

        # If the range becomes too small, extend it slightly
        if y_max_rounded - y_min_rounded < 2 * step:
            if y_max_rounded + step <= 1.0:
                y_max_rounded += step
            elif y_min_rounded - step >= 0:
                y_min_rounded -= step

        # Generate ticks with the chosen step
        yticks = []
        current = y_min_rounded
        while current <= y_max_rounded and len(yticks) < 8:  # Limit to max 8 ticks
            yticks.append(current)
            current += step
            current = round(current, 3)  # Avoid floating point precision issues

        # Ensure we have at least 3 ticks for readability
        if len(yticks) < 3 and yticks[-1] + step <= 1.0:
            yticks.append(round(yticks[-1] + step, 3))

        yticklabels = [f'{tick:.2f}' for tick in yticks]

        return yticks, yticklabels

    # Configuration
    # graph_name = 'doc_num_analysis.pdf'
    qa_metric = 'Accuracy'
    code_metric = 'Pass@1'

    # Create figure
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # x_positions = range(len(qa_ks))

    # Create subplots using helper function
    for col, dataset_name in enumerate(qa_dataset_names[:3]):  # Ensure max 3
        perf_datas = dataset_perf_data[dataset_name]

        all_values = []
        for perf_list in perf_datas:
            all_values.extend(perf_list)
        yticks, yticklabels = get_adaptive_yticks(all_values)

        create_dataset_subplot(axes[0, col], perf_datas=perf_datas, llm_names=llm_names,
                               # x_positions=x_positions,
                               xs=qa_ks, dataset_name=dataset_name,
                               ylabel='Accuracy', yticks=yticks, yticklabels=yticklabels)

    for col, dataset_name in enumerate(code_dataset_names[:3]):
        perf_datas = dataset_perf_data[dataset_name]

        all_values = []
        for perf_list in perf_datas:
            all_values.extend(perf_list)
        yticks, yticklabels = get_adaptive_yticks(all_values)

        create_dataset_subplot(axes[1, col], perf_datas=perf_datas, llm_names=llm_names,
                               # x_positions=x_positions,
                               xs=code_ks, dataset_name=dataset_name,
                               ylabel='Pass@1', yticks=yticks, yticklabels=yticklabels)

    # Add row labels
    axes[0, 0].text(-0.27, 0.5, 'QA Tasks', transform=axes[0, 0].transAxes,
                    fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.27, 0.5, 'Code Tasks', transform=axes[1, 0].transAxes,
                    fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')

    # Create shared legend
    ax_handles, ax_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels,
               loc='lower center', ncol=3, fontsize=24,
               bbox_to_anchor=(0.5, -0.06))

    # fig.suptitle('Model Performance Comparison Across Datasets\n(Solid lines: With Retrieval, Dashed lines: No Retrieval)',
    #     fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'graph/{output_filename}', bbox_inches='tight', dpi=300)
    plt.show()





# def make_ret_recall_perplexity():
#     graph_name = 'ret_recall_perplexity.pdf'
#     qa_metric = 'perplexity'; code_metric = 'perplexity'
#     gpt_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         gpt_perf_datas.append(
#             [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         gpt_perf_datas.append(
#             [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         llama_perf_datas.append(
#             [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         llama_perf_datas.append(
#             [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     x = range(len(gpt_perf_datas[0]))
#     plt.style.use('ggplot')
#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     markers = ['D', 'o', '^']
#     qa_colors, code_colors = colors[:4], colors[4:]
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # ax1: qa, ax2: code
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
#         line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#     ax1.set_xlabel('Retrieval Recall', fontsize=22)
#     ax1.set_ylabel(f'Perplexity', fontsize=22)
#     # ax1.set_yticks([1.055, 1.06, 1.065, 1.07, 1.075])
#     # ax1.set_yticklabels([1.055, 1.06, 1.065, 1.07, 1.075], fontsize=20)
#     ax1.set_yticks([1.05, 1.06, 1.07, 1.08])
#     ax1.set_yticklabels([1.05, 1.06, 1.07, 1.08], fontsize=20)
#     ax1.set_xticks(x, ret_recalls)
#     ax1.set_xticklabels(ret_recalls, fontsize=20)
#     ax1.set_title('a. Llama2-13B, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
#         line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#     ax2.set_xlabel('Retrieval Recall', fontsize=22)
#     ax2.set_ylabel(f'Perplexity', fontsize=22)
#     ax2.set_yticks([1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
#     ax2.set_yticklabels([1.11, 1.12, 1.13, 1.14, 1.15, 1.16], fontsize=16)
#     ax2.set_xticks(x, ret_recalls)
#     ax2.set_xticklabels(ret_recalls, fontsize=20)
#     ax2.set_title('b. Llama2-13B, Code Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
#         line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#     ax3.set_xlabel('Retrieval Recall', fontsize=22)
#     ax3.set_ylabel(f'Perplexity', fontsize=22)
#     # ax3.set_yticks([1.015, 1.025, 1.035, 1.045, 1.055])
#     # ax3.set_yticklabels([1.015, 1.025, 1.035, 1.045, 1.055], fontsize=20)
#     ax3.set_yticks([1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
#     ax3.set_yticklabels([1.01, 1.02, 1.03, 1.04, 1.05, 1.06], fontsize=20)
#     ax3.set_xticks(x, ret_recalls)
#     ax3.set_xticklabels(ret_recalls, fontsize=20)
#     ax3.set_title('c. GPT-3.5, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
#         line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#     ax4.set_xlabel('Retrieval Recall', fontsize=22)
#     ax4.set_ylabel(f'Perplexity', fontsize=22)
#     # ax4.set_yticks([1.025, 1.03, 1.035, 1.04, 1.045, 1.05])
#     # ax4.set_yticklabels([1.025, 1.03, 1.035, 1.04, 1.045, 1.05], fontsize=20)
#     ax4.set_yticks([1.02, 1.03, 1.04, 1.05])
#     ax4.set_yticklabels([1.02, 1.03, 1.04, 1.05], fontsize=20)
#     ax4.set_xticks(x, ret_recalls)
#     ax4.set_xticklabels(ret_recalls, fontsize=20)
#     ax4.set_title('d. GPT-3.5, Code Tasks', fontsize=22)
#
#     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
#     ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.12))
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()


# def make_ret_recall_analysis():
#     graph_name = 'ret_recall_analysis.pdf'
#     qa_metric = 'has_answer'; code_metric = 'pass@1'
#     gpt_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         gpt_perf_datas.append(
#             [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         gpt_perf_datas.append(
#             [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     llama_perf_datas = []
#     for dataset_name in qa_dataset_names:
#         llama_perf_datas.append(
#             [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
#     for dataset_name in code_dataset_names:
#         llama_perf_datas.append(
#             [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
#     gpt_perf_none = [results.qa_ret_doc_type_gpt_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
#     gpt_perf_none.extend([results.code_ret_doc_type_gpt_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
#     llama_perf_none = [results.qa_ret_doc_type_llama_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
#     llama_perf_none.extend([results.code_ret_doc_type_llama_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
#     x = range(len(gpt_perf_datas[0]))
#     plt.style.use('ggplot')
#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     markers = ['D', 'o', '^']
#     qa_colors, code_colors = colors[:4], colors[4:]
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # ax1: qa, ax2: code
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
#         line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#         # ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result
#     ax1.set_xlabel('Retrieval Recall', fontsize=22)
#     ax1.set_ylabel('Accuracy', fontsize=22)
#     ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#     ax1.set_xticks(x, ret_recalls)
#     ax1.set_xticklabels(ret_recalls, fontsize=20)
#     ax1.set_title('a. Llama2-13B, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
#         line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#         # ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result
#     ax2.set_xlabel('Retrieval Recall', fontsize=22)
#     ax2.set_ylabel('Pass@1', fontsize=22)
#     ax2.set_yticks([0, 0.2, 0.4, 0.6])
#     ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=20)
#     ax2.set_xticks(x, ret_recalls)
#     ax2.set_xticklabels(ret_recalls, fontsize=20)
#     ax2.set_title('b. Llama2-13B, Code Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
#         line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
#         # ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result
#     ax3.set_xlabel('Retrieval Recall', fontsize=22)
#     ax3.set_ylabel('Accuracy', fontsize=22)
#     ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#     ax3.set_xticks(x, ret_recalls)
#     ax3.set_xticklabels(ret_recalls, fontsize=20)
#     ax3.set_title('c. GPT-3.5, QA Tasks', fontsize=22)
#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
#         line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
#         # ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result
#     ax4.set_xlabel('Retrieval Recall', fontsize=22)
#     ax4.set_ylabel('Pass@1', fontsize=22)
#     ax4.set_yticks([0.2, 0.4, 0.6, 0.8])
#     ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=20)
#     ax4.set_xticks(x, ret_recalls)
#     ax4.set_xticklabels(ret_recalls, fontsize=20)
#     ax4.set_title('d. GPT-3.5, Code Tasks', fontsize=22)
#
#     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
#     ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
#     handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.12))
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()


# def make_qa_code_retriever_perf():
#     graph_name = 'qa_code_perf_on_retriever.pdf'
#     gpt_qa_retriever_perf_datas, llama_qa_retriever_perf_datas = dict(), dict()
#     gpt_code_retriever_perf_datas, llama_code_retriever_perf_datas = dict(), dict()
#     for retriever_name in retriever_names:
#         if retriever_name == 'contriever':
#             gpt_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3 # get mean perf of qa dataset
#             llama_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
#         elif retriever_name == 'codeT5':
#             gpt_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
#             llama_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
#         else:
#             gpt_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
#             llama_qa_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[qa_name][retriever_name]['recall'] for qa_name in qa_dataset_names])/3
#             gpt_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_gpt[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
#             llama_code_retriever_perf_datas[retriever_name] = sum([results.retriever_perf_llama[code_name][retriever_name]['pass@1'] for code_name in code_dataset_names])/3
#
#     plt.style.use('ggplot')
#     bar_width = 0.8 / 4
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 2))  # ax1: qa, ax2: code
#     colors = plt.cm.viridis(np.linspace(0, 1, len(retriever_names)))
#     axs = [ax1, ax2, ax3, ax4]
#     retriever_data_list = [llama_qa_retriever_perf_datas, gpt_qa_retriever_perf_datas, llama_code_retriever_perf_datas, gpt_code_retriever_perf_datas]
#     for ax_idx, (ax, retriever_data) in enumerate(zip(axs, retriever_data_list)):
#         for idx, retriever_name in enumerate(retriever_names):
#             if retriever_name in retriever_data.keys():
#                 ax.bar(idx*bar_width if retriever_name != 'codeT5' else (idx-1)*bar_width, retriever_data[retriever_name], width=bar_width, label=retriever_name, color=colors[idx])
#         ax.set_ylabel('Performance')
#         ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#         if ax_idx == 0: ax.set_title('QA Datasets, llama2-13b', fontsize=10)
#         elif ax_idx == 1: ax.set_title('QA Datasets, gpt-3.5', fontsize=10)
#         elif ax_idx == 2: ax.set_title('Code Datasets, llama2-13b', fontsize=10)
#         else: ax.set_title('Code Datasets, gpt-3.5', fontsize=10)
#     # for idx, retriever_name in enumerate(retriever_names):
#     #     if retriever_name in llama_qa_retriever_perf_datas.keys():
#     #         ax1.bar(idx*bar_width, llama_qa_retriever_perf_datas[retriever_name], width=bar_width, label=retriever_name, color=colors[idx])
#
#     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
#     ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
#     handles, labels = list(dict.fromkeys(ax1_handles + ax3_handles)), list(dict.fromkeys(ax1_labels + ax3_labels))
#     fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.15))
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()
#
#
#
#
# def make_qa_code_ret_recall():
#     auth_retriever_names = ['BM25', 'MiniLM', 'text-embedding', 'Contriever', 'CodeT5']
#     graph_name = 'qa_code_ret_recall.pdf'
#     qa_retrieval_acc_datas = []
#     for retriever_index, retriever_name in enumerate(retriever_names):  # for qa
#         if retriever_name == 'codeT5':
#             qa_retrieval_acc_datas.append(None)
#             continue
#         qa_retrieval_acc_datas.append([0 for _ in top_ks])
#         for dataset_name in qa_dataset_names:
#             for top_k_index, top_k in enumerate(top_ks):
#                 qa_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
#         for top_k_index in range(len(top_ks)):
#             qa_retrieval_acc_datas[retriever_index][top_k_index] /= len(qa_dataset_names)
#     code_retrieval_acc_datas = []
#     for retriever_index, retriever_name in enumerate(retriever_names):  # for code
#         if retriever_name == 'contriever':
#             code_retrieval_acc_datas.append(None)
#             continue
#         code_retrieval_acc_datas.append([0 for _ in top_ks])
#         for dataset_name in code_dataset_names:
#             for top_k_index, top_k in enumerate(top_ks):
#                 code_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
#         for top_k_index in range(len(top_ks)):
#             code_retrieval_acc_datas[retriever_index][top_k_index] /= len(code_dataset_names)
#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     markers = ['D', 'o', '^', '*', '+']
#     qa_colors, code_colors = colors[:4], colors[4:]
#
#     # x = range(len(qa_retrieval_acc_datas[0]))
#     # x = [1, 3, 5, 10, 20, 50, 100]
#     x = [1, 3, 6, 10, 15, 22, 33]
#     plt.style.use('ggplot')
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))   # ax1: qa, ax2: code
#     for idx, (retrieval_acc_data, retriever_name) in enumerate(zip(qa_retrieval_acc_datas, retriever_names)):
#         if retrieval_acc_data: ax1.plot(x, retrieval_acc_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_retriever_names[idx])
#         else: ax1.plot([], [], marker='o', linestyle='-', label=retriever_name)
#     ax1.set_xlabel('Top K Documents', fontsize=18)
#     # ax1.set_ylabel('Retrieval Recall', fontsize=16)
#     ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
#     ax1.set_xticks(x, top_ks)
#     ax1.set_xticklabels(x, fontsize=18)
#     ax1.set_ylabel('Retrieval Recall', fontsize=20)
#     # ax1.set_xticks(top_ks)
#     ax1.set_xticklabels(top_ks, fontsize=18)
#     ax1.set_title('QA Datasets', fontsize=18)
#     for idx, (retrieval_acc_data, retriever_name) in enumerate(zip(code_retrieval_acc_datas, retriever_names)):
#         if retrieval_acc_data: ax2.plot(x, retrieval_acc_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_retriever_names[idx])
#         else: ax2.plot([], [], marker='o', linestyle='-', label=retriever_name)
#     ax2.set_xlabel('Top K Documents', fontsize=18)
#     # ax2.set_ylabel('Retrieval Recall', fontsize=16)
#     ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
#     ax2.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
#     ax2.set_xticks(x, top_ks)
#     ax2.set_xticklabels(top_ks, fontsize=18)
#     ax2.set_title('Code Datasets', fontsize=18)
#     ax2.set_ylabel('Rretrieval Recall', fontsize=20)
#
#     handles, labels = ax1.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=16, bbox_to_anchor=(0.5, -0.1))
#     plt.savefig('graph/' + graph_name, bbox_inches='tight')
#     plt.show()
#
#
# def make_avg_ret_recall():
#     graph_name = 'avg_retrieval_recall.pdf'
#     avg_retrieval_acc_datas = []
#     for retriever_index, retriever_name in enumerate(retriever_names):
#         avg_retrieval_acc_datas.append([0 for _ in top_ks])
#         for dataset_name in dataset_names:
#             for top_k_index, top_k in enumerate(top_ks):
#                 avg_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
#         for top_k_index in range(len(top_ks)):
#             avg_retrieval_acc_datas[retriever_index][top_k_index] /= len(dataset_names)
#
#     x = range(len(avg_retrieval_acc_datas[0]))
#     colors = ['b', 'g', 'r', 'c']
#     plt.style.use('ggplot')
#     for avg_retrieval_acc_data, retriever_name, color in zip(avg_retrieval_acc_datas, retriever_names, colors):
#         plt.plot(x, avg_retrieval_acc_data, marker='o', linestyle='-', label=retriever_name)
#     plt.xlabel('top k')
#     plt.ylabel('Recall')
#     plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     plt.xticks(x, top_ks)
#     plt.title('Avg Retrieval Recall of Six Datasets for Retrievers')
#     plt.legend()
#     plt.savefig('graph/' + graph_name)
#     plt.show()
#
#
# def make_prompt_method_avg_correctness():
#     graph_name = 'prompt_method_avg_correctness.pdf'
#
#     prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
#     qa_metric = 'has_answer'
#     code_metric = 'pass@1'
#
#     qa_gpt_avg_prompt_perf_datas = []
#     qa_llama_avg_prompt_perf_datas = []
#     code_gpt_avg_prompt_perf_datas = []
#     code_llama_avg_prompt_perf_datas = []
#     for prompt_method in prompt_method_list:
#         try: avg_qa_gpt_data = sum([results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names])/3
#         except: avg_qa_gpt_data = 0
#         qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
#         try: avg_qa_llama_data = sum([results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names])/3
#         except: avg_qa_llama_data = 0
#         qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
#         try: avg_code_gpt_data = sum([results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names])/3
#         except: avg_code_gpt_data = 0
#         code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
#         try: avg_code_llama_data = sum([results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names])/3
#         except: avg_code_llama_data = 0
#         code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)
#
#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     plt.style.use('ggplot')
#     fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(12, 24), sharex=True)  # ax1: qa, ax2: code
#     axes = [ax1, ax2, ax3, ax4]
#     perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas, code_llama_avg_prompt_perf_datas]
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
#         ax.bar(range(len(perf_datas)), perf_datas, width=0.6)
#         if ax_idx == 3:
#             ax.set_xticks(range(len(perf_datas)))
#             ax.set_xticklabels(prompt_method_list, rotation=45, ha='right', fontsize=16)
#         else:
#             ax.set_xticks([])
#         if ax_idx in [0, 1]:
#             ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#             ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
#             ax.set_ylabel(qa_metric, fontsize=16)
#         else:
#             ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
#             ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
#             ax.set_ylabel(code_metric, fontsize=16)
#
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name)
#     plt.show()
#
#
# def make_prompt_method_correctness():
#     graph_name = 'prompt_method_correctness.pdf'
#
#     # prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
#     # auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN', 'IR-CoT', 'FLARE']
#     prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
#     auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN']
#     qa_metric = 'has_answer'
#     code_metric = 'pass@1'
#
#     qa_gpt_avg_prompt_perf_datas = []
#     qa_llama_avg_prompt_perf_datas = []
#     code_gpt_avg_prompt_perf_datas = []
#     code_llama_avg_prompt_perf_datas = []
#     for prompt_method in prompt_method_list:
#         try:
#             avg_qa_gpt_data = [results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#             avg_qa_gpt_data.append(sum(avg_qa_gpt_data)/3)  # avg
#         except: avg_qa_gpt_data = [0, 0, 0, 0]
#         qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
#         try:
#             avg_qa_llama_data = [results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#             avg_qa_llama_data.append(sum(avg_qa_llama_data) / 3)  # avg
#         except: avg_qa_llama_data = [0, 0, 0, 0, 0]
#         qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
#         try:
#             avg_code_gpt_data = [results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#             avg_code_gpt_data.append(sum(avg_code_gpt_data) / 3)  # avg
#         except: avg_code_gpt_data = [0, 0, 0, 0]
#         code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
#         try:
#             avg_code_llama_data = [results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#             avg_code_llama_data.append(sum(avg_code_llama_data) / 3)  # avg
#         except: avg_code_llama_data = [0, 0, 0, 0]
#         code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)
#     print(qa_gpt_avg_prompt_perf_datas)
#     print(qa_llama_avg_prompt_perf_datas)
#     print(code_gpt_avg_prompt_perf_datas)
#     print(code_llama_avg_prompt_perf_datas)
#
#     # colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
#     # colors = ['skyblue', 'lightgreen', 'salmon']
#     # colors = ['skyblue', 'lightgreen', 'tomato', '#228B22']
#     colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
#     colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
#     hatches = ['/', '||', 'X', 'O']
#     edge_colors = ['red', 'green', 'green', 'blue']
#     edge_colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
#     edge_colors = ['#4D4D4D']*4
#     line_colors = ['#8B0000', '#FF6347', '#228B22', '#4169E1']
#     plt.style.use('ggplot')
#     fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(24, 12))  # ax1: qa, ax2: code
#     axes = [ax1, ax2, ax3, ax4]
#     perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas,
#                        code_llama_avg_prompt_perf_datas]
#     for perf_datas in perf_datas_list:
#         print(len(perf_datas))
#         for perf_data in perf_datas:
#             print(len(perf_data))
#     auth_qa_dataset_names.append('avg of QA tasks')
#     auth_code_dataset_names.append('avg of code tasks')
#
#     special_idx = 1
#     bar_width = 0.2
#     prompt_method_list.remove('0shot')
#     index = np.arange(len(prompt_method_list))
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
#         if ax_idx in [0,1]:
#             dataset_names = auth_qa_dataset_names
#         else:
#             dataset_names = auth_code_dataset_names
#         for dataset_idx, dataset_name in enumerate(dataset_names):
#             offset = dataset_idx * bar_width
#             bar_data = [item[dataset_idx] for item in perf_datas][1:]
#             hatch_styles = [hatches[dataset_idx] if idx != special_idx or ax_idx in [2,3] else 'XX' for idx in index]  # Dense hatch for unreliable
#
#             ax.bar(index+offset, bar_data, width=bar_width, label=dataset_name,
#                    color=colors[dataset_idx], hatch=hatch_styles, edgecolor=edge_colors[dataset_idx])
#             ax.axhline(y=[item[dataset_idx] for item in perf_datas][0], color=line_colors[dataset_idx], linestyle=':', linewidth=4)  # plot baseline result
#         if ax_idx == 3 or ax_idx == 1:
#             ax.set_xticks(index+bar_width*1.5)
#             ax.set_xticklabels(auth_prompt_method_names[1:], rotation=45, ha='right', fontsize=22)
#         else:
#             ax.set_xticks([])
#         if ax_idx in [0, 1]:
#             ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#             ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#             ax.set_ylabel(qa_metric, fontsize=24)
#         else:
#             ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
#             ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
#             ax.set_ylabel(code_metric, fontsize=24)
#         if ax_idx == 0:
#             ax.set_title('a. GPT-3.5, QA Tasks', fontsize=22)
#         if ax_idx == 2:
#             ax.set_title('b. GPT-3.5, Code Tasks', fontsize=22)
#         if ax_idx == 1:
#             ax.set_title('c. Llama2-13B, QA Tasks', fontsize=22)
#         if ax_idx == 3:
#             ax.set_title('d. CodeLlama-13B, Code Tasks', fontsize=22)
#         ax.legend(loc='upper right', fontsize=18, ncol=4)
#
#         if ax_idx in [0,1]:
#             bar_center = index[special_idx] + bar_width
#             # ax.text(bar_center, 0.9, 'Unreliable', ha='center', fontsize=12, color='black')
#             ax.annotate('Unreliable', xy=(bar_center, 0.75), xytext=(bar_center + 0.3, 0.82),
#                         arrowprops=dict(facecolor='black', shrink=0.01), fontsize=22)
#
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name)
#     plt.show()
#
#
#
# def make_prompt_method_perplexity():
#     graph_name = 'prompt_method_perplexity.pdf'
#
#     # prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
#     # auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN', 'IR-CoT', 'FLARE']
#     prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
#     auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN']
#     qa_metric = 'perplexity'
#     code_metric = 'perplexity'
#
#     qa_gpt_avg_prompt_perf_datas = []
#     qa_llama_avg_prompt_perf_datas = []
#     code_gpt_avg_prompt_perf_datas = []
#     code_llama_avg_prompt_perf_datas = []
#     for prompt_method in prompt_method_list:
#         avg_qa_gpt_data = [results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#         print(f'QA LLAMA {prompt_method}: ', avg_qa_gpt_data)
#         avg_qa_gpt_data = sum(avg_qa_gpt_data)/3  # avg
#         qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
#         avg_qa_llama_data = [results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#         print(f'QA GPT {prompt_method}: ', avg_qa_llama_data)
#         avg_qa_llama_data = sum(avg_qa_llama_data)/3  # avg
#         qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
#         avg_code_gpt_data = [results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#         print(f'CODE GPT {prompt_method}: ', avg_code_gpt_data)
#         avg_code_gpt_data = sum(avg_code_gpt_data)/3  # avg
#         code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
#         avg_code_llama_data = [results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#         print(f'QA LLAMA {prompt_method}: ', avg_code_llama_data)
#         avg_code_llama_data = sum(avg_code_llama_data)/3  # avg
#         code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)
#
#
#     colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
#     edge_colors = ['#4D4D4D']*4
#     plt.style.use('ggplot')
#     fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 8))  # ax1: qa, ax2: code
#     axes = [ax1, ax2, ax3, ax4]
#     perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas,
#                        code_llama_avg_prompt_perf_datas]
#
#     bar_width = 0.5
#     index = np.arange(len(prompt_method_list))
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
#         if ax_idx in [0,1]:
#             dataset_name = 'avg of QA tasks'
#         else:
#             dataset_name = 'avg of code tasks'
#
#         ax.bar(index, perf_datas, width=bar_width, label=dataset_name, color=colors[3], edgecolor=edge_colors[3])
#         if ax_idx == 3 or ax_idx == 1:
#             ax.set_xticks(index)
#             ax.set_xticklabels(auth_prompt_method_names, rotation=90, ha='right', fontsize=22)
#         else:
#             ax.set_xticks(ticks=range(len(auth_prompt_method_names)))
#         if ax_idx in [1, 2]:
#             ax.set_ylim(1, 1.4)
#             ax.set_yticks([1, 1.1, 1.2, 1.3, 1.4])
#             ax.set_yticklabels([1, 1.1, 1.2, 1.3, 1.4], fontsize=20)
#             ax.set_ylabel('PPL', fontsize=24)
#         elif ax_idx in [0, 3]:
#             ax.set_ylim(1, 1.4)
#             ax.set_yticks([1, 1.1, 1.2, 1.3, 1.4])
#             ax.set_yticklabels([1, 1.1, 1.2, 1.3, 1.4], fontsize=20)
#             ax.set_ylabel('PPL', fontsize=24)
#         if ax_idx == 0:
#             ax.set_title('a. GPT-3.5, QA Tasks', fontsize=22)
#         if ax_idx == 2:
#             ax.set_title('b. GPT-3.5, Code Tasks', fontsize=22)
#         if ax_idx == 1:
#             ax.set_title('c. Llama2-13B, QA Tasks', fontsize=22)
#         if ax_idx == 3:
#             ax.set_title('d. Llama2-13B, Code Tasks', fontsize=22)
#         ax.legend(loc='upper right', fontsize=18, ncol=1)
#
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name)
#     plt.show()
#
#
#
# def make_prompt_method_percentage_of_only_correct():
#     graph_name = 'prompt_method_percentage_only_correct.pdf'
#
#     # prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
#     # auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN', 'IR-CoT', 'FLARE']
#     prompt_method_list = ['3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
#     auth_prompt_method_names = ['few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN']
#     qa_metric = 'has_answer'
#     code_metric = 'pass@1'
#
#     qa_gpt_avg_prompt_perf_datas = []
#     qa_llama_avg_prompt_perf_datas = []
#     code_gpt_avg_prompt_perf_datas = []
#     code_llama_avg_prompt_perf_datas = []
#     for prompt_method in prompt_method_list:
#         try:
#             avg_qa_gpt_data = [results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#             avg_qa_gpt_data.append(sum(avg_qa_gpt_data) / 3)  # avg
#         except: avg_qa_gpt_data = [0, 0, 0, 0]
#         qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
#         try:
#             avg_qa_llama_data = [results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
#             avg_qa_llama_data.append(sum(avg_qa_llama_data) / 3)  # avg
#         except: avg_qa_llama_data = [0, 0, 0, 0, 0]
#         qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
#         try:
#             avg_code_gpt_data = [results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#             avg_code_gpt_data.append(sum(avg_code_gpt_data) / 3)  # avg
#         except: avg_code_gpt_data = [0, 0, 0, 0]
#         code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
#         try:
#             avg_code_llama_data = [results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
#             avg_code_llama_data.append(sum(avg_code_llama_data) / 3)  # avg
#         except: avg_code_llama_data = [0, 0, 0, 0]
#         code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)
#
#     percentage_datas = [percentage_of_correct_prompt_method_gpt_qa, percentage_of_correct_prompt_method_llama_qa, percentage_of_correct_prompt_method_gpt_code, percentage_of_correct_prompt_method_llama_code]
#     for idx, percentage_data in enumerate(percentage_datas):
#         new_percentage_data = [[] for _ in range(len(auth_prompt_method_names))]
#         for key, value in percentage_data.items():
#             for value_idx, v in enumerate(value):
#                 new_percentage_data[value_idx].append(v)
#         for xx_idx in range(len(auth_prompt_method_names)): new_percentage_data[xx_idx].append(sum(new_percentage_data[xx_idx])/3)
#         percentage_datas[idx] = new_percentage_data
#
#     perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas,
#                        code_llama_avg_prompt_perf_datas]
#     for idx, perf_datas in enumerate(perf_datas_list):
#         print(perf_datas)
#         print(percentage_datas[idx])
#         for xx_idx, perf_data in enumerate(perf_datas):
#             perf_datas_list[idx][xx_idx] = [a/100 for a,b in zip(percentage_datas[idx][xx_idx], perf_datas_list[idx][xx_idx])]
#         print(perf_datas_list[idx])
#
#     auth_qa_dataset_names.append('avg of QA tasks')
#     auth_code_dataset_names.append('avg of code tasks')
#
#
#     colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
#     hatches = ['/', '||', 'X', 'O']
#     edge_colors = ['#4D4D4D'] * 4
#     line_colors = ['#8B0000', '#FF6347', '#228B22', '#4169E1']
#     plt.style.use('ggplot')
#     fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 8))  # ax1: qa, ax2: code
#     axes = [ax1, ax2, ax3, ax4]
#     special_idx = 1
#     bar_width = 0.5
#     index = np.arange(len(prompt_method_list))
#     for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
#         if ax_idx in [0, 1]:
#             dataset_names = auth_qa_dataset_names
#         else:
#             dataset_names = auth_code_dataset_names
#         ax.bar(index, [item[3] for item in perf_datas], width=bar_width, label=dataset_names[3],
#                color=colors[3], edgecolor=edge_colors[3])
#         if ax_idx == 3 or ax_idx == 1:
#             ax.set_xticks(index)
#             ax.set_xticklabels(auth_prompt_method_names, rotation=90, ha='right', fontsize=22)
#         else:
#             ax.set_xticks(ticks=range(len(auth_prompt_method_names)))
#         ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
#         ax.set_yticklabels([0, 5, 10, 15, 20], fontsize=20)
#         ax.set_ylabel('percentage', fontsize=22)
#         if ax_idx == 0:
#             ax.set_title('a. GPT-3.5, QA Tasks', fontsize=22)
#         if ax_idx == 2:
#             ax.set_title('b. GPT-3.5, Code Tasks', fontsize=22)
#         if ax_idx == 1:
#             ax.set_title('c. Llama2-13B, QA Tasks', fontsize=22)
#         if ax_idx == 3:
#             ax.set_title('d. CodeLlama-13B, Code Tasks', fontsize=22)
#         ax.legend(loc='upper right', fontsize=18, ncol=4)
#
#         if ax_idx in [0, 1]:
#             bar_center = index[special_idx] + bar_width
#             # ax.text(bar_center, 0.9, 'Unreliable', ha='center', fontsize=12, color='black')
#             ax.annotate('Unreliable', xy=(bar_center-bar_width, 0.05), xytext=(bar_center, 0.1),
#                         arrowprops=dict(facecolor='black', shrink=0.01), fontsize=22)
#
#     plt.tight_layout()
#     plt.savefig('graph/' + graph_name)
#     plt.show()


def make_doc_num_analysis_llm_focused(dataset_perf_data, llm_names,
                                      qa_dataset_names=['NQ', 'TriviaQA', 'HotpotQA'],
                                      code_dataset_names=['CoNaLa', 'DS1000', 'PNE'],
                                      qa_ks=[1, 3, 5, 10, 15, 20, 30, 40],
                                      code_ks=[1, 3, 5, 7, 10, 13, 16, 20],
                                      output_filename='llm_focused_analysis.pdf'):
    """
    Create a comprehensive figure showing both QA and Code tasks.
    Each subplot represents one LLM, showing its performance across multiple datasets.

    Args:
        dataset_perf_data: Original format - {dataset_name: [llm1_perf, llm2_perf, ...]}
        llm_names: List of LLM names
    """

    def get_adaptive_yticks(if_qa, all_values, force_zero_start=False, n_ticks=5):
        """Generate adaptive y-axis ticks based on data range"""
        if not all_values:
            return [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        min_val = min(all_values)
        max_val = max(all_values)

        print(f"Debug: min_val={min_val:.4f}, max_val={max_val:.4f}")

        if force_zero_start:
            y_min = 0
        else:
            # For better visualization, add padding and don't force zero start
            range_val = max_val - min_val
            padding = max(range_val * 0.1, 0.02)  # At least 2% padding
            y_min = min_val - padding
            # Only enforce zero if we're already very close to zero
            if y_min < 0 and min_val < 0.05:
                y_min = 0
            elif y_min < 0:
                y_min = 0  # Don't go negative, but allow starting above 0

        # Add padding to top - no upper constraint now
        range_val = max_val - min_val
        padding_top = max(range_val * 0.05, 0.01)
        y_max = max_val + padding_top
        # Remove the constraint: y_max = min(1.0, y_max)

        print(f"Debug: y_min={y_min:.4f}, y_max={y_max:.4f}")

        # Determine appropriate step size based on data range
        data_range = y_max - y_min

        if data_range <= 0.05:
            step = 0.01
        elif data_range <= 0.1:
            step = 0.02
        elif data_range <= 0.2:
            step = 0.05
        elif data_range <= 0.5:
            step = 0.10
        elif data_range <= 1.0:
            step = 0.2
        elif data_range <= 2.0:
            step = 0.5
        else:
            step = 1.0

        print(f"Debug: data_range={data_range:.4f}, step={step:.4f}")

        # Round bounds to nice step boundaries
        if if_qa:
            y_min_rounded = (int(y_min / step) + 1) * step
            y_max_rounded = ((int(y_max / step))) * step
        else:
            y_min_rounded = (int(y_min / step)) * step
            y_max_rounded = ((int(y_max / step)) + 1) * step

        # Make sure we don't go below original bounds unnecessarily
        if not force_zero_start and y_min_rounded < y_min - step / 2:
            y_min_rounded += step

        # Ensure we don't go below 0 (but no upper limit now)
        y_min_rounded = max(0, y_min_rounded)
        # Remove: y_max_rounded = min(1.0, y_max_rounded)

        # Ensure minimum range for visibility (at least 3-4 ticks)
        min_range = 3 * step
        if y_max_rounded - y_min_rounded < min_range:
            # Try to expand upward first (no limit now)
            y_max_rounded += step
            # If we're not forcing zero start, can also expand downward
            if not force_zero_start and y_min_rounded - step >= 0:
                y_min_rounded -= step

        print(f"Debug: y_min_rounded={y_min_rounded:.4f}, y_max_rounded={y_max_rounded:.4f}")

        # Generate ticks
        yticks = []
        current = y_min_rounded
        while current <= y_max_rounded + 0.001 and len(yticks) < 12:  # Allow more ticks for larger ranges
            yticks.append(round(current, 4))
            current += step

        # Format labels based on step size and magnitude
        if step <= 0.01:
            yticklabels = [f'{tick:.3f}' for tick in yticks]
        elif step <= 0.05:
            yticklabels = [f'{tick:.2f}' for tick in yticks]
        elif step < 1.0:
            yticklabels = [f'{tick:.1f}' for tick in yticks]
        else:
            yticklabels = [f'{tick:.0f}' for tick in yticks]

        print(f"Debug: Generated {len(yticks)} ticks: {yticks[:5]}{'...' if len(yticks) > 5 else ''}")
        return yticks, yticklabels

    def transform_data_for_llm_view(dataset_perf_data, dataset_names, llm_names):
        """
        Transform data from original format to LLM-focused format

        Original: {dataset_name: [llm1_perf, llm2_perf, ...]}
        New: {llm_name: {dataset_name: perf_data}}
        """
        llm_data = {}

        for llm_idx, llm_name in enumerate(llm_names):
            llm_data[llm_name] = {}
            for dataset_name in dataset_names:
                if dataset_name in dataset_perf_data:
                    # Extract this LLM's performance for this dataset
                    llm_data[llm_name][dataset_name] = dataset_perf_data[dataset_name][llm_idx]
                else:
                    print(f"Warning: Dataset {dataset_name} not found in data")

        return llm_data

    def create_llm_subplot(ax, llm_name, llm_datasets_data, dataset_names, ks,
                           ylabel, task_type):
        """Create subplot for one LLM showing multiple datasets with individual y-axis scaling"""

        # Colors and markers for different datasets
        if task_type == 'qa':
            colors = ['#DC143C', '#FF8C00', '#228B22']  # Red, Orange, Green
            markers = ['D', 'o', '^']
            force_zero = False  # QA accuracy often benefits from seeing full 0-1 scale
        else:  # code
            colors = ['#4169E1', '#8B4513', '#C71585']  # Blue, Brown, Purple
            markers = ['s', 'v', 'p']
            force_zero = False  # Code Pass@1 benefits from zoomed-in view

        # Collect all values for this specific LLM to determine optimal y-range
        llm_all_values = []
        for dataset_name in dataset_names:
            if dataset_name in llm_datasets_data:
                perf_data = llm_datasets_data[dataset_name]
                llm_all_values.extend(perf_data)

        # Get individual y-ticks for this specific LLM's data range
        if llm_all_values:
            yticks, yticklabels = get_adaptive_yticks(task_type=='qa', llm_all_values, force_zero_start=force_zero)
        else:
            yticks, yticklabels = [0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

        # Plot each dataset for this LLM
        if llm_name == 'Llama2-13B' and 'NQ' in dataset_names: proc_ks = ks[:6]
        else: proc_ks = ks
        for dataset_idx, dataset_name in enumerate(dataset_names):
            if dataset_name in llm_datasets_data:
                perf_data = llm_datasets_data[dataset_name]
                ax.plot(proc_ks, perf_data,
                        marker=markers[dataset_idx],
                        markersize=8,
                        linestyle='-',
                        label=dataset_name,
                        color=colors[dataset_idx],
                        linewidth=2)

        ax.set_xlabel('k', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=16)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks, fontsize=16)
        ax.set_title(f'{llm_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Show legend for each subplot
        # ax.legend(fontsize=10, loc='best')

        print(f"LLM {llm_name} ({task_type}): y-range {yticks[0]:.3f} to {yticks[-1]:.3f}")

    # Transform data for both QA and Code tasks
    qa_datasets = qa_dataset_names[:3]
    code_datasets = code_dataset_names[:3]

    qa_llm_data = transform_data_for_llm_view(dataset_perf_data, qa_datasets, llm_names)
    code_llm_data = transform_data_for_llm_view(dataset_perf_data, code_datasets, llm_names)

    # Don't use global y-axis ranges - each subplot will have its own optimized range
    print("Using individual y-axis ranges for each LLM subplot for better visualization")

    # Create figure - 2 rows (QA, Code) x n_llms columns
    n_llms = len(llm_names)
    fig, axes = plt.subplots(2, n_llms, figsize=(5 * n_llms, 10))

    # Handle case where we have only 1 LLM
    if n_llms == 1:
        axes = axes.reshape(-1, 1)

    # Create QA subplots (top row) - each with individual y-axis scaling
    for llm_idx, llm_name in enumerate(llm_names):
        create_llm_subplot(axes[0, llm_idx], llm_name, qa_llm_data[llm_name],
                           qa_datasets, qa_ks, 'PPL', 'qa')

    # Create Code subplots (bottom row) - each with individual y-axis scaling
    for llm_idx, llm_name in enumerate(llm_names):
        create_llm_subplot(axes[1, llm_idx], llm_name, code_llm_data[llm_name],
                           code_datasets, code_ks, 'PPL', 'code')

    # Add row labels
    axes[0, 0].text(-0.35, 0.5, 'QA Tasks', transform=axes[0, 0].transAxes,
                    fontsize=18, fontweight='bold', rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.35, 0.5, 'Code Tasks', transform=axes[1, 0].transAxes,
                    fontsize=18, fontweight='bold', rotation=90, va='center', ha='center')

    # Add overall title
    # fig.suptitle('LLM Performance Comparison Across QA and Code Tasks',
    #              fontsize=20, fontweight='bold', y=1.0)

    ax_handles, ax_labels = axes[0, 0].get_legend_handles_labels()
    ax_handles_code, ax_labels_code = axes[1, 0].get_legend_handles_labels()
    fig.legend(ax_handles+ax_handles_code, ax_labels+ax_labels_code,
               loc='lower center', ncol=3, fontsize=16,
               bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.savefig(f'graph/{output_filename}', bbox_inches='tight', dpi=300)
    plt.show()

    # Print data transformation summary
    print(f"Data transformed successfully!")
    print(f"QA Datasets: {qa_datasets}")
    print(f"Code Datasets: {code_datasets}")
    print(f"LLMs: {llm_names}")
    print(f"Figure saved as: graph/{output_filename}")


if __name__ == '__main__':
    # make_avg_ret_recall()

    # make_qa_code_ret_recall()

    make_ret_recall_analysis()
    # recall_perf_data = dict(NQ=      [[0.213, 0.322, 0.421, 0.527, 0.635, 0.735],
    #                                   [0.158, 0.275, 0.401, 0.526, 0.648, 0.770],
    #                                   [0.273, 0.380, 0.486, 0.592, 0.696, 0.787]],
    #                         TriviaQA=[[0.528, 0.600, 0.670, 0.734, 0.811, 0.888],
    #                                   [0.514, 0.580, 0.654, 0.731, 0.813, 0.888],
    #                                   [0.642, 0.705, 0.748, 0.789, 0.849, 0.905]],
    #                         HotpotQA=[[0.201, 0.310, 0.418, 0.506, 0.598, 0.692],
    #                                   [0.213, 0.316, 0.417, 0.492, 0.585, 0.684],
    #                                   [0.239, 0.357, 0.458, 0.552, 0.637, 0.730]],
    #                         CoNaLa=  [[0.226, 0.298, 0.381, 0.417, 0.488, 0.524],
    #                                   [0.369, 0.393, 0.500, 0.512, 0.512, 0.548],
    #                                   [0.357, 0.440, 0.488, 0.512, 0.548, 0.595]],
    #                         DS1000=  [[0.140, 0.166, 0.166, 0.197, 0.248, 0.242],
    #                                   [0.261, 0.293, 0.280, 0.293, 0.299, 0.338],
    #                                   [0.433, 0.446, 0.452, 0.446, 0.459, 0.490]],
    #                         PNE=     [[0.545, 0.551, 0.581, 0.599, 0.617, 0.659],
    #                                   [0.671, 0.677, 0.737, 0.754, 0.766, 0.808],
    #                                   [0.760, 0.784, 0.778, 0.796, 0.802, 0.838]],
    #                         )
    #
    # recall_baseline_data = dict(NQ=      [0.401, 0.496, 0.500],
    #                             TriviaQA=[0.737, 0.862, 0.872],
    #                             HotpotQA=[0.277, 0.324, 0.354],
    #                             CoNaLa=  [0.429, 0.512, 0.571],
    #                             DS1000=  [0.210, 0.338, 0.465],
    #                             PNE=     [0.689, 0.796, 0.790],
    #                         )
    #
    # make_ret_recall_analysis(dataset_perf_data=recall_perf_data, dataset_baseline_data=recall_baseline_data, llm_names=['Llama2-13B', 'gpt-3.5-turbo', 'gpt-4o-mini'],)



    # docnum_perf_data = dict(NQ=      [[0.435, 0.516, 0.544, 0.559, 0.552, 0.546],
    #                                   [0.427, 0.500, 0.525, 0.543, 0.545, 0.545, 0.549, 0.549],
    #                                   [0.495, 0.550, 0.587, 0.602, 0.613, 0.618, 0.624, 0.627]],
    #                         TriviaQA=[[0.732, 0.779, 0.801, 0.825, 0.836, 0.835],
    #                                   [0.740, 0.776, 0.789, 0.818, 0.824, 0.820, 0.828, 0.840],
    #                                   [0.779, 0.815, 0.823, 0.841, 0.850, 0.857, 0.861, 0.865]],
    #                         HotpotQA=[[0.354, 0.400, 0.415, 0.443, 0.436, 0.429],
    #                                   [0.346, 0.391, 0.407, 0.423, 0.428, 0.427, 0.436, 0.438],
    #                                   [0.413, 0.440, 0.464, 0.494, 0.497, 0.500, 0.515, 0.524]],
    #                         CoNaLa=  [[0.238, 0.226, 0.226, 0.238, 0.214, 0.298, 0.250, 0.274],     # llama2
    #                                   [0.345, 0.286, 0.333, 0.345, 0.286, 0.345, 0.357, 0.405],     # gpt-3.5
    #                                   [0.369, 0.393, 0.417, 0.429, 0.393, 0.393, 0.393, 0.429]],    # gpt-4o
    #                         DS1000=  [[0.146, 0.121, 0.135, 0.121, 0.121, 0.140, 0.127, 0.166],
    #                                   [0.268, 0.248, 0.261, 0.287, 0.318, 0.318, 0.318, 0.312],
    #                                   [0.446, 0.408, 0.439, 0.420, 0.382, 0.376, 0.414, 0.357]],
    #                         PNE=     [[0.539, 0.533, 0.539, 0.557, 0.611, 0.611, 0.569, 0.551],
    #                                   [0.569, 0.647, 0.659, 0.707, 0.701, 0.731, 0.713, 0.725],
    #                                   [0.647, 0.689, 0.695, 0.695, 0.701, 0.701, 0.749, 0.731]],
    #                         )
    #
    # make_doc_num_analysis_llm_focused(dataset_perf_data=docnum_perf_data, llm_names=['Llama2-13B', 'gpt-3.5-turbo', 'gpt-4o-mini'], )

    # doc_num_ppl_data = dict(NQ=      [[1.06837, 1.06516, 1.06768, 1.07612, 1.07635, 1.08315],
    #                                   [1.03980, 1.03840, 1.03562, 1.03920, 1.04009, 1.04160, 1.04449, 1.04781],
    #                                   [1.05381, 1.05160, 1.05217, 1.05398, 1.05459, 1.05508, 1.05548, 1.05567]],
    #                         TriviaQA=[[1.06275, 1.06077, 1.05993, 1.06818, 1.07315, 1.08441],
    #                                   [1.02798, 1.02465, 1.02303, 1.02326, 1.02504, 1.02534, 1.02542, 1.02858],
    #                                   [1.03095, 1.02708, 1.02676, 1.02706, 1.02720, 1.02618, 1.02617, 1.02686]],
    #                         HotpotQA=[[1.06208, 1.06002, 1.05983, 1.06011, 1.06482, 1.06920],
    #                                   [1.04118, 1.03687, 1.03857, 1.03861, 1.03993, 1.03921, 1.04174, 1.04285],
    #                                   [1.05581, 1.04737, 1.04721, 1.04606, 1.04569, 1.04594, 1.04580, 1.04661]],
    #                         CoNaLa=  [[1.15786, 1.16620, 1.16587, 1.16477, 1.17476, 1.17538, 1.17445, 1.16720],  # llama2
    #                                   [1.03702, 1.04643, 1.04296, 1.04402, 1.05064, 1.05248, 1.05434, 1.04689],  # gpt-3.5
    #                                   [1.03608, 1.04007, 1.03521, 1.03829, 1.03717, 1.03771, 1.03699, 1.03549]],  # gpt-4o
    #                         DS1000=  [[1.14383, 1.12741, 1.12806, 1.12710, 1.12337, 1.11702, 1.11472, 1.10787],
    #                                   [1.03635, 1.03939, 1.03984, 1.03805, 1.03957, 1.04168, 1.04079, 1.04605],
    #                                   [1.03150, 1.03604, 1.03141, 1.03313, 1.03190, 1.03205, 1.02931, 1.03142]],
    #                         PNE=     [[1.11725, 1.13907, 1.13781, 1.14587, 1.14441, 1.14197, 1.13808, 1.12771],
    #                                   [1.02660, 1.02309, 1.02266, 1.02137, 1.02051, 1.02087, 1.02030, 1.02166],
    #                                   [1.01680, 1.02220, 1.02178, 1.02115, 1.02020, 1.01803, 1.01740, 1.01688]],
    #                         )
    #
    # make_doc_num_analysis_llm_focused(dataset_perf_data=doc_num_ppl_data, llm_names=['Llama2-13B', 'gpt-3.5-turbo', 'gpt-4o-mini'], output_filename='doc_num_ppl.pdf')

    # make_ret_recall_perplexity()

    # make_ret_doc_type_analysis()

    # make_doc_selection_topk_analysis()

    # make_qa_code_retriever_perf()

    # make_ret_doc_type_perplexity()

    # make_doc_selection_topk_syntax_semantic_error()

    # make_doc_selection_topk_perplexity()

    # make_doc_selection_topk_ret_recall()

    # make_prompt_method_avg_correctness()

    # make_prompt_method_correctness()

    # make_prompt_method_perplexity()

    # make_prompt_method_percentage_of_only_correct()

    # make_doc_selection_percentage_of_mistakes()
