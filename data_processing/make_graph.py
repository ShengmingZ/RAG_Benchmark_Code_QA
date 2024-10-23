import matplotlib.pyplot as plt
from matplotlib import gridspec

import results
import numpy as np


dataset_names = ['NQ', 'TriviaQA', 'hotpotQA', 'conala', 'DS1000', 'pandas_numpy_eval']
auth_dataset_names = ['NQ', 'TriviaQA', 'HotpotQA', 'CoNaLa', 'DS1000', 'PNE']
qa_dataset_names, code_dataset_names = dataset_names[:3], dataset_names[3:]
auth_qa_dataset_names, auth_code_dataset_names = auth_dataset_names[:3], auth_dataset_names[3:]
retriever_names = ['BM25', 'miniLM', 'openai-embedding', 'contriever', 'codeT5']
top_ks = [1, 3, 5, 10, 20, 50, 100]
ret_recalls = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# ret_doc_types = ["oracle", "retrieved", "distracting", "random", "irrelevant_dummy", "irrelevant_diff", "none"]
ret_doc_types = ["oracle", "distracting", "random", "irrelevant_dummy", "irrelevant_diff"]
qa_gpt_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20', 'top_40', 'top_60', 'top_80']
qa_llama_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
code_gpt_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']
code_llama_doc_selection_types = ['top_1', 'top_5', 'top_10', 'top_15', 'top_20']


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

percentage_of_mistakes_as_documents_added_gpt = [
    [3.6, 4.3, 5.1, 5.8, 6.4, 7.1],
    [2.4, 2.9, 3.4, 3, 3.4, 4],
    [4.4, 5.2, 5.9, 6.1, 7.6, 7.9],
    [3.5, 4.1, 4.8, 5.0, 5.8, 6.3],
    [4.8, 6, 7.1],
    [5.1, 6.4, 8.3],
    [4.2, 5.4, 5.4],
    [4.7, 5.9, 6.9]
]

percentage_of_mistakes_as_documents_added_llama = [
    [5.4, 6.8, 9],
    [3, 3.2, 4.3],
    [4, 5.9, 7.4],
    [4.1, 5.3, 6.9],
    [4.8, 8.3, 9.5],
    [1.9, 2.5, 4.5],
    [3.6, 4.2, 6.6],
    [3.4, 5, 6.9]
]


percentage_of_correct_prompt_method_gpt_qa = {
    'NQ': [0.054, 0.183, 0.053, 0.054, 0.07, 0.065, 0.069, 0.077],
    'TriviaQA': [0.035, 0.091, 0.041, 0.038, 0.05, 0.042, 0.051, 0.043],
    'hotpotQA': [0.054, 0.178, 0.087, 0.084, 0.101, 0.122, 0.069, 0.081]
}

percentage_of_correct_prompt_method_gpt_code = {
    'conala': [],
    'DS1000': [],
    'pandas_numpy_eval': [],
}

percentage_of_correct_prompt_method_llama_qa = {
    'NQ': [],
    'TriviaQA': [],
    'hotpotQA': []
}

percentage_of_correct_prompt_method_llama_code = {
    'conala': [],
    'DS1000': [],
    'pandas_numpy_eval': [],
}


def make_doc_selection_percentage_of_mistakes():
    graph_name = 'select_topk_percentage_of_mistakes.pdf'

    plt.style.use('ggplot')
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

    qa_dataset_names.append('avg of QA')
    code_dataset_names.append('avg of code')
    auth_qa_dataset_names.append('avg of QA tasks')
    auth_code_dataset_names.append('avg of code tasks')

    x = [0, 5, 10, 20, 30, 40]
    axs = [ax1, ax2, ax3, ax4]
    code_llama_datas = percentage_of_mistakes_as_documents_added_llama[4:]
    qa_llama_datas = percentage_of_mistakes_as_documents_added_llama[:4]
    code_gpt_datas = percentage_of_mistakes_as_documents_added_gpt[4:]
    qa_gpt_datas = percentage_of_mistakes_as_documents_added_gpt[:4]
    perf_datas_list = [code_llama_datas, qa_llama_datas, code_gpt_datas, qa_gpt_datas]
    topk_list = [code_llama_doc_selection_types[2:], qa_llama_doc_selection_types[2:], code_gpt_doc_selection_types[2:], qa_gpt_doc_selection_types[2:]]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx % 2 == 0:
            dataset_names = code_dataset_names
            auth_dataset_names = auth_code_dataset_names
            tmp_colors = colors[4:]
        else:
            dataset_names = qa_dataset_names
            auth_dataset_names = auth_qa_dataset_names
            tmp_colors = colors[:4]
        for idx, dataset_name in enumerate(dataset_names):
            if ax_idx == 3:
                ax.plot(x, perf_datas[idx], marker=markers[idx],
                        markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
            else:
                ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx],
                        markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
        ax.set_ylabel('percentage of instances', fontsize=14)
        ax.set_xlabel('top k documents', fontsize=16)
        if ax_idx == 3:
            ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        else:
            ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        # if ax_idx in [1, 3]:
        #     ax.set_yticks([0.3, 0.5, 0.7, 0.9])
        #     ax.set_yticklabels([0.3, 0.5, 0.7, 0.9], fontsize=12)
        # elif ax_idx == 0:
        #     ax.set_yticks([0.1, 0.3, 0.5, 0.7])
        #     ax.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize=12)
        # elif ax_idx == 2:
        #     ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        #     ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=12)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontsize=16)
        if ax_idx == 0:
            ax.set_title('CodeLlama-13B, Code Datasets', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('Llama2-13B, QA Datasets', fontsize=14)
        elif ax_idx == 2:
            ax.set_title('GPT-3.5, Code Datasets', fontsize=14)
        else:
            ax.set_title('GPT-3.5, QA Datasets', fontsize=14)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_doc_selection_topk_syntax_semantic_error():
    graph_name = 'select_topk_syntax_error.pdf'
    llama_syntax_errors = []
    gpt_syntax_errors = []
    llama_semantic_errors = []
    gpt_semantic_errors = []
    for dataset_name in code_dataset_names:
        llama_syntax_errors.append(
            [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type]['syntax_error_percent'] for doc_selection_type in code_llama_doc_selection_types])
        gpt_syntax_errors.append(
            [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type]['syntax_error_percent'] for doc_selection_type in code_gpt_doc_selection_types])
        llama_semantic_errors.append(
            [results.code_ret_doc_selection_topk_llama_n_1[dataset_name][doc_selection_type]['semantic_error_percent'] for doc_selection_type in code_llama_doc_selection_types])
        gpt_semantic_errors.append(
            [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type]['semantic_error_percent'] for doc_selection_type in code_gpt_doc_selection_types])

    def get_avg_data(perf_datas, doc_selection_types, dataset_names):
        avg_perf_datas = [0] * len(doc_selection_types)
        for data in perf_datas:
            avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
        avg_perf_datas = [item / len(dataset_names) for item in avg_perf_datas]
        return avg_perf_datas

    datas_list = [llama_syntax_errors, gpt_syntax_errors, llama_semantic_errors, gpt_semantic_errors]
    topk_list = [code_llama_doc_selection_types, code_gpt_doc_selection_types, code_llama_doc_selection_types, code_gpt_doc_selection_types]
    for datas, topk in zip(datas_list, topk_list):
        datas.append(get_avg_data(datas, topk, code_dataset_names))
    qa_dataset_names.append('avg syntax error')
    code_dataset_names.append('avg semantic error')

    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
    colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(code_dataset_names)))
    colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))

    axs = [ax1, ax2, ax3, ax4]
    error_datas_list = [llama_syntax_errors, gpt_syntax_errors, llama_semantic_errors, gpt_semantic_errors]
    topk_list = [code_llama_doc_selection_types, code_gpt_doc_selection_types, code_llama_doc_selection_types, code_gpt_doc_selection_types]
    for ax_idx, (ax, datas) in enumerate(zip(axs, error_datas_list)):
        dataset_names = code_dataset_names
        for idx, dataset_name in enumerate(dataset_names):
            ax.plot(topk_list[ax_idx], datas[idx], marker='o', linestyle='-', label=dataset_name, color=colors1[idx])
        ax.set_ylabel('error percentage')
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if ax_idx == 0:
            ax.set_title('syntax error, llama2-13b', fontsize=10)
        elif ax_idx == 1:
            ax.set_title('syntax error, gpt-3.5', fontsize=10)
        elif ax_idx == 2:
            ax.set_title('semantic error, llama2-13b', fontsize=10)
        else:
            ax.set_title('syntax error, gpt-3.5', fontsize=10)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    handles, labels = ax3_handles, ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()

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
        ax.set_xlabel('top k documents', fontsize=16)
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
            ax.set_title('CodeLlama-13B, Code Datasets', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('Llama2-13B, QA Datasets', fontsize=14)
        elif ax_idx == 2:
            ax.set_title('GPT-3.5, Code Datasets', fontsize=14)
        else:
            ax.set_title('GPT-3.5, QA Datasets', fontsize=14)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_doc_selection_topk_perplexity():
    graph_name = 'select_topk_perplexity.pdf'
    qa_gpt_perf_datas = []
    qa_metric = 'perplexity'
    code_metric = 'perplexity'
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
    fig = plt.figure(figsize=(18, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.75, 1])
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[2])
    # colors1 = plt.cm.viridis(np.linspace(0, 0.9, len(qa_dataset_names)))
    # colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(code_dataset_names)))
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
    markers = ['D', 'o', '^', '*']

    x = [0, 5, 10, 15, 20, 30, 40, 50]
    axs = [ax1, ax2, ax3, ax4]
    perf_datas_list = [code_llama_perf_datas, qa_llama_perf_datas, code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_llama_doc_selection_types, qa_llama_doc_selection_types, code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            auth_dataset_names = auth_code_dataset_names
            tmp_colors = colors[4:]
            metric = code_metric
        else:
            dataset_names = qa_dataset_names
            auth_dataset_names = auth_qa_dataset_names
            tmp_colors = colors[:4]
            metric = qa_metric
        for idx, dataset_name in enumerate(dataset_names):
            if ax_idx == 3:
                ax.plot(x, perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
            else:
                ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
        ax.set_ylabel(metric, fontsize=16)
        ax.set_xlabel('top k documents', fontsize=16)
        if ax_idx == 3:
            ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        else:
            ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        if ax_idx == 0:
            ax.set_yticks([1.12, 1.13, 1.14, 1.15, 1.16])
            ax.set_yticklabels([1.12, 1.13, 1.14, 1.15, 1.16], fontsize=16)
        elif ax_idx == 1:
            ax.set_yticks([1.05, 1.06, 1.07, 1.08, 1.09])
            ax.set_yticklabels([1.05, 1.06, 1.07, 1.08, 1.09], fontsize=16)
        elif ax_idx == 2:
            ax.set_yticks([1.02, 1.03, 1.04, 1.05, 1.06])
            ax.set_yticklabels([1.02, 1.03, 1.04, 1.05, 1.06], fontsize=16)
        elif ax_idx == 3:
            ax.set_yticks([1.02, 1.03, 1.04, 1.05, 1.06])
            ax.set_yticklabels([1.02, 1.03, 1.04, 1.05, 1.06], fontsize=16)
        if ax_idx == 0:
            ax.set_title('CodeLlama-13B, Code Datasets', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('Llama2-13B, QA Datasets', fontsize=14)
        elif ax_idx == 2:
            ax.set_title('GPT-3.5, Code Datasets', fontsize=14)
        else:
            ax.set_title('GPT-3.5, QA Datasets', fontsize=14)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()



def make_doc_selection_topk_ret_recall():
    graph_name = 'select_topk_ret_recall.pdf'
    qa_gpt_perf_datas = []
    qa_metric = 'ret_recall'
    code_metric = 'ret_recall'
    for dataset_name in qa_dataset_names:
        qa_gpt_perf_datas.append(
            [results.qa_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][qa_metric] for doc_selection_type in qa_gpt_doc_selection_types])
    code_gpt_perf_datas = []
    for dataset_name in code_dataset_names:
        code_gpt_perf_datas.append(
            [results.code_ret_doc_selection_topk_gpt_n_1[dataset_name][doc_selection_type][code_metric] for doc_selection_type in code_gpt_doc_selection_types])

    def get_avg_data(perf_datas, doc_selection_types, dataset_names):
        avg_perf_datas = [0]*len(doc_selection_types)
        for data in perf_datas:
            avg_perf_datas = [a + b for a, b in zip(avg_perf_datas, data)]
        avg_perf_datas = [item/len(dataset_names) for item in avg_perf_datas]
        return avg_perf_datas

    qa_gpt_perf_datas.append(get_avg_data(qa_gpt_perf_datas, qa_gpt_doc_selection_types, qa_dataset_names))
    code_gpt_perf_datas.append(get_avg_data(code_gpt_perf_datas, code_gpt_doc_selection_types, code_dataset_names))
    qa_dataset_names.append('qa avg')
    code_dataset_names.append('code avg')
    auth_qa_dataset_names.append('avg of QA tasks')
    auth_code_dataset_names.append('avg of code tasks')

    x = [0, 5, 10, 15, 20, 30, 40, 50]
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.75, 1])
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    ax2 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
    markers = ['D', 'o', '^', '*']

    axs = [ax1, ax2]
    perf_datas_list = [code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            auth_dataset_names = auth_code_dataset_names
            tmp_colors = colors[4:]
            metric = code_metric
        else:
            dataset_names = qa_dataset_names
            auth_dataset_names = auth_qa_dataset_names
            tmp_colors = colors[:4]
            metric = qa_metric
        for idx, dataset_name in enumerate(dataset_names):
            if ax_idx == 0:
                ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
            else:
                ax.plot(x, perf_datas[idx], marker=markers[idx], markersize=7.5, linestyle='-', label=auth_dataset_names[idx], color=tmp_colors[idx])
        ax.set_ylabel('Retrieval Recall', fontsize=16)
        ax.set_xlabel('top k documents', fontsize=16)
        if ax_idx == 0:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=16)
            ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        elif ax_idx == 1:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            ax.set_xticks(x, [item.split('_')[1] for item in topk_list[ax_idx]])
            ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=16)
        if ax_idx == 0:
            ax.set_title('Code Generation Datasets', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('QA Datasets', fontsize=14)

    ax3_handles, ax3_labels = ax1.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax2.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_ret_doc_type_perplexity():
    graph_name = 'ret_doc_type_perplexity.pdf'
    gpt_perf_datas = []
    for dataset_name in qa_dataset_names:
        gpt_perf_datas.append(
            [results.qa_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
    for dataset_name in code_dataset_names:
        gpt_perf_datas.append(
            [results.code_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
    llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        llama_perf_datas.append(
            [results.qa_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in ret_doc_types])
    for dataset_name in code_dataset_names:
        llama_perf_datas.append(
            [results.code_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['perplexity'] for ret_doc_type in
             ret_doc_types])
    qa_gpt_perf_datas, code_gpt_perf_datas = gpt_perf_datas[:3], gpt_perf_datas[3:]
    qa_llama_perf_datas, code_llama_perf_datas = llama_perf_datas[:3], llama_perf_datas[3:]

    plt.style.use('ggplot')
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(24, 6))  # ax1: qa, ax2: code
    x = len(ret_doc_types)
    # fig.suptitle('retrieval document type analysis', fontsize=16)
    fig.text(0.5, -0.035, '(2) perplexity of Llama2-13B / CodeLlama-13B over six datasets', ha='center', va='center', fontsize=26)
    fig.text(0.5, 0.505, '(1) perplexity of GPT-3.5 over six datasets', ha='center', va='center', fontsize=26)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    doc_type_index = np.arange(x)
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
    # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
    axs = [ax1, ax2, ax3]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        # ax.set_xlabel(qa_dataset_names[idx], fontsize=16)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('PPL', fontsize=24)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax4, ax5, ax6]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('PPL', fontsize=24)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax7, ax8, ax9]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('PPL', fontsize=24)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax10, ax11, ax12]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('PPL', fontsize=24)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])

    handles, labels = ax1.get_legend_handles_labels()
    # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=24, bbox_to_anchor=(0.5, -0.21))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_ret_doc_type_analysis():
    graph_name = 'ret_doc_type_analysis.pdf'
    metric = 'has_answer'
    gpt_perf_datas = []
    for dataset_name in qa_dataset_names:
        gpt_perf_datas.append(
            [results.qa_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type][metric] for ret_doc_type in ret_doc_types])
    for dataset_name in code_dataset_names:
        gpt_perf_datas.append(
            [results.code_ret_doc_type_gpt_n_1[dataset_name][ret_doc_type]['pass@1'] for ret_doc_type in ret_doc_types])
    llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        llama_perf_datas.append(
            [results.qa_ret_doc_type_llama_n_1[dataset_name][ret_doc_type][metric] for ret_doc_type in ret_doc_types])
    for dataset_name in code_dataset_names:
        llama_perf_datas.append(
            [results.code_ret_doc_type_llama_n_1[dataset_name][ret_doc_type]['pass@1'] for ret_doc_type in ret_doc_types])
    qa_gpt_perf_datas, code_gpt_perf_datas = gpt_perf_datas[:3], gpt_perf_datas[3:]
    qa_llama_perf_datas, code_llama_perf_datas = llama_perf_datas[:3], llama_perf_datas[3:]

    # plt.style.use('ggplot')
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # ax1: qa, ax2: code
    # bar_width = 0.8 / len(qa_gpt_perf_datas)
    # x = len(ret_doc_types)
    # doc_type_index = np.arange(x)
    # colors1 = plt.cm.viridis(np.linspace(0, 0.5, len(qa_gpt_perf_datas)))
    # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
    # for idx, perf_data in enumerate(qa_gpt_perf_datas):
    #     ax1.bar(doc_type_index+idx*bar_width, perf_data, width=bar_width, label=qa_dataset_names[idx], color=colors1[idx])
    # ax1.set_xlabel('document type')
    # ax1.set_ylabel('performance')
    # ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax1.set_xticks(doc_type_index+bar_width*(x/2-0.5))  # set place of xticks
    # ax1.set_xticklabels(ret_doc_types, rotation=45, ha='right')
    # ax1.set_title('Document Type: llama2-13b performance')
    # for idx, perf_data in enumerate(code_gpt_perf_datas):
    #     ax2.bar(doc_type_index + idx * bar_width, perf_data, width=bar_width, label=code_dataset_names[idx], color=colors2[idx])
    # ax2.set_xlabel('document type')
    # ax2.set_ylabel('performance')
    # ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax2.set_xticks(doc_type_index + bar_width * (x / 2 - 0.5))  # set place of xticks
    # ax2.set_xticklabels(ret_doc_types, rotation=45, ha='right')
    # ax2.set_title('Document Type: gpt-3.5 performance')
    # ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
    # fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    # plt.savefig('graph/' + graph_name, bbox_inches='tight')
    # plt.show()

    plt.style.use('ggplot')
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(24, 6))  # ax1: qa, ax2: code
    # fig.suptitle('retrieval document type analysis', fontsize=16)
    fig.text(0.5, -0.035, '(2) correctness of Llama2-13B / CodeLlama-13B over six datasets', ha='center', va='center', fontsize=26)
    fig.text(0.5, 0.505, '(1) correctness of GPT-3.5 over six datasets', ha='center', va='center', fontsize=26)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    x = len(ret_doc_types)
    doc_type_index = np.arange(x)
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
    # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
    axs = [ax1, ax2, ax3]
    yticks_list = [[0,1], [0,1], [0,1]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel(metric, fontsize=20)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax4, ax5, ax6]
    yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('pass@1', fontsize=20)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax7, ax8, ax9]
    yticks_list = [[0, 1], [0, 1], [0, 1]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel(metric, fontsize=20)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax10, ax11, ax12]
    yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel('pass@1', fontsize=20)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=20)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])

    handles, labels = ax1.get_legend_handles_labels()
    # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=24, bbox_to_anchor=(0.5, -0.21))
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_ret_recall_perplexity():
    graph_name = 'ret_recall_perplexity.pdf'
    qa_metric = 'perplexity'; code_metric = 'perplexity'
    gpt_perf_datas = []
    for dataset_name in qa_dataset_names:
        gpt_perf_datas.append(
            [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
    for dataset_name in code_dataset_names:
        gpt_perf_datas.append(
            [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
    llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        llama_perf_datas.append(
            [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
    for dataset_name in code_dataset_names:
        llama_perf_datas.append(
            [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
    x = range(len(gpt_perf_datas[0]))
    plt.style.use('ggplot')
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
    markers = ['D', 'o', '^']
    qa_colors, code_colors = colors[:4], colors[4:]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))  # ax1: qa, ax2: code
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
        line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
    ax1.set_xlabel('Retrieval Recall', fontsize=22)
    ax1.set_ylabel(f'{qa_metric}', fontsize=22)
    # ax1.set_yticks([1.055, 1.06, 1.065, 1.07, 1.075])
    # ax1.set_yticklabels([1.055, 1.06, 1.065, 1.07, 1.075], fontsize=20)
    ax1.set_yticks([1.05, 1.06, 1.07, 1.08])
    ax1.set_yticklabels([1.05, 1.06, 1.07, 1.08], fontsize=20)
    ax1.set_xticks(x, ret_recalls)
    ax1.set_xticklabels(ret_recalls, fontsize=20)
    ax1.set_title('Llama2-13B, QA Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
        line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
    ax2.set_xlabel('Retrieval Recall', fontsize=22)
    ax2.set_ylabel(code_metric, fontsize=22)
    ax2.set_yticks([1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
    ax2.set_yticklabels([1.11, 1.12, 1.13, 1.14, 1.15, 1.16], fontsize=16)
    ax2.set_xticks(x, ret_recalls)
    ax2.set_xticklabels(ret_recalls, fontsize=20)
    ax2.set_title('CodeLlama-13B, Code Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
        line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
    ax3.set_xlabel('Retrieval Recall', fontsize=22)
    ax3.set_ylabel(qa_metric, fontsize=22)
    # ax3.set_yticks([1.015, 1.025, 1.035, 1.045, 1.055])
    # ax3.set_yticklabels([1.015, 1.025, 1.035, 1.045, 1.055], fontsize=20)
    ax3.set_yticks([1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
    ax3.set_yticklabels([1.01, 1.02, 1.03, 1.04, 1.05, 1.06], fontsize=20)
    ax3.set_xticks(x, ret_recalls)
    ax3.set_xticklabels(ret_recalls, fontsize=20)
    ax3.set_title('GPT-3.5, QA Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
        line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
    ax4.set_xlabel('Retrieval Recall', fontsize=22)
    ax4.set_ylabel(code_metric, fontsize=22)
    # ax4.set_yticks([1.025, 1.03, 1.035, 1.04, 1.045, 1.05])
    # ax4.set_yticklabels([1.025, 1.03, 1.035, 1.04, 1.045, 1.05], fontsize=20)
    ax4.set_yticks([1.02, 1.03, 1.04, 1.05])
    ax4.set_yticklabels([1.02, 1.03, 1.04, 1.05], fontsize=20)
    ax4.set_xticks(x, ret_recalls)
    ax4.set_xticklabels(ret_recalls, fontsize=20)
    ax4.set_title('GPT-3.5, Code Datasets', fontsize=22)

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=24, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_ret_recall_analysis():
    graph_name = 'ret_recall_analysis.pdf'
    qa_metric = 'has_answer'; code_metric = 'pass@1'
    gpt_perf_datas = []
    for dataset_name in qa_dataset_names:
        gpt_perf_datas.append(
            [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
    for dataset_name in code_dataset_names:
        gpt_perf_datas.append(
            [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
    llama_perf_datas = []
    for dataset_name in qa_dataset_names:
        llama_perf_datas.append(
            [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])
    for dataset_name in code_dataset_names:
        llama_perf_datas.append(
            [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])
    gpt_perf_none = [results.qa_ret_doc_type_gpt_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
    gpt_perf_none.extend([results.code_ret_doc_type_gpt_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
    llama_perf_none = [results.qa_ret_doc_type_llama_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names]
    llama_perf_none.extend([results.code_ret_doc_type_llama_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])
    x = range(len(gpt_perf_datas[0]))
    plt.style.use('ggplot')
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
    markers = ['D', 'o', '^']
    qa_colors, code_colors = colors[:4], colors[4:]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))  # ax1: qa, ax2: code
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
        line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
        ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--')  # plot none result
    ax1.set_xlabel('Retrieval Recall', fontsize=22)
    ax1.set_ylabel(qa_metric, fontsize=22)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    ax1.set_xticks(x, ret_recalls)
    ax1.set_xticklabels(ret_recalls, fontsize=20)
    ax1.set_title('Llama2-13B, QA Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
        line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
        ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--')  # plot none result
    ax2.set_xlabel('Retrieval Recall', fontsize=22)
    ax2.set_ylabel(code_metric, fontsize=22)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=20)
    ax2.set_xticks(x, ret_recalls)
    ax2.set_xticklabels(ret_recalls, fontsize=20)
    ax2.set_title('CodeLlama-13B, Code Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
        line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
        ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--')   # plot none result
    ax3.set_xlabel('Retrieval Recall', fontsize=22)
    ax3.set_ylabel(qa_metric, fontsize=22)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    ax3.set_xticks(x, ret_recalls)
    ax3.set_xticklabels(ret_recalls, fontsize=20)
    ax3.set_title('GPT-3.5, QA Datasets', fontsize=22)
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
        line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
        ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--')   # plot none result
    ax4.set_xlabel('Retrieval Recall', fontsize=22)
    ax4.set_ylabel(code_metric, fontsize=22)
    ax4.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=20)
    ax4.set_xticks(x, ret_recalls)
    ax4.set_xticklabels(ret_recalls, fontsize=20)
    ax4.set_title('GPT-3.5, Code Datasets', fontsize=22)

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=24, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


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




def make_qa_code_ret_recall():
    graph_name = 'qa_code_ret_recall.pdf'
    qa_retrieval_acc_datas = []
    for retriever_index, retriever_name in enumerate(retriever_names):  # for qa
        if retriever_name == 'codeT5':
            qa_retrieval_acc_datas.append(None)
            continue
        qa_retrieval_acc_datas.append([0 for _ in top_ks])
        for dataset_name in qa_dataset_names:
            for top_k_index, top_k in enumerate(top_ks):
                qa_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
        for top_k_index in range(len(top_ks)):
            qa_retrieval_acc_datas[retriever_index][top_k_index] /= len(qa_dataset_names)
    code_retrieval_acc_datas = []
    for retriever_index, retriever_name in enumerate(retriever_names):  # for code
        if retriever_name == 'contriever':
            code_retrieval_acc_datas.append(None)
            continue
        code_retrieval_acc_datas.append([0 for _ in top_ks])
        for dataset_name in code_dataset_names:
            for top_k_index, top_k in enumerate(top_ks):
                code_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
        for top_k_index in range(len(top_ks)):
            code_retrieval_acc_datas[retriever_index][top_k_index] /= len(code_dataset_names)
    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
    markers = ['D', 'o', '^', '*', '+']
    qa_colors, code_colors = colors[:4], colors[4:]

    # x = range(len(qa_retrieval_acc_datas[0]))
    # x = [1, 3, 5, 10, 20, 50, 100]
    x = [1, 3, 6, 10, 15, 22, 33]
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))   # ax1: qa, ax2: code
    for idx, (retrieval_acc_data, retriever_name) in enumerate(zip(qa_retrieval_acc_datas, retriever_names)):
        if retrieval_acc_data: ax1.plot(x, retrieval_acc_data, marker=markers[idx], markersize=10, linestyle='-', label=retriever_name)
        else: ax1.plot([], [], marker='o', linestyle='-', label=retriever_name)
    ax1.set_xlabel('top k', fontsize=18)
    # ax1.set_ylabel('Retrieval Recall', fontsize=16)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    ax1.set_xticks(x, top_ks)
    ax1.set_xticklabels(x, fontsize=18)
    ax1.set_ylabel('Avg Retrieval Recall', fontsize=18)
    # ax1.set_xticks(top_ks)
    ax1.set_xticklabels(top_ks, fontsize=18)
    ax1.set_title('QA Datasets', fontsize=18)
    for idx, (retrieval_acc_data, retriever_name) in enumerate(zip(code_retrieval_acc_datas, retriever_names)):
        if retrieval_acc_data: ax2.plot(x, retrieval_acc_data, marker=markers[idx], markersize=10, linestyle='-', label=retriever_name)
        else: ax2.plot([], [], marker='o', linestyle='-', label=retriever_name)
    ax2.set_xlabel('top k', fontsize=18)
    # ax2.set_ylabel('Retrieval Recall', fontsize=16)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=18)
    ax2.set_xticks(x, top_ks)
    ax2.set_xticklabels(top_ks, fontsize=18)
    ax2.set_title('Code Generation Datasets', fontsize=18)
    ax2.set_ylabel('Avg Retrieval Recall', fontsize=18)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=16, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('graph/' + graph_name, bbox_inches='tight')
    plt.show()


def make_avg_ret_recall():
    graph_name = 'avg_retrieval_recall.pdf'
    avg_retrieval_acc_datas = []
    for retriever_index, retriever_name in enumerate(retriever_names):
        avg_retrieval_acc_datas.append([0 for _ in top_ks])
        for dataset_name in dataset_names:
            for top_k_index, top_k in enumerate(top_ks):
                avg_retrieval_acc_datas[retriever_index][top_k_index] += results.retrieval_accuracy[retriever_name][dataset_name][top_k]
        for top_k_index in range(len(top_ks)):
            avg_retrieval_acc_datas[retriever_index][top_k_index] /= len(dataset_names)

    x = range(len(avg_retrieval_acc_datas[0]))
    colors = ['b', 'g', 'r', 'c']
    plt.style.use('ggplot')
    for avg_retrieval_acc_data, retriever_name, color in zip(avg_retrieval_acc_datas, retriever_names, colors):
        plt.plot(x, avg_retrieval_acc_data, marker='o', linestyle='-', label=retriever_name)
    plt.xlabel('top k')
    plt.ylabel('Recall')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(x, top_ks)
    plt.title('Avg Retrieval Recall of Six Datasets for Retrievers')
    plt.legend()
    plt.savefig('graph/' + graph_name)
    plt.show()


def make_prompt_method_avg_correctness():
    graph_name = 'prompt_method_avg_correctness.pdf'

    prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
    qa_metric = 'has_answer'
    code_metric = 'pass@1'

    qa_gpt_avg_prompt_perf_datas = []
    qa_llama_avg_prompt_perf_datas = []
    code_gpt_avg_prompt_perf_datas = []
    code_llama_avg_prompt_perf_datas = []
    for prompt_method in prompt_method_list:
        try: avg_qa_gpt_data = sum([results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names])/3
        except: avg_qa_gpt_data = 0
        qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
        try: avg_qa_llama_data = sum([results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names])/3
        except: avg_qa_llama_data = 0
        qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
        try: avg_code_gpt_data = sum([results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names])/3
        except: avg_code_gpt_data = 0
        code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
        try: avg_code_llama_data = sum([results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names])/3
        except: avg_code_llama_data = 0
        code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)

    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
    plt.style.use('ggplot')
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(12, 24), sharex=True)  # ax1: qa, ax2: code
    axes = [ax1, ax2, ax3, ax4]
    perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas, code_llama_avg_prompt_perf_datas]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
        ax.bar(range(len(perf_datas)), perf_datas, width=0.6)
        if ax_idx == 3:
            ax.set_xticks(range(len(perf_datas)))
            ax.set_xticklabels(prompt_method_list, rotation=45, ha='right', fontsize=16)
        else:
            ax.set_xticks([])
        if ax_idx in [0, 1]:
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            ax.set_ylabel(qa_metric, fontsize=16)
        else:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
            ax.set_ylabel(code_metric, fontsize=16)

    plt.tight_layout()
    plt.savefig('graph/' + graph_name)
    plt.show()


def make_prompt_method_correctness():
    graph_name = 'prompt_method_correctness.pdf'

    # prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
    # auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN', 'IR-CoT', 'FLARE']
    prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
    auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN']
    qa_metric = 'has_answer'
    code_metric = 'pass@1'

    qa_gpt_avg_prompt_perf_datas = []
    qa_llama_avg_prompt_perf_datas = []
    code_gpt_avg_prompt_perf_datas = []
    code_llama_avg_prompt_perf_datas = []
    for prompt_method in prompt_method_list:
        try:
            avg_qa_gpt_data = [results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
            avg_qa_gpt_data.append(sum(avg_qa_gpt_data)/3)  # avg
        except: avg_qa_gpt_data = [0, 0, 0, 0]
        qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
        try:
            avg_qa_llama_data = [results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
            avg_qa_llama_data.append(sum(avg_qa_llama_data) / 3)  # avg
        except: avg_qa_llama_data = [0, 0, 0, 0, 0]
        qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
        try:
            avg_code_gpt_data = [results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
            avg_code_gpt_data.append(sum(avg_code_gpt_data) / 3)  # avg
        except: avg_code_gpt_data = [0, 0, 0, 0]
        code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
        try:
            avg_code_llama_data = [results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
            avg_code_llama_data.append(sum(avg_code_llama_data) / 3)  # avg
        except: avg_code_llama_data = [0, 0, 0, 0]
        code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)

    # colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585']
    # colors = ['skyblue', 'lightgreen', 'salmon']
    # colors = ['skyblue', 'lightgreen', 'tomato', '#228B22']
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']
    colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
    hatches = ['/', '||', 'X', 'O']
    edge_colors = ['red', 'green', 'green', 'blue']
    edge_colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
    edge_colors = ['#4D4D4D']*4
    line_colors = ['#8B0000', '#FF6347', '#228B22', '#4169E1']
    plt.style.use('ggplot')
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(24, 12))  # ax1: qa, ax2: code
    axes = [ax1, ax2, ax3, ax4]
    perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas,
                       code_llama_avg_prompt_perf_datas]
    for perf_datas in perf_datas_list:
        print(len(perf_datas))
        for perf_data in perf_datas:
            print(len(perf_data))
    auth_qa_dataset_names.append('avg of QA tasks')
    auth_code_dataset_names.append('avg of code tasks')

    special_idx = 1
    bar_width = 0.2
    prompt_method_list.remove('0shot')
    index = np.arange(len(prompt_method_list))
    for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
        if ax_idx in [0,1]:
            dataset_names = auth_qa_dataset_names
        else:
            dataset_names = auth_code_dataset_names
        for dataset_idx, dataset_name in enumerate(dataset_names):
            offset = dataset_idx * bar_width
            bar_data = [item[dataset_idx] for item in perf_datas][1:]
            hatch_styles = [hatches[dataset_idx] if idx != special_idx or ax_idx in [2,3] else 'XX' for idx in index]  # Dense hatch for unreliable

            ax.bar(index+offset, bar_data, width=bar_width, label=dataset_name,
                   color=colors[dataset_idx], hatch=hatch_styles, edgecolor=edge_colors[dataset_idx])
            ax.axhline(y=[item[dataset_idx] for item in perf_datas][0], color=line_colors[dataset_idx], linestyle=':', linewidth=4)  # plot baseline result
        if ax_idx == 3 or ax_idx == 1:
            ax.set_xticks(index+bar_width*1.5)
            ax.set_xticklabels(auth_prompt_method_names[1:], rotation=45, ha='right', fontsize=22)
        else:
            ax.set_xticks([])
        if ax_idx in [0, 1]:
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
            ax.set_ylabel(qa_metric, fontsize=24)
        else:
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
            ax.set_ylabel(code_metric, fontsize=24)
        ax.legend(loc='upper right', fontsize=18, ncol=4)

        if ax_idx in [0,1]:
            bar_center = index[special_idx] + bar_width
            # ax.text(bar_center, 0.9, 'Unreliable', ha='center', fontsize=12, color='black')
            ax.annotate('Unreliable', xy=(bar_center, 0.75), xytext=(bar_center + 0.3, 0.82),
                        arrowprops=dict(facecolor='black', shrink=0.01), fontsize=14)

    plt.tight_layout()
    plt.savefig('graph/' + graph_name)
    plt.show()



def make_prompt_method_perplexity():
    graph_name = 'prompt_method_perplexity.pdf'

    # prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'self-consistency', 'least_to_most', 'plan_and_solve', 'self-refine', 'con', 'ir-cot', 'flare']
    # auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Self-Consistency', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN', 'IR-CoT', 'FLARE']
    prompt_method_list = ['0shot', '3shot', 'RaR', 'cot', 'least_to_most', 'plan_and_solve', 'self-refine', 'con']
    auth_prompt_method_names = ['zero-shot', 'few-shot', 'RaR', 'CoT', 'Least-to-Most', 'Plan-and-Solve', 'Self-Refine', 'CoN']
    qa_metric = 'perplexity'
    code_metric = 'perplexity'

    qa_gpt_avg_prompt_perf_datas = []
    qa_llama_avg_prompt_perf_datas = []
    code_gpt_avg_prompt_perf_datas = []
    code_llama_avg_prompt_perf_datas = []
    for prompt_method in prompt_method_list:
        avg_qa_gpt_data = [results.prompt_method_gpt[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
        avg_qa_gpt_data = sum(avg_qa_gpt_data)/3  # avg
        qa_gpt_avg_prompt_perf_datas.append(avg_qa_gpt_data)
        avg_qa_llama_data = [results.prompt_method_llama[qa_dataset_name][prompt_method][qa_metric] for qa_dataset_name in qa_dataset_names]
        avg_qa_llama_data = sum(avg_qa_llama_data)/3  # avg
        qa_llama_avg_prompt_perf_datas.append(avg_qa_llama_data)
        avg_code_gpt_data = [results.prompt_method_gpt[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
        avg_code_gpt_data = sum(avg_code_gpt_data)/3  # avg
        code_gpt_avg_prompt_perf_datas.append(avg_code_gpt_data)
        avg_code_llama_data = [results.prompt_method_llama[code_dataset_name][prompt_method][code_metric] for code_dataset_name in code_dataset_names]
        avg_code_llama_data = sum(avg_code_llama_data)/3  # avg
        code_llama_avg_prompt_perf_datas.append(avg_code_llama_data)


    colors = ['#D62728', '#FF9896', '#2CA02C', '#1F77B4']
    edge_colors = ['#4D4D4D']*4
    plt.style.use('ggplot')
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 8))  # ax1: qa, ax2: code
    axes = [ax1, ax2, ax3, ax4]
    perf_datas_list = [qa_gpt_avg_prompt_perf_datas, qa_llama_avg_prompt_perf_datas, code_gpt_avg_prompt_perf_datas,
                       code_llama_avg_prompt_perf_datas]

    bar_width = 0.3
    index = np.arange(len(prompt_method_list))
    for ax_idx, (ax, perf_datas) in enumerate(zip(axes, perf_datas_list)):
        if ax_idx in [0,1]:
            dataset_name = 'avg of QA tasks'
        else:
            dataset_name = 'avg of code tasks'

        ax.bar(index, perf_datas, width=bar_width, label=dataset_name, color=colors[3], edgecolor=edge_colors[3])
        if ax_idx == 3 or ax_idx == 1:
            ax.set_xticks(index)
            ax.set_xticklabels(auth_prompt_method_names, rotation=90, ha='right', fontsize=22)
        else:
            ax.set_xticks([])
        if ax_idx in [0, 1]:
            ax.set_ylim(1, 1.3)
            ax.set_yticks([1, 1.1, 1.2, 1.3])
            ax.set_yticklabels([1, 1.1, 1.2, 1.3], fontsize=20)
            ax.set_ylabel('PPL', fontsize=24)
        else:
            ax.set_ylim(1, 1.3)
            ax.set_yticks([1, 1.1, 1.2, 1.3])
            ax.set_yticklabels([1, 1.1, 1.2, 1.3], fontsize=20)
            ax.set_ylabel('PPL', fontsize=24)
        ax.legend(loc='upper right', fontsize=18, ncol=1)

    plt.tight_layout()
    plt.savefig('graph/' + graph_name)
    plt.show()


if __name__ == '__main__':
    # make_avg_ret_recall()

    # make_qa_code_ret_recall()

    # make_ret_recall_analysis()

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

    make_prompt_method_perplexity()

    # make_doc_selection_percentage_of_mistakes()
