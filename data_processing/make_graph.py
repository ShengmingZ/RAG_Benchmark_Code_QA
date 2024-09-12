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

    axs = [ax1, ax2, ax3, ax4]
    perf_datas_list = [code_llama_perf_datas, qa_llama_perf_datas, code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_llama_doc_selection_types, qa_llama_doc_selection_types, code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            metric = code_metric
            tmp_colors = colors[4:]
        else:
            dataset_names = qa_dataset_names
            metric = qa_metric
            tmp_colors = colors[:4]
        for idx, dataset_name in enumerate(dataset_names):
            ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker='o', linestyle='-', label=dataset_name, color=tmp_colors[idx])
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xlabel('top k documents', fontsize=14)
        ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
        ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=12)
        if ax_idx in [1,3]:
            ax.set_yticks([0.3, 0.5, 0.7, 0.9])
            ax.set_yticklabels([0.3, 0.5, 0.7, 0.9], fontsize=12)
        elif ax_idx == 0:
            ax.set_yticks([0.1, 0.3, 0.5, 0.7])
            ax.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize=12)
        elif ax_idx == 2:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=12)
        if ax_idx == 0:
            ax.set_title('Llama2-13B, Code Generation Datasets', fontsize=11)
        elif ax_idx == 1:
            ax.set_title('Llama2-13B, QA Datasets', fontsize=11)
        elif ax_idx == 2:
            ax.set_title('GPT-3.5, Code Generation Datasets', fontsize=11)
        else:
            ax.set_title('GPT-3.5, QA Datasets', fontsize=11)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=13, bbox_to_anchor=(0.5, -0.2))
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
    qa_dataset_names.append('qa avg')
    code_dataset_names.append('code avg')

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

    axs = [ax1, ax2, ax3, ax4]
    perf_datas_list = [code_llama_perf_datas, qa_llama_perf_datas, code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_llama_doc_selection_types, qa_llama_doc_selection_types, code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            tmp_colors = colors[:4]
            metric = code_metric
        else:
            dataset_names = qa_dataset_names
            tmp_colors = colors[4:]
            metric = qa_metric
        for idx, dataset_name in enumerate(dataset_names):
            ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker='o', linestyle='-', label=dataset_name, color=tmp_colors[idx])
        ax.set_ylabel(metric)
        ax.set_xlabel('top k documents', fontsize=12)
        if ax_idx == 0: ax.set_yticks([1.12, 1.17])
        elif ax_idx == 1: ax.set_yticks([1.05, 1.1])
        elif ax_idx == 2: ax.set_yticks([1.02, 1.07])
        elif ax_idx == 3: ax.set_yticks([1.02, 1.07])
        if ax_idx == 0:
            ax.set_title('(1) Code Datasets, Llama2-13B', fontsize=12)
        elif ax_idx == 1:
            ax.set_title('(2) QA Datasets, Llama2-13B', fontsize=12)
        elif ax_idx == 2:
            ax.set_title('(3) Code Datasets, GPT-3.5', fontsize=12)
        else:
            ax.set_title('(4) QA Datasets, GPT-3.5', fontsize=12)

    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax4.get_legend_handles_labels()
    handles, labels = ax3_handles + ax4_handles, ax3_labels + ax4_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=12, bbox_to_anchor=(0.5, -0.18))
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

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.75, 1])
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    ax2 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    colors = ['#E03C31', '#FF6F00', '#FFB300', '#BA55D3', '#4169E1', '#00BFFF', '#708090', '#228B22']

    axs = [ax1, ax2]
    perf_datas_list = [code_gpt_perf_datas, qa_gpt_perf_datas]
    topk_list = [code_gpt_doc_selection_types, qa_gpt_doc_selection_types]
    for ax_idx, (ax, perf_datas) in enumerate(zip(axs, perf_datas_list)):
        if ax_idx%2 == 0:
            dataset_names = code_dataset_names
            tmp_colors = colors[4:]
            metric = code_metric
        else:
            dataset_names = qa_dataset_names
            tmp_colors = colors[:4]
            metric = qa_metric
        for idx, dataset_name in enumerate(dataset_names):
            ax.plot([item.split('_')[1] for item in topk_list[ax_idx]], perf_datas[idx], marker='o', linestyle='-', label=dataset_name, color=tmp_colors[idx])
        ax.set_ylabel('Retrieval Recall', fontsize=14)
        ax.set_xlabel('top k documents', fontsize=14)
        if ax_idx == 0:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4], fontsize=14)
        elif ax_idx == 1:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
        ax.set_xticks([item.split('_')[1] for item in topk_list[ax_idx]])
        ax.set_xticklabels([item.split('_')[1] for item in topk_list[ax_idx]], fontsize=14)
        if ax_idx == 0:
            ax.set_title('Code Generation Datasets', fontsize=14)
        elif ax_idx == 1:
            ax.set_title('QA Datasets', fontsize=14)

    ax3_handles, ax3_labels = ax1.get_legend_handles_labels()
    ax4_handles, ax4_labels = ax2.get_legend_handles_labels()
    handles, labels = ax4_handles + ax3_handles, ax4_labels + ax3_labels
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.21))
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
    fig.text(0.5, 0.03, '(2). perplexity of Llama2-13B over six datasets', ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.48, '(1). perplexity of GPT-3.5 over six datasets', ha='center', va='center', fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    doc_type_index = np.arange(x)
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
    # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
    axs = [ax1, ax2, ax3]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        # ax.set_xlabel(qa_dataset_names[idx], fontsize=16)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        # ax.set_ylabel('perplexity', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax4, ax5, ax6]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        # ax.set_ylabel('perplexity', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax7, ax8, ax9]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        # ax.set_ylabel('perplexity', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax10, ax11, ax12]
    yticks_list = [[1, 1.2], [1, 1.2], [1, 1.2]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        # ax.set_ylabel('perplexity', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])

    handles, labels = ax1.get_legend_handles_labels()
    # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=16, bbox_to_anchor=(0.5, -0.1))
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
    fig.text(0.5, 0.03, '(2). performance of Llama2-13B over six datasets', ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.48, '(1). performance of GPT-3.5 over six datasets', ha='center', va='center', fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    x = len(ret_doc_types)
    doc_type_index = np.arange(x)
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(ret_doc_types)))
    # colors2 = plt.cm.plasma(np.linspace(0.5, 1, len(code_gpt_perf_datas)))
    axs = [ax1, ax2, ax3]
    yticks_list = [[0,1], [0,1], [0,1]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_gpt_perf_datas[idx])), qa_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        ax.set_ylabel(metric, fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax4, ax5, ax6]
    yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_gpt_perf_datas[idx])), code_gpt_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        ax.set_ylabel('pass@1', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax7, ax8, ax9]
    yticks_list = [[0, 1], [0, 1], [0, 1]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(qa_llama_perf_datas[idx])), qa_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_qa_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        ax.set_ylabel(metric, fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])
    axs = [ax10, ax11, ax12]
    yticks_list = [[0, 0.4], [0, 0.4], [0.4, 0.8]]
    for idx, ax in enumerate(axs):
        ax.bar(range(len(code_llama_perf_datas[idx])), code_llama_perf_datas[idx], label=ret_doc_types, color=colors1)
        ax.set_xlabel(auth_code_dataset_names[idx], fontsize=16)
        ax.set_xticks([])
        ax.set_ylabel('pass@1', fontsize=16)
        ax.set_yticks(yticks_list[idx])
        ax.set_yticklabels(yticks_list[idx], fontsize=16)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.set_ylim(yticks_list[idx][0], yticks_list[idx][-1])

    handles, labels = ax1.get_legend_handles_labels()
    # ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    # handles, labels = ax1_handles + ax2_handles, ax1_labels + ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=16, bbox_to_anchor=(0.5, -0.08))
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    qa_colors, code_colors = colors[:3], colors[3:]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))  # ax1: qa, ax2: code
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
        line, = ax1.plot(x, perf_data, marker='o', linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
    ax1.set_xlabel('Retrieval Recall', fontsize=16)
    # ax1.set_ylabel(f'{qa_metric}')
    ax1.set_yticks([1.055, 1.06, 1.065, 1.07, 1.075])
    ax1.set_yticklabels([1.055, 1.06, 1.065, 1.07, 1.075], fontsize=16)
    ax1.set_xticks(x, ret_recalls)
    ax1.set_xticklabels(ret_recalls, fontsize=16)
    ax1.set_title('Llama2-13B, QA Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
        line, = ax2.plot(x, perf_data, marker='o', linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
    ax2.set_xlabel('Retrieval Recall', fontsize=16)
    # ax2.set_ylabel(code_metric)
    ax2.set_yticks([1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
    ax2.set_yticklabels([1.11, 1.12, 1.13, 1.14, 1.15, 1.16], fontsize=16)
    ax2.set_xticks(x, ret_recalls)
    ax2.set_xticklabels(ret_recalls, fontsize=16)
    ax2.set_title('Llama2-13B, Code Generation Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
        line, = ax3.plot(x, perf_data, marker='o', linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
    ax3.set_xlabel('Retrieval Recall', fontsize=16)
    # ax3.set_ylabel(qa_metric)
    ax3.set_yticks([1.015, 1.025, 1.035, 1.045, 1.055])
    ax3.set_yticklabels([1.015, 1.025, 1.035, 1.045, 1.055], fontsize=16)
    ax3.set_xticks(x, ret_recalls)
    ax3.set_xticklabels(ret_recalls, fontsize=16)
    ax3.set_title('GPT-3.5, QA Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
        line, = ax4.plot(x, perf_data, marker='o', linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
    ax4.set_xlabel('Retrieval Recall', fontsize=16)
    # ax4.set_ylabel(code_metric)
    ax4.set_yticks([1.025, 1.03, 1.035, 1.04, 1.045, 1.05])
    ax4.set_yticklabels([1.025, 1.03, 1.035, 1.04, 1.045, 1.05], fontsize=16)
    ax4.set_xticks(x, ret_recalls)
    ax4.set_xticklabels(ret_recalls, fontsize=16)
    ax4.set_title('GPT-3.5, Code Generation Datasets')

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=16, bbox_to_anchor=(0.5, -0.08))
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
    qa_colors, code_colors = colors[:4], colors[4:]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))  # ax1: qa, ax2: code
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):
        line, = ax1.plot(x, perf_data, marker='o', linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
        ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--')  # plot none result
    ax1.set_xlabel('Retrieval Recall', fontsize=16)
    ax1.set_ylabel(qa_metric, fontsize=16)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
    ax1.set_xticks(x, ret_recalls)
    ax1.set_xticklabels(ret_recalls, fontsize=16)
    ax1.set_title('Llama2-13B, QA Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):
        line, = ax2.plot(x, perf_data, marker='o', linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
        ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--')  # plot none result
    ax2.set_xlabel('Retrieval Recall', fontsize=16)
    ax2.set_ylabel(code_metric, fontsize=16)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=16)
    ax2.set_xticks(x, ret_recalls)
    ax2.set_xticklabels(ret_recalls, fontsize=16)
    ax2.set_title('Llama2-13B, Code Generation Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):
        line, = ax3.plot(x, perf_data, marker='o', linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])
        ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--')   # plot none result
    ax3.set_xlabel('Retrieval Recall', fontsize=16)
    ax3.set_ylabel(qa_metric, fontsize=16)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
    ax3.set_xticks(x, ret_recalls)
    ax3.set_xticklabels(ret_recalls, fontsize=16)
    ax3.set_title('GPT-3.5, QA Datasets')
    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):
        line, = ax4.plot(x, perf_data, marker='o', linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])
        ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--')   # plot none result
    ax4.set_xlabel('Retrieval Recall', fontsize=16)
    ax4.set_ylabel(code_metric, fontsize=16)
    ax4.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=16)
    ax4.set_xticks(x, ret_recalls)
    ax4.set_xticklabels(ret_recalls, fontsize=16)
    ax4.set_title('GPT-3.5, Code Generation Datasets')

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=16, bbox_to_anchor=(0.5, -0.08))
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

    x = range(len(qa_retrieval_acc_datas[0]))
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))   # ax1: qa, ax2: code
    for retrieval_acc_data, retriever_name in zip(qa_retrieval_acc_datas, retriever_names):
        if retrieval_acc_data: ax1.plot(x, retrieval_acc_data, marker='o', linestyle='-', label=retriever_name)
        else: ax1.plot([], [], marker='o', linestyle='-', label=retriever_name)
    ax1.set_xlabel('top k', fontsize=18)
    # ax1.set_ylabel('Retrieval Recall', fontsize=16)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    ax1.set_xticks(x, top_ks)
    ax1.set_ylabel('Avg Retrieval Recall', fontsize=18)
    # ax1.set_xticks(top_ks)
    ax1.set_xticklabels(top_ks, fontsize=18)
    ax1.set_title('QA Datasets', fontsize=18)
    for retrieval_acc_data, retriever_name in zip(code_retrieval_acc_datas, retriever_names):
        if retrieval_acc_data: ax2.plot(x, retrieval_acc_data, marker='o', linestyle='-', label=retriever_name)
        else: ax2.plot([], [], marker='o', linestyle='-', label=retriever_name)
    ax2.set_xlabel('top k', fontsize=16)
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

    make_doc_selection_topk_perplexity()

    # make_doc_selection_topk_ret_recall()
