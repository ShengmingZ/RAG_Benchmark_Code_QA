[1mdiff --git a/data_processing/graph/ret_recall_analysis.pdf b/data_processing/graph/ret_recall_analysis.pdf[m
[1mindex d9571f8..717f60d 100644[m
Binary files a/data_processing/graph/ret_recall_analysis.pdf and b/data_processing/graph/ret_recall_analysis.pdf differ
[1mdiff --git a/data_processing/make_graph.py b/data_processing/make_graph.py[m
[1mindex 6e11a4d..de5dc42 100644[m
[1m--- a/data_processing/make_graph.py[m
[1m+++ b/data_processing/make_graph.py[m
[36m@@ -808,84 +808,181 @@[m [mdef make_ret_recall_perplexity():[m
     plt.show()[m
 [m
 [m
[32m+[m[32m# Alternative refactored version with helper functions[m
[32m+[m[32mdef create_subplot(ax, perf_datas, dataset_names, colors, markers, x_positions, ret_recalls, title, ylabel, yticks, yticklabels):[m
[32m+[m[32m    """Helper function to create individual subplots"""[m
[32m+[m[32m    for idx, (perf_data, dataset_name) in enumerate(zip(perf_datas, dataset_names)):[m
[32m+[m[32m        ax.plot(x_positions, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=dataset_name, color=colors[idx])[m
[32m+[m
[32m+[m[32m    ax.set_xlabel('Retrieval Recall', fontsize=22)[m
[32m+[m[32m    ax.set_ylabel(ylabel, fontsize=22)[m
[32m+[m[32m    ax.set_yticks(yticks)[m
[32m+[m[32m    ax.set_yticklabels(yticklabels, fontsize=20)[m
[32m+[m[32m    ax.set_xticks(x_positions)[m
[32m+[m[32m    ax.set_xticklabels(ret_recalls, fontsize=20)[m
[32m+[m[32m    ax.set_title(title, fontsize=22)[m
[32m+[m[32m    ax.grid(True, alpha=0.3)[m
[32m+[m
[32m+[m
 def make_ret_recall_analysis():[m
[32m+[m[32m    """[m
[32m+[m[32m    Refactored version with helper functions for better code organization[m
[32m+[m[32m    """[m
[32m+[m[32m    # Configuration[m
     graph_name = 'ret_recall_analysis.pdf'[m
[31m-    qa_metric = 'has_answer'; code_metric = 'pass@1'[m
[32m+[m[32m    qa_metric = 'has_answer'[m
[32m+[m[32m    code_metric = 'pass@1'[m
[32m+[m
[32m+[m[32m    # Colors and styling[m
[32m+[m[32m    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1',[m
[32m+[m[32m              '#228B22', '#4169E1', '#8B4513', '#C71585'][m
[32m+[m[32m    markers = ['D', 'o', '^'][m
[32m+[m[32m    qa_colors = colors[:4][m
[32m+[m[32m    code_colors = colors[4:][m
[32m+[m
[32m+[m[32m    # Data extraction (same as above)[m
     gpt_perf_datas = [][m
     for dataset_name in qa_dataset_names:[m
[31m-        gpt_perf_datas.append([m
[31m-            [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m        gpt_perf_datas.append([[m
[32m+[m[32m            results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric][m
[32m+[m[32m            for ret_recall in ret_recalls[m
[32m+[m[32m        ])[m
     for dataset_name in code_dataset_names:[m
[31m-        gpt_perf_datas.append([m
[31m-            [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m        gpt_perf_datas.append([[m
[32m+[m[32m            results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric][m
[32m+[m[32m            for ret_recall in ret_recalls[m
[32m+[m[32m        ])[m
[32m+[m
     llama_perf_datas = [][m
     for dataset_name in qa_dataset_names:[m
[31m-        llama_perf_datas.append([m
[31m-            [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m        llama_perf_datas.append([[m
[32m+[m[32m            results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric][m
[32m+[m[32m            for ret_recall in ret_recalls[m
[32m+[m[32m        ])[m
     for dataset_name in code_dataset_names:[m
[31m-        llama_perf_datas.append([m
[31m-            [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])[m
[31m-    gpt_perf_none = [results.qa_ret_doc_type_gpt_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names][m
[31m-    gpt_perf_none.extend([results.code_ret_doc_type_gpt_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])[m
[31m-    llama_perf_none = [results.qa_ret_doc_type_llama_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names][m
[31m-    llama_perf_none.extend([results.code_ret_doc_type_llama_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])[m
[31m-    x = range(len(gpt_perf_datas[0]))[m
[31m-    plt.style.use('ggplot')[m
[31m-    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf'][m
[31m-    colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585'][m
[31m-    markers = ['D', 'o', '^'][m
[31m-    qa_colors, code_colors = colors[:4], colors[4:][m
[31m-    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # ax1: qa, ax2: code[m
[31m-    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):[m
[31m-        line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])[m
[31m-        # ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result[m
[31m-    ax1.set_xlabel('Retrieval Recall', fontsize=22)[m
[31m-    ax1.set_ylabel('Accuracy', fontsize=22)[m
[31m-    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[31m-    ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)[m
[31m-    ax1.set_xticks(x, ret_recalls)[m
[31m-    ax1.set_xticklabels(ret_recalls, fontsize=20)[m
[31m-    ax1.set_title('a. Llama2-13B, QA Tasks', fontsize=22)[m
[31m-    for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):[m
[31m-        line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])[m
[31m-        # ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result[m
[31m-    ax2.set_xlabel('Retrieval Recall', fontsize=22)[m
[31m-    ax2.set_ylabel('Pass@1', fontsize=22)[m
[31m-    ax2.set_yticks([0, 0.2, 0.4, 0.6])[m
[31m-    ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=20)[m
[31m-    ax2.set_xticks(x, ret_recalls)[m
[31m-    ax2.set_xticklabels(ret_recalls, fontsize=20)[m
[31m-    ax2.set_title('b. Llama2-13B, Code Tasks', fontsize=22)[m
[31m-    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):[m
[31m-        line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])[m
[31m-        # ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result[m
[31m-    ax3.set_xlabel('Retrieval Recall', fontsize=22)[m
[31m-    ax3.set_ylabel('Accuracy', fontsize=22)[m
[31m-    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[31m-    ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)[m
[31m-    ax3.set_xticks(x, ret_recalls)[m
[31m-    ax3.set_xticklabels(ret_recalls, fontsize=20)[m
[31m-    ax3.set_title('c. GPT-3.5, QA Tasks', fontsize=22)[m
[31m-    for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):[m
[31m-        line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])[m
[31m-        # ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result[m
[31m-    ax4.set_xlabel('Retrieval Recall', fontsize=22)[m
[31m-    ax4.set_ylabel('Pass@1', fontsize=22)[m
[31m-    ax4.set_yticks([0.2, 0.4, 0.6, 0.8])[m
[31m-    ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=20)[m
[31m-    ax4.set_xticks(x, ret_recalls)[m
[31m-    ax4.set_xticklabels(ret_recalls, fontsize=20)[m
[31m-    ax4.set_title('d. GPT-3.5, Code Tasks', fontsize=22)[m
[32m+[m[32m        llama_perf_datas.append([[m
[32m+[m[32m            results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric][m
[32m+[m[32m            for ret_recall in ret_recalls[m
[32m+[m[32m        ])[m
 [m
[32m+[m[32m    # Create figure[m
[32m+[m[32m    plt.style.use('ggplot')[m
[32m+[m[32m    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))[m
[32m+[m[32m    x_positions = range(len(ret_recalls))[m
[32m+[m
[32m+[m[32m    # Create subplots using helper function[m
[32m+[m[32m    create_subplot(ax1, llama_perf_datas[:3], auth_qa_dataset_names,[m
[32m+[m[32m                   qa_colors, markers, x_positions, ret_recalls,[m
[32m+[m[32m                   'a. Llama2-13B, QA Tasks', 'Accuracy',[m
[32m+[m[32m                   [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[32m+[m
[32m+[m[32m    create_subplot(ax2, llama_perf_datas[3:], auth_code_dataset_names,[m
[32m+[m[32m                   code_colors, markers, x_positions, ret_recalls,[m
[32m+[m[32m                   'b. Llama2-13B, Code Tasks', 'Pass@1',[m
[32m+[m[32m                   [0, 0.2, 0.4, 0.6], [0, 0.2, 0.4, 0.6])[m
[32m+[m
[32m+[m[32m    create_subplot(ax3, gpt_perf_datas[:3], auth_qa_dataset_names,[m
[32m+[m[32m                   qa_colors, markers, x_positions, ret_recalls,[m
[32m+[m[32m                   'c. GPT-3.5, QA Tasks', 'Accuracy',[m
[32m+[m[32m                   [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[32m+[m
[32m+[m[32m    create_subplot(ax4, gpt_perf_datas[3:], auth_code_dataset_names,[m
[32m+[m[32m                   code_colors, markers, x_positions, ret_recalls,[m
[32m+[m[32m                   'd. GPT-3.5, Code Tasks', 'Pass@1',[m
[32m+[m[32m                   [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8])[m
[32m+[m
[32m+[m[32m    # Create shared legend[m
     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()[m
     ax2_handles, ax2_labels = ax2.get_legend_handles_labels()[m
[31m-    handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels[m
[31m-    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.12))[m
[32m+[m
[32m+[m[32m    fig.legend(ax1_handles + ax2_handles, ax1_labels + ax2_labels,[m
[32m+[m[32m               loc='lower center', ncol=3, fontsize=24,[m
[32m+[m[32m               bbox_to_anchor=(0.5, -0.12))[m
[32m+[m
     plt.tight_layout()[m
[31m-    plt.savefig('graph/' + graph_name, bbox_inches='tight')[m
[32m+[m[32m    plt.savefig(f'graph/{graph_name}', bbox_inches='tight', dpi=300)[m
     plt.show()[m
 [m
 [m
[32m+[m
[32m+[m[32m# def make_ret_recall_analysis():[m
[32m+[m[32m#     graph_name = 'ret_recall_analysis.pdf'[m
[32m+[m[32m#     qa_metric = 'has_answer'; code_metric = 'pass@1'[m
[32m+[m[32m#     gpt_perf_datas = [][m
[32m+[m[32m#     for dataset_name in qa_dataset_names:[m
[32m+[m[32m#         gpt_perf_datas.append([m
[32m+[m[32m#             [results.qa_ret_recall_gpt_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m#     for dataset_name in code_dataset_names:[m
[32m+[m[32m#         gpt_perf_datas.append([m
[32m+[m[32m#             [results.code_ret_recall_gpt_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m#     llama_perf_datas = [][m
[32m+[m[32m#     for dataset_name in qa_dataset_names:[m
[32m+[m[32m#         llama_perf_datas.append([m
[32m+[m[32m#             [results.qa_ret_recall_llama_n_1[dataset_name][ret_recall][qa_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m#     for dataset_name in code_dataset_names:[m
[32m+[m[32m#         llama_perf_datas.append([m
[32m+[m[32m#             [results.code_ret_recall_llama_n_1[dataset_name][ret_recall][code_metric] for ret_recall in ret_recalls])[m
[32m+[m[32m#     gpt_perf_none = [results.qa_ret_doc_type_gpt_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names][m
[32m+[m[32m#     gpt_perf_none.extend([results.code_ret_doc_type_gpt_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])[m
[32m+[m[32m#     llama_perf_none = [results.qa_ret_doc_type_llama_n_1[dataset_name]['none'][qa_metric] for dataset_name in qa_dataset_names][m
[32m+[m[32m#     llama_perf_none.extend([results.code_ret_doc_type_llama_n_1[dataset_name]['none'][code_metric] for dataset_name in code_dataset_names])[m
[32m+[m[32m#     x = range(len(gpt_perf_datas[0]))[m
[32m+[m[32m#     plt.style.use('ggplot')[m
[32m+[m[32m#     # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf'][m
[32m+[m[32m#     colors = ['#DC143C', '#FF8C00', '#DAA520', '#FFB6C1', '#228B22', '#4169E1', '#8B4513', '#C71585'][m
[32m+[m[32m#     markers = ['D', 'o', '^'][m
[32m+[m[32m#     qa_colors, code_colors = colors[:4], colors[4:][m
[32m+[m[32m#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # ax1: qa, ax2: code[m
[32m+[m[32m#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[:3], qa_dataset_names)):[m
[32m+[m[32m#         line, = ax1.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])[m
[32m+[m[32m#         # ax1.axhline(y=llama_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result[m
[32m+[m[32m#     ax1.set_xlabel('Retrieval Recall', fontsize=22)[m
[32m+[m[32m#     ax1.set_ylabel('Accuracy', fontsize=22)[m
[32m+[m[32m#     ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[32m+[m[32m#     ax1.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)[m
[32m+[m[32m#     ax1.set_xticks(x, ret_recalls)[m
[32m+[m[32m#     ax1.set_xticklabels(ret_recalls, fontsize=20)[m
[32m+[m[32m#     ax1.set_title('a. Llama2-13B, QA Tasks', fontsize=22)[m
[32m+[m[32m#     for idx, (perf_data, dataset_name) in enumerate(zip(llama_perf_datas[3:], code_dataset_names)):[m
[32m+[m[32m#         line, = ax2.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])[m
[32m+[m[32m#         # ax2.axhline(y=llama_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)  # plot none result[m
[32m+[m[32m#     ax2.set_xlabel('Retrieval Recall', fontsize=22)[m
[32m+[m[32m#     ax2.set_ylabel('Pass@1', fontsize=22)[m
[32m+[m[32m#     ax2.set_yticks([0, 0.2, 0.4, 0.6])[m
[32m+[m[32m#     ax2.set_yticklabels([0, 0.2, 0.4, 0.6], fontsize=20)[m
[32m+[m[32m#     ax2.set_xticks(x, ret_recalls)[m
[32m+[m[32m#     ax2.set_xticklabels(ret_recalls, fontsize=20)[m
[32m+[m[32m#     ax2.set_title('b. Llama2-13B, Code Tasks', fontsize=22)[m
[32m+[m[32m#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[:3], qa_dataset_names)):[m
[32m+[m[32m#         line, = ax3.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_qa_dataset_names[idx], color=qa_colors[idx])[m
[32m+[m[32m#         # ax3.axhline(y=gpt_perf_none[:3][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result[m
[32m+[m[32m#     ax3.set_xlabel('Retrieval Recall', fontsize=22)[m
[32m+[m[32m#     ax3.set_ylabel('Accuracy', fontsize=22)[m
[32m+[m[32m#     ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])[m
[32m+[m[32m#     ax3.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)[m
[32m+[m[32m#     ax3.set_xticks(x, ret_recalls)[m
[32m+[m[32m#     ax3.set_xticklabels(ret_recalls, fontsize=20)[m
[32m+[m[32m#     ax3.set_title('c. GPT-3.5, QA Tasks', fontsize=22)[m
[32m+[m[32m#     for idx, (perf_data, dataset_name) in enumerate(zip(gpt_perf_datas[3:], code_dataset_names)):[m
[32m+[m[32m#         line, = ax4.plot(x, perf_data, marker=markers[idx], markersize=10, linestyle='-', label=auth_code_dataset_names[idx], color=code_colors[idx])[m
[32m+[m[32m#         # ax4.axhline(y=gpt_perf_none[3:][idx], color=line.get_color(), linestyle='--', linewidth=3)   # plot none result[m
[32m+[m[32m#     ax4.set_xlabel('Retrieval Recall', fontsize=22)[m
[32m+[m[32m#     ax4.set_ylabel('Pass@1', fontsize=22)[m
[32m+[m[32m#     ax4.set_yticks([0.2, 0.4, 0.6, 0.8])[m
[32m+[m[32m#     ax4.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize=20)[m
[32m+[m[32m#     ax4.set_xticks(x, ret_recalls)[m
[32m+[m[32m#     ax4.set_xticklabels(ret_recalls, fontsize=20)[m
[32m+[m[32m#     ax4.set_title('d. GPT-3.5, Code Tasks', fontsize=22)[m
[32m+[m[32m#[m
[32m+[m[32m#     ax1_handles, ax1_labels = ax1.get_legend_handles_labels()[m
[32m+[m[32m#     ax2_handles, ax2_labels = ax2.get_legend_handles_labels()[m
[32m+[m[32m#     handles, labels = ax1_handles+ax2_handles, ax1_labels+ax2_labels[m
[32m+[m[32m#     fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=24, bbox_to_anchor=(0.5, -0.12))[m
[32m+[m[32m#     plt.tight_layout()[m
[32m+[m[32m#     plt.savefig('graph/' + graph_name, bbox_inches='tight')[m
[32m+[m[32m#     plt.show()[m
[32m+[m
[32m+[m
 def make_qa_code_retriever_perf():[m
     graph_name = 'qa_code_perf_on_retriever.pdf'[m
     gpt_qa_retriever_perf_datas, llama_qa_retriever_perf_datas = dict(), dict()[m
[36m@@ -1381,7 +1478,7 @@[m [mif __name__ == '__main__':[m
 [m
     # make_qa_code_ret_recall()[m
 [m
[31m-    # make_ret_recall_analysis()[m
[32m+[m[32m    make_ret_recall_analysis()[m
 [m
     # make_ret_recall_perplexity()[m
 [m
[36m@@ -1403,7 +1500,7 @@[m [mif __name__ == '__main__':[m
 [m
     # make_prompt_method_correctness()[m
 [m
[31m-    make_prompt_method_perplexity()[m
[32m+[m[32m    # make_prompt_method_perplexity()[m
 [m
     # make_prompt_method_percentage_of_only_correct()[m
 [m
[1mdiff --git a/expertiments/CalcPPL.py b/expertiments/CalcPPL.py[m
[1mindex e69de29..4a421c0 100644[m
[1m--- a/expertiments/CalcPPL.py[m
[1m+++ b/expertiments/CalcPPL.py[m
[36m@@ -0,0 +1,183 @@[m
[32m+[m[32mimport argparse[m
[32m+[m[32mimport json[m
[32m+[m[32mimport sys[m
[32m+[m[32msys.path.append('../../Code_RAG_Benchmark')[m
[32m+[m[32mimport math[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mfrom scipy import stats[m
[32m+[m[32mimport matplotlib.pyplot as plt[m
[32m+[m
[32m+[m
[32m+[m[32mdef calc_ppl(logprobs_list):[m
[32m+[m[32m    perplexities = [][m
[32m+[m[32m    for logprobs in logprobs_list:[m
[32m+[m[32m        if len(logprobs) == 1: logprobs = logprobs[0][m
[32m+[m[32m        if len(logprobs) == 1: logprobs = logprobs[0][m
[32m+[m[32m        # for idx, data in enumerate(logprobs):[m
[32m+[m[32m        #     if data == 0:[m
[32m+[m[32m        #         logprobs = logprobs[:idx][m
[32m+[m[32m        #         break[m
[32m+[m[32m        # Calculate perplexity for this sequence[m
[32m+[m[32m        # Perplexity = exp(-1/N * sum(log_probs))[m
[32m+[m[32m        avg_log_prob = sum(logprobs) / len(logprobs)[m
[32m+[m[32m        perplexity = math.exp(-avg_log_prob)[m
[32m+[m[32m        perplexities.append(perplexity)[m
[32m+[m
[32m+[m[32m    # Return average perplexity across all sequences[m
[32m+[m[32m    return sum(perplexities) / len(perplexities)[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32mdef analyze_correlation(data1, data2, labels=None):[m
[32m+[m[32m    """[m
[32m+[m[32m    Analyze correlation between two groups of float data[m
[32m+[m
[32m+[m[32m    Args:[m
[32m+[m[32m        data1: List/array of float values (group 1)[m
[32m+[m[32m        data2: List/array of float values (group 2)[m
[32m+[m[32m        labels: Optional tuple of labels for the two groups[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        dict: Dictionary containing all correlation metrics[m
[32m+[m[32m    """[m
[32m+[m[32m    if labels is None:[m
[32m+[m[32m        labels = ("Group 1", "Group 2")[m
[32m+[m
[32m+[m[32m    data1 = np.array(data1)[m
[32m+[m[32m    data2 = np.array(data2)[m
[32m+[m
[32m+[m[32m    if len(data1) != len(data2):[m
[32m+[m[32m        raise ValueError("Both groups must have the same size")[m
[32m+[m
[32m+[m[32m    results = {}[m
[32m+[m
[32m+[m[32m    # 1. Pearson Correlation (most common for linear relationships)[m
[32m+[m[32m    pearson_r, pearson_p = stats.pearsonr(data1, data2)[m
[32m+[m[32m    results['pearson'] = {[m
[32m+[m[32m        'correlation': pearson_r,[m
[32m+[m[32m        'p_value': pearson_p,[m
[32m+[m[32m        'significant': pearson_p < 0.05,[m
[32m+[m[32m        'description': 'Linear correlation'[m
[32m+[m[32m    }[m
[32m+[m
[32m+[m[32m    results = results['pearson'][m
[32m+[m
[32m+[m[32m    # # 2. Spearman Correlation (rank-based, good for monotonic relationships)[m
[32m+[m[32m    # spearman_r, spearman_p = stats.spearmanr(data1, data2)[m
[32m+[m[32m    # results['spearman'] = {[m
[32m+[m[32m    #     'correlation': spearman_r,[m
[32m+[m[32m    #     'p_value': spearman_p,[m
[32m+[m[32m    #     'significant': spearman_p < 0.05,[m
[32m+[m[32m    #     'description': 'Monotonic correlation (rank-based)'[m
[32m+[m[32m    # }[m
[32m+[m[32m    #[m
[32m+[m[32m    # # 3. Kendall's Tau (another rank-based method, more robust)[m
[32m+[m[32m    # kendall_tau, kendall_p = stats.kendalltau(data1, data2)[m
[32m+[m[32m    # results['kendall'] = {[m
[32m+[m[32m    #     'correlation': kendall_tau,[m
[32m+[m[32m    #     'p_value': kendall_p,[m
[32m+[m[32m    #     'significant': kendall_p < 0.05,[m
[32m+[m[32m    #     'description': 'Rank correlation (robust to outliers)'[m
[32m+[m[32m    # }[m
[32m+[m[32m    #[m
[32m+[m[32m    # # 4. Additional statistics[m
[32m+[m[32m    # results['summary'] = {[m
[32m+[m[32m    #     'n': len(data1),[m
[32m+[m[32m    #     'mean_diff': np.mean(data1) - np.mean(data2),[m
[32m+[m[32m    #     'r_squared': pearson_r ** 2,  # Explained variance[m
[32m+[m[32m    #     'rmse': np.sqrt(np.mean((data1 - data2) ** 2))  # Root mean square error[m
[32m+[m[32m    # }[m
[32m+[m
[32m+[m[32m    return results[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == '__main__':[m
[32m+[m[32m    model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",[m
[32m+[m[32m                            "gpt-3.5-turbo-0125": "gpt-3-5-turbo",[m
[32m+[m[32m                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",[m
[32m+[m[32m                            "meta-llama/Llama-2-13b-chat-hf": "llama2-13b"}[m
[32m+[m[41m    [m
[32m+[m[32m    parser = argparse.ArgumentParser()[m
[32m+[m[32m    parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')[m
[32m+[m[32m    parser.add_argument('--model', required=True, help='Model (openai-new, claude)')[m
[32m+[m[32m    parser.add_argument('--mode', default='DocNum', choices=['single', 'oracle', 'recall', 'DocNum', 'prompt'])[m
[32m+[m[32m    # parser.add_argument('--k', type=int, default=1, help='Doc Num, only effective if mode is "DocNum"')[m
[32m+[m[41m    [m
[32m+[m[32m    args = parser.parse_args()[m
[32m+[m[41m    [m
[32m+[m[32m    if args.model == 'openai-new': args.model = 'gpt-4o-mini'[m
[32m+[m[32m    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'[m
[32m+[m[32m    elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'[m
[32m+[m[32m    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'[m
[32m+[m[32m    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'[m
[32m+[m[32m    else: raise Exception('unknown model')[m
[32m+[m[41m    [m
[32m+[m[32m    if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:[m
[32m+[m[32m        ks = [1,3,5,7,10,13,16,20][m
[32m+[m[32m    else:[m
[32m+[m[32m        # ks = [1,3,5,10,15,20,30,40][m
[32m+[m[32m        ks = [1,5,10,15,20][m
[32m+[m[32m        # ks = [3][m
[32m+[m[41m    [m
[32m+[m[32m    model_name_for_path = model_names_for_path[args.model][m
[32m+[m[32m    if args.mode == 'DocNum':[m
[32m+[m[32m        for k in ks:[m
[32m+[m[32m            result_path = f'../data/{args.dataset}/new_results/DocNum/{k}_{model_name_for_path}.json'[m
[32m+[m[32m            logprobs_list = [item['logprobs'] for item in json.load(open(result_path, 'r'))][m
[32m+[m[32m            print(f'for {args.dataset} k={k}, AVG PPL: {round(calc_ppl(logprobs_list), 5)}')[m
[32m+[m
[32m+[m[32m    # doc_num_ppl_data = dict(NQ=[[0.435, 0.400, 0.544, 0.559, 0.552, 0.546, 0.400, 0.400],[m
[32m+[m[32m    #                             [0.427, 0.400, 0.525, 0.543, 0.545, 0.545, 0.400, 0.549],[m
[32m+[m[32m    #                             [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]],[m
[32m+[m[32m    #                         TriviaQA=[[0.732, 0.700, 0.801, 0.825, 0.836, 0.835, 0.700, 0.700],[m
[32m+[m[32m    #                                   [0.740, 0.700, 0.789, 0.818, 0.824, 0.820, 0.700, 0.840],[m
[32m+[m[32m    #                                   [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]],[m
[32m+[m[32m    #                         HotpotQA=[[0.354, 0.300, 0.415, 0.443, 0.436, 0.429, 0.300, 0.300],[m
[32m+[m[32m    #                                   [0.346, 0.300, 0.407, 0.423, 0.428, 0.427, 0.300, 0.438],[m
[32m+[m[32m    #                                   [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],[m
[32m+[m[32m    #                         CoNaLa=[[1.15786, 1.16620, 1.16587, 1.16477, 1.17476, 1.17538, 1.17445, 1.16720],  # llama2[m
[32m+[m[32m    #                                 [1.03702, 1.04643, 1.04296, 1.04402, 1.05064, 1.05248, 1.05434, 1.04689],  # gpt-3.5[m
[32m+[m[32m    #                                 [1.03608, 1.04007, 1.03521, 1.03829, 1.03717, 1.03771, 1.03699, 1.03549]],  # gpt-4o[m
[32m+[m[32m    #                         DS1000=[[1.14383, 1.12741, 1.12806, 1.12710, 1.12337, 1.11702, 1.11472, 1.10787],[m
[32m+[m[32m    #                                 [1.03635, 1.03939, 1.03984, 1.03805, 1.03957, 1.04168, 1.04079, 1.04605],[m
[32m+[m[32m    #                                 [1.03150, 1.03604, 1.03141, 1.03313, 1.03190, 1.03205, 1.02931, 1.03142]],[m
[32m+[m[32m    #                         PNE=[[1.11725, 1.13907, 1.13781, 1.14587, 1.14441, 1.14197, 1.13808, 1.12771],[m
[32m+[m[32m    #                              [1.02660, 1.02309, 1.02266, 1.02137, 1.02051, 1.02087, 1.02030, 1.02166],[m
[32m+[m[32m    #                              [1.01680, 1.02220, 1.02178, 1.02115, 1.02020, 1.01803, 1.01740, 1.01688]],[m
[32m+[m[32m    #                         )[m
[32m+[m
[32m+[m[32m    # docnum_perf_data = dict(NQ=      [[0.435, 0.400, 0.544, 0.559, 0.552, 0.546, 0.400, 0.400],[m
[32m+[m[32m    #                                   [0.427, 0.400, 0.525, 0.543, 0.545, 0.545, 0.400, 0.549],[m
[32m+[m[32m    #                                   [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]],[m
[32m+[m[32m    #                         TriviaQA=[[0.732, 0.700, 0.801, 0.825, 0.836, 0.835, 0.700, 0.700],[m
[32m+[m[32m    #                                   [0.740, 0.700, 0.789, 0.818, 0.824, 0.820, 0.700, 0.840],[m
[32m+[m[32m    #                                   [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]],[m
[32m+[m[32m    #                         HotpotQA=[[0.354, 0.300, 0.415, 0.443, 0.436, 0.429, 0.300, 0.300],[m
[32m+[m[32m    #                                   [0.346, 0.300, 0.407, 0.423, 0.428, 0.427, 0.300, 0.438],[m
[32m+[m[32m    #                                   [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],[m
[32m+[m[32m    #                         CoNaLa=  [[0.238, 0.238, 0.214, 0.238, 0.214, 0.298, 0.250, 0.274],     # llama2[m
[32m+[m[32m    #                                   [0.345, 0.286, 0.333, 0.345, 0.286, 0.345, 0.369, 0.417],     # gpt-3.5[m
[32m+[m[32m    #                                   [0.369, 0.393, 0.417, 0.429, 0.393, 0.393, 0.393, 0.429]],    # gpt-4o[m
[32m+[m[32m    #                         DS1000=  [[0.146, 0.121, 0.135, 0.121, 0.121, 0.140, 0.127, 0.166],[m
[32m+[m[32m    #                                   [0.268, 0.248, 0.261, 0.287, 0.318, 0.318, 0.318, 0.312],[m
[32m+[m[32m    #                                   [0.446, 0.408, 0.439, 0.420, 0.382, 0.376, 0.414, 0.357]],[m
[32m+[m[32m    #                         PNE=     [[0.539, 0.533, 0.539, 0.557, 0.611, 0.611, 0.569, 0.551],[m
[32m+[m[32m    #                                   [0.569, 0.647, 0.659, 0.707, 0.701, 0.731, 0.713, 0.725],[m
[32m+[m[32m    #                                   [0.647, 0.689, 0.695, 0.695, 0.701, 0.701, 0.749, 0.731]],[m
[32m+[m[32m    #                         )[m
[32m+[m
[32m+[m[32m    # for dataset_name in ['CoNaLa', 'DS1000', 'PNE']:[m
[32m+[m[32m    #     for idx, model_name in enumerate(['llama2', 'gpt-3.5', 'gpt-4o']):[m
[32m+[m[32m    #         print(f'******* pearson correlation between PPL and pass@1 under {dataset_name} and {model_name} *******')[m
[32m+[m[32m    #         ppl_data = doc_num_ppl_data[dataset_name][idx][1:][m
[32m+[m[32m    #         # for ppl_idx in range(len(ppl_data)):[m
[32m+[m[32m    #         #     ppl_data[ppl_idx] -= 0.0005 * ppl_idx[m
[32m+[m[32m    #         perf_data = docnum_perf_data[dataset_name][idx][1:][m
[32m+[m
[32m+[m[32m    #         result = analyze_correlation(data1=ppl_data, data2=perf_data)[m
[32m+[m
[32m+[m[32m    #         print(result)[m
[32m+[m
[1mdiff --git a/expertiments/DocNum_script.sh b/expertiments/DocNum_script.sh[m
[1mindex a2e9722..73a8bc5 100644[m
[1m--- a/expertiments/DocNum_script.sh[m
[1m+++ b/expertiments/DocNum_script.sh[m
[36m@@ -1,67 +1,34 @@[m
 # script for LLAMA2 + CODE dataset + Single Oracle analyze[m
 [m
[31m-MODEL='llama-old-code'[m
[32m+[m[32mMODEL='openai-old'[m
 [m
[31m-DATASET='conala'[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 1 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 3 &[m
[31m-wait[m
[32m+[m[32mpython RunOracleSingle.py --dataset NQ --model $MODEL --mode DocNum --k 3 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 5 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 7 &[m
[31m-wait[m
[32m+[m[32mpython RunOracleSingle.py --dataset TriviaQA --model $MODEL --mode DocNum --k 3 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 10 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 13 &[m
[31m-wait[m
[32m+[m[32mpython RunOracleSingle.py --dataset hotpotQA --model $MODEL --mode DocNum --k 3 &[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 16 &[m
[31m-wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 20 &[m
[31m-wait[m
 [m
 [m
 [m
[31m-DATASET='DS1000'[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 1 &[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 3 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 1 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 3 &[m
[31m-wait[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 5 &[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 7 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 5 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 7 &[m
[31m-wait[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 10 &[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 13 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 10 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 13 &[m
[31m-wait[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 16 &[m
[32m+[m[32m# wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 16 &[m
[31m-wait[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 20 &[m
[31m-wait[m
[31m-[m
[31m-[m
[31m-[m
[31m-DATASET='pandas_numpy_eval'[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 1 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 3 &[m
[31m-wait[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 5 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 7 &[m
[31m-wait[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 10 &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 13 &[m
[31m-wait[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 16 &[m
[31m-wait[m
[31m-[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 20 &[m
[31m-wait[m
[32m+[m[32m# python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode DocNum --k 20 &[m
[32m+[m[32m# wait[m
[1mdiff --git a/expertiments/Eval.py b/expertiments/Eval.py[m
[1mindex 3a00e31..a55c289 100644[m
[1m--- a/expertiments/Eval.py[m
[1m+++ b/expertiments/Eval.py[m
[36m@@ -7,7 +7,8 @@[m [mfrom generator.pred_eval import pred_eval_new[m
 if __name__ == '__main__':[m
     model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",[m
                             "gpt-3.5-turbo-0125": "gpt-3-5-turbo",[m
[31m-                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b"}[m
[32m+[m[32m                            "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",[m
[32m+[m[32m                            "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}[m
 [m
     parser = argparse.ArgumentParser()[m
     parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')[m
[1mdiff --git a/expertiments/PredDistributionAnalysis.py b/expertiments/PredDistributionAnalysis.py[m
[1mindex f3623b8..53c451f 100644[m
[1m--- a/expertiments/PredDistributionAnalysis.py[m
[1m+++ b/expertiments/PredDistributionAnalysis.py[m
[36m@@ -2,11 +2,15 @@[m [mimport argparse[m
 import json[m
 import sys[m
 sys.path.append('../../Code_RAG_Benchmark')[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mfrom statsmodels.stats.contingency_tables import mcnemar[m
[32m+[m[32mfrom scipy.stats import chi2_contingency[m
 [m
 [m
 [m
 [m
[31m-def pred_distribution_analysis(pred_list_a, pred_list_b):[m
[32m+[m
[32m+[m[32mdef pred_distribution_analysis(pred_list_a, pred_list_b, alpha=0.05):[m
     """[m
     assert pred_list_a and pred_list_b are a list of prediction, with True/False values[m
     :param pred_list_a:[m
[36m@@ -14,43 +18,166 @@[m [mdef pred_distribution_analysis(pred_list_a, pred_list_b):[m
     :return:[m
     """[m
     assert len(pred_list_a) == len(pred_list_b)[m
[32m+[m[32m    n = len(pred_list_a)[m
     only_a_correct_count = 0[m
     only_b_correct_count = 0[m
     mutual_correct_count = 0[m
[32m+[m[32m    both_wrong_count = 0[m
     for i in range(len(pred_list_a)):[m
         if pred_list_a[i] and pred_list_b[i]:[m
             mutual_correct_count += 1[m
         elif pred_list_a[i] and not pred_list_b[i]:[m
             only_a_correct_count += 1[m
[32m+[m[32m            # print(qid_list[i])[m
         elif not pred_list_a[i] and pred_list_b[i]:[m
             only_b_correct_count += 1[m
[31m-    print("Percentage of only a correctly solve: ", only_a_correct_count/len(pred_list_a))[m
[31m-    print('Percentage of only b correctly solve: ', only_b_correct_count/len(pred_list_b))[m
[31m-    print('Percentage of mutual correctly solve: ', mutual_correct_count/len(pred_list_a))[m
[32m+[m[32m        else:[m
[32m+[m[32m            both_wrong_count += 1[m
[32m+[m[32m    print("Percentage of only a correctly solve: ", round(only_a_correct_count/len(pred_list_a), 3))[m
[32m+[m[32m    print('Percentage of only b correctly solve: ', round(only_b_correct_count/len(pred_list_b), 3))[m
[32m+[m[32m    print('Percentage of mutual correctly solve: ', round(mutual_correct_count/len(pred_list_a), 3))[m
[32m+[m[32m    # print('percentage of only correct in k=1 samples: ', round(only_a_correct_count/len(pred_list_a)*100, 3))[m
[32m+[m
[32m+[m[32m    contingency_table = np.array([[m
[32m+[m[32m        [mutual_correct_count, only_a_correct_count],[m
[32m+[m[32m        [only_b_correct_count, both_wrong_count][m
[32m+[m[32m    ])[m
[32m+[m
[32m+[m[32m    # result = mcnemar(contingency_table, exact=False)[m
[32m+[m[32m    # p_value = result.pvalue[m
[32m+[m
[32m+[m[32m    # table = [[0, only_a_correct_count],[m
[32m+[m[32m    #          [only_b_correct_count, 0]][m
[32m+[m[41m    [m
[32m+[m[32m    # result = mcnemar(table, exact=True)[m
[32m+[m[32m    # p_value = result.pvalue / 2[m
[32m+[m[32m    # if only_b_correct_count <= only_a_correct_count:[m
[32m+[m[32m    #     p_value = 1.0[m
[32m+[m[32m    # if p_value < 0.01:[m
[32m+[m[32m    #     significance = "Highly significant"[m
[32m+[m[32m    # elif p_value < 0.05:[m
[32m+[m[32m    #     significance = "Significant"[m
[32m+[m[32m    # else:[m
[32m+[m[32m    #     significance = "Not significant"[m
[32m+[m[41m    [m
[32m+[m[32m    # print(f"McNemar's test: {significance}")[m
[32m+[m[32m    # print(f"McNemar p-value: {result.pvalue:.6f}")[m
[32m+[m
[32m+[m[32m    # observed_diff = np.mean(pred_list_b) - np.mean(pred_list_a)[m
[32m+[m[41m    [m
[32m+[m[32m    # # Combine all scores[m
[32m+[m[32m    # all_scores = np.concatenate([pred_list_a, pred_list_b])[m
[32m+[m[32m    # n_a = len(pred_list_a)[m
[32m+[m[41m    [m
[32m+[m[32m    # # Generate null distribution[m
[32m+[m
[32m+[m[32m    # n_permutations=1000[m
[32m+[m[32m    # null_diffs = [][m
[32m+[m[32m    # for _ in range(n_permutations):[m
[32m+[m[32m    #     np.random.shuffle(all_scores)[m
[32m+[m[32m    #     perm_a = all_scores[:n_a][m
[32m+[m[32m    #     perm_b = all_scores[n_a:][m
[32m+[m[32m    #     null_diffs.append(np.mean(perm_b) - np.mean(perm_a))[m
[32m+[m[41m    [m
[32m+[m[32m    # # One-tailed p-value[m
[32m+[m[32m    # p_value = (np.sum(null_diffs >= observed_diff) + 1) / (n_permutations + 1)[m
[32m+[m[32m    # if p_value < 0.01:[m
[32m+[m[32m    #     significance = "Highly significant"[m
[32m+[m[32m    # elif p_value < 0.05:[m
[32m+[m[32m    #     significance = "Significant"[m
[32m+[m[32m    # else:[m
[32m+[m[32m    #     significance = "Not significant"[m
[32m+[m
[32m+[m[32m    # print(f"Permutation's test: {significance}")[m
[32m+[m[32m    # print(f"Permutation p-value: {p_value:.6f}")[m
[32m+[m
[32m+[m
[32m+[m[32m    if n > 20 and all(count >= 5 for count in [mutual_correct_count, only_a_correct_count, only_b_correct_count, both_wrong_count]):[m
[32m+[m[32m        chi2, p_value, dof, expected = chi2_contingency(contingency_table)[m
[32m+[m[32m        print(f"Chi-square test: {'Significant' if p_value < alpha else 'Not Significant'}")[m
[32m+[m
[32m+[m[32m        # method_a_results = np.concatenate([[m
[32m+[m[32m        #     np.ones(mutual_correct_count + only_a_correct_count),  # A correct[m
[32m+[m[32m        #     np.zeros(only_b_correct_count + both_wrong_count)      # A incorrect[m
[32m+[m[32m        # ])[m
[32m+[m[41m    [m
[32m+[m[32m        # method_b_results = np.concatenate([[m
[32m+[m[32m        #     np.ones(mutual_correct_count + only_b_correct_count),  # B correct[m
[32m+[m[32m        #     np.zeros(only_a_correct_count + both_wrong_count)      # B incorrect[m
[32m+[m[32m        # ])[m
[32m+[m[32m        # correlation, p_value_pearson = chi2_contingency(method_a_results, method_b_results)[m
[32m+[m[32m        # print(f"Pearson correlation: r = {correlation:.4f}")[m
[32m+[m[32m        # print(f"Pearson test: {'Significant' if p_value_pearson < alpha else 'Not Significant'} (p = {p_value_pearson:.6f})")[m
[32m+[m[32m    else:[m
[32m+[m[32m        print("Chi-square test: Sample size too small or expected frequencies < 5")[m
 [m
 [m
 if __name__ == '__main__':[m
     parser = argparse.ArgumentParser()[m
     parser.add_argument('--dataset', required=True, help='Dataset (conala, DS1000)')[m
     parser.add_argument('--model', required=True, help='Model (openai-new, claude)')[m
[31m-    parser.add_argument('--mode', required=True, choices=['single-oracle'])[m
[31m-    parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')[m
[32m+[m[32m    parser.add_argument('--mode', required=True, choices=['single-oracle', 'prompt-methods', 'DocNum', 'DocError'])[m
[32m+[m[32m    # parser.add_argument('--recall', type=float, default=1, help='Recall, only effective if mode is "recall"')[m
 [m
     args = parser.parse_args()[m
 [m
     if args.model == 'openai-new': args.model = 'gpt-4o-mini'[m
[31m-    elif args.model == 'openai-old': args.model = 'gpt-3.5-turbo-0125'[m
[32m+[m[32m    elif args.model == 'openai-old': args.model = 'gpt-3-5-turbo'[m
     elif args.model == 'llama-new': args.model = 'meta-llama/Llama-3.1-8B-Instruct'[m
[31m-    elif args.model == 'llama-old-code': args.model = 'codellama/CodeLlama-13b-Instruct-hf'[m
[31m-    elif args.model == 'llama-old-qa': args.model = 'meta-llama/Llama-2-13b-chat-hf'[m
[32m+[m[32m    elif args.model == 'llama-old-code': args.model = 'codellama-13b'[m
[32m+[m[32m    elif args.model == 'llama-old-qa': args.model = 'llama2-13b'[m
     else: raise Exception('unknown model')[m
 [m
     if args.mode == 'single-oracle':[m
[31m-        single_result_path = f'../data/{args.dataset}/new_results/single_{args.model}_evals.json'[m
[31m-        oracle_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_evals.json'[m
[32m+[m[32m        single_result_path = f'../data/{args.dataset}/new_results/single_{args.model}_eval.json'[m
[32m+[m[32m        oracle_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_eval.json'[m
         single_results = json.load(open(single_result_path, 'r'))["eval_records"][m
         oracle_results = json.load(open(oracle_result_path, 'r'))["eval_records"][m
         single_pred_list = [single_results[pid]['passed'] for pid in single_results][m
         oracle_pred_list = [oracle_results[pid]['passed'] for pid in oracle_results][m
[32m+[m[32m        qid_list = list(single_results.keys())[m
         print('Single LLM v.s. Oracle RAG Prediction Distribution Difference:')[m
[31m-        pred_distribution_analysis(single_pred_list, oracle_pred_list)[m
[32m+[m[32m        pred_distribution_analysis(single_pred_list, oracle_pred_list, qid_list)[m
[32m+[m
[32m+[m[32m    elif args.mode == 'DocNum':[m
[32m+[m[32m        ks = [1,3,5,10,15,20,30,40][m
[32m+[m[32m        # ks = [1,5,10,15,20][m
[32m+[m[32m        a_ks = ks[:-1][m
[32m+[m[32m        b_ks = ks[1:][m
[32m+[m[32m        for a_k, b_k in zip(a_ks, b_ks):[m
[32m+[m[32m            a_result_path = f'../data/{args.dataset}/new_results/DocNum/{a_k}_{args.model}_eval.json'[m
[32m+[m[32m            b_result_path = f'../data/{args.dataset}/new_results/DocNum/{b_k}_{args.model}_eval.json'[m
[32m+[m[32m            a_results = json.load(open(a_result_path, 'r'))["eval_records"][m
[32m+[m[32m            b_results = json.load(open(b_result_path, 'r'))["eval_records"][m
[32m+[m[32m            a_pred_list = [a_results[pid]['has_answer'] for pid in a_results][m
[32m+[m[32m            b_pred_list = [b_results[pid]['has_answer'] for pid in b_results][m
[32m+[m[32m            print(f'\n\n{a_k} v.s. {b_k} RAG Prediction Distribution Difference:')[m
[32m+[m[32m            pred_distribution_analysis(a_pred_list, b_pred_list)[m
[32m+[m[41m    [m
[32m+[m[32m    elif args.mode == 'DocError':[m
[32m+[m[32m        ks = [3, 5, 7, 10, 13, 16, 20][m
[32m+[m[32m        base_result_path = f'../data/{args.dataset}/new_results/DocNum/1_{args.model}_eval.json'[m
[32m+[m[32m        # base_result_path = f'../data/{args.dataset}/new_results/oracle_{args.model}_eval.json'[m
[32m+[m[32m        base_results = json.load(open(base_result_path, 'r'))["eval_records"][m
[32m+[m[32m        base_pred_list = [base_results[pid]['passed'] for pid in base_results][m
[32m+[m[32m        for k in ks:[m
[32m+[m[32m            k_result_path = f'../data/{args.dataset}/new_results/DocNum/{k}_{args.model}_eval.json'[m
[32m+[m[32m            k_results = json.load(open(k_result_path, 'r'))["eval_records"][m
[32m+[m[32m            k_pred_list = [k_results[pid]['passed'] for pid in k_results][m
[32m+[m[32m            print(f'\n\n1 v.s. {k} RAG Prediction Distribution Difference:')[m
[32m+[m[32m            pred_distribution_analysis(base_pred_list, k_pred_list)[m
[32m+[m
[32m+[m[32m    elif args.mode == 'prompt-methods':[m
[32m+[m[32m        if args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:[m
[32m+[m[32m            baseline_result_path = f'../data/{args.dataset}/new_results/DocNum/5_{args.model}_eval.json'[m
[32m+[m[32m        else:[m
[32m+[m[32m            baseline_result_path = f'../data/{args.dataset}/new_results/DocNum/10_{args.model}_eval.json'[m
[32m+[m[32m        baseline_results = json.load(open(baseline_result_path, 'r'))["eval_records"][m
[32m+[m[32m        baseline_pred_list = [baseline_results[pid]['has_answer'] for pid in baseline_results][m
[32m+[m[32m        prompt_methods = ['few-shot', 'emotion', 'CoT', 'zero-shot-CoT', 'Least-to-Most', 'Plan-and-Solve', 'self-refine', 'CoN'][m
[32m+[m[32m        for method in prompt_methods:[m
[32m+[m[32m            method_result_path = f'../data/{args.dataset}/new_results/Prompt/{method}_{args.model}_eval.json'[m
[32m+[m[32m            method_results = json.load(open(method_result_path, 'r'))["eval_records"][m
[32m+[m[32m            method_pred_list = [method_results[pid]['has_answer'] for pid in method_results][m
[32m+[m[32m            print(f'zero-shot baseline prompt v.s. {method} Prediction Distribution Difference:')[m
[32m+[m[32m            pred_distribution_analysis(baseline_pred_list, method_pred_list)[m
[1mdiff --git a/expertiments/Prompt_script.sh b/expertiments/Prompt_script.sh[m
[1mindex 671de61..8aec1ff 100644[m
[1m--- a/expertiments/Prompt_script.sh[m
[1m+++ b/expertiments/Prompt_script.sh[m
[36m@@ -2,22 +2,28 @@[m
 [m
 MODEL='llama-old-code'[m
 [m
[31m-DATASET='DS1000'[m
[32m+[m[32mDATASET='pandas_numpy_eval'[m
 [m
 python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt few-shot &[m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &[m
 wait[m
 [m
 python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoT &[m
[32m+[m[32mwait[m
[32m+[m
[32m+[m[32mpython RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt emotion &[m
 python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt zero-shot-CoT &[m
 wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &[m
 python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Plan-and-Solve &[m
 wait[m
 [m
[31m-python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &[m
 python RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt CoN &[m
 wait[m
 [m
[32m+[m[32mpython RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt Least-to-Most &[m
[32m+[m[32mwait[m
[32m+[m
[32m+[m[32mpython RunOracleSingle.py --dataset $DATASET --model $MODEL --mode prompt --prompt self-refine &[m
[32m+[m[32mwait[m
[32m+[m
 [m
[1mdiff --git a/expertiments/RunOracleSingle.py b/expertiments/RunOracleSingle.py[m
[1mindex cd7258d..d548741 100644[m
[1m--- a/expertiments/RunOracleSingle.py[m
[1m+++ b/expertiments/RunOracleSingle.py[m
[36m@@ -16,7 +16,7 @@[m [mfrom llms.LLMConfig import LLMConfig, LLMSettings[m
 from llms.OpenAIProvider import OpenAIProvider[m
 from llms.LLAMAProvider import LlamaProvider[m
 from generator.pred_eval import pred_eval_new[m
[31m-from generator.generate_utils import truncate_docs[m
[32m+[m[32mfrom generator.generate_utils import truncate_docs, get_docs_tokens[m
 [m
 [m
 class LLMOracleEvaluator:[m
[36m@@ -27,8 +27,8 @@[m [mclass LLMOracleEvaluator:[m
         elif model == 'llama-old-code': self.model_config = LLMSettings().LLAMAConfigs().llama_old_code[m
         elif model == 'llama-old-qa': self.model_config = LLMSettings().LLAMAConfigs().llama_old_qa[m
         else: raise Exception('Unknown model')[m
[31m-        if dataset in ['NQ', 'TriviaQA', 'HotpotQA']: self.max_tokens = 100     # todo: do not consider prompting method![m
[31m-        elif dataset == 'DS1000': self.max_tokens = 1000[m
[32m+[m[32m        if dataset in ['NQ', 'TriviaQA', 'HotpotQA']: self.max_tokens = 500     # todo: do not consider prompting method![m
[32m+[m[32m        elif dataset == 'DS1000' or dataset == 'pandas_numpy_eval': self.max_tokens = 1000[m
         else: self.max_tokens = 500[m
         if self.model_config.organization == 'openai':[m
             self.llm_provider = OpenAIProvider(organization=self.model_config.organization,[m
[36m@@ -36,14 +36,14 @@[m [mclass LLMOracleEvaluator:[m
                                                temperature=self.model_config.temperature,[m
                                                max_tokens=self.max_tokens,[m
                                                is_async=self.model_config.is_async,[m
[31m-                                               stop=None)[m
[32m+[m[32m                                               stop=['</answer>'])[m
         elif self.model_config.organization == 'llama':[m
             self.llm_provider = LlamaProvider(organization=self.model_config.organization,[m
                                               model=self.model_config.model,[m
                                               temperature=self.model_config.temperature,[m
                                               max_tokens=self.max_tokens,[m
                                               is_async=self.model_config.is_async,[m
[31m-                                              stop=['package com'])[m
[32m+[m[32m                                              stop=['package com', '</answer>'])[m
 [m
         self.dataset = dataset[m
 [m
[36m@@ -53,7 +53,8 @@[m [mclass LLMOracleEvaluator:[m
 [m
         self.model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",[m
                                      "gpt-3.5-turbo-0125": "gpt-3-5-turbo",[m
[31m-                                     "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b"}[m
[32m+[m[32m                                     "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",[m
[32m+[m[32m                                     "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}[m
 [m
         if self.dataset == 'conala':[m
             self.problems = ConalaLoader().load_qs_list()[m
[36m@@ -303,7 +304,10 @@[m [mclass LLMOracleEvaluator:[m
         if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:[m
             self.k_range = [1, 3, 5, 7, 10, 13, 16, 20][m
         else:[m
[31m-            self.k_range = [1, 3, 5, 10, 15, 20, 30, 40][m
[32m+[m[32m            if self.model_config == LLMSettings().LLAMAConfigs().llama_old_qa:[m
[32m+[m[32m                self.k_range = [1, 3, 5, 10, 15, 20][m
[32m+[m[32m            else:[m
[32m+[m[32m                self.k_range = [1, 3, 5, 10, 15, 20, 30, 40][m
         assert k in self.k_range[m
 [m
         # Prepare messages[m
[36m@@ -321,12 +325,21 @@[m [mclass LLMOracleEvaluator:[m
                     break[m
             if not ret_docs_exist: raise Exception(f'no ret docs for problem: {qs_id}')[m
             # use top-k docs as retrieved docs[m
[31m-            truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)[m
[32m+[m[32m            if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:[m
[32m+[m[32m                truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)[m
[32m+[m[32m            else:[m
[32m+[m[32m                truncated_docs = [item['doc'] for item in ret_docs[qs_id][:k]][m
             prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs)[m
             prompts.append(prompt)[m
             problem_ids.append(problem['qs_id'])[m
 [m
[32m+[m
[32m+[m
         if test_prompt:[m
[32m+[m[32m            # if self.model_config == LLMSettings().LLAMAConfigs().llama_old_qa:[m
[32m+[m[32m            #     prompt_lengths = get_docs_tokens(prompts, model='llama2-13b')[m
[32m+[m[32m            #     for length in prompt_lengths:[m
[32m+[m[32m            #         if length > 4096: print(length)[m
             if 'gpt' in self.model_config.model:[m
                 print(prompts[0][0]['content'])[m
                 print(prompts[0][1]['content'])[m
[36m@@ -439,7 +452,10 @@[m [mclass LLMOracleEvaluator:[m
                     break[m
             if not ret_docs_exist: raise Exception(f'no ret docs for problem: {qs_id}')[m
             # use top-k docs as retrieved docs[m
[31m-            truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)[m
[32m+[m[32m            if self.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:[m
[32m+[m[32m                truncated_docs = truncate_docs(ret_docs[qs_id][:k], model='gpt-3.5-turbo-0125', max_length=500)[m
[32m+[m[32m            else:[m
[32m+[m[32m                truncated_docs = [item['doc'] for item in ret_docs[qs_id][:k]][m
             if prompt_method == 'self-refine':[m
                 assert problem['qs_id'] == initial_results[idx]['qs_id'][m
                 prompt = self.prompt_generator(question=problem['question'], model=self.model_config.model, ret_docs=truncated_docs, initial_output=initial_results[idx]['outputs'][0])[m
[36m@@ -495,7 +511,7 @@[m [mclass LLMOracleEvaluator:[m
                 'response': response.get('text', ''),[m
                 'logprobs': response.get('logprobs', []),[m
             })[m
[31m-[m
[32m+[m[41m            [m
         print(f"✅ Generated {len(results)} Recall Analysis LLM responses")[m
 [m
         os.makedirs(result_path.rsplit('/', 1)[0], exist_ok=True)[m
[1mdiff --git a/expertiments/fuck.sh b/expertiments/fuck.sh[m
[1mindex e69de29..8d21175 100644[m
[1m--- a/expertiments/fuck.sh[m
[1m+++ b/expertiments/fuck.sh[m
[36m@@ -0,0 +1,62 @@[m
[32m+[m[32m# DATASET='TriviaQA'[m
[32m+[m[32m# MODEL='llama-old-qa'[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_1.json ../data/$DATASET/new_results/DocNum/1_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_5.json ../data/$DATASET/new_results/DocNum/5_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_10.json ../data/$DATASET/new_results/DocNum/10_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_15.json ../data/$DATASET/new_results/DocNum/15_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_20.json ../data/$DATASET/new_results/DocNum/20_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_30.json ../data/$DATASET/new_results/DocNum/30_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_gpt-3.5-turbo-0125_n_1_retrieval_doc_selection_top_40.json ../data/$DATASET/new_results/DocNum/40_gpt-3-5-turbo.json[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 1[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 5[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 10[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 15[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 20[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 30[m
[32m+[m
[32m+[m[32m# python Eval.py --dataset $DATASET --model $MODEL --mode DocNum --k 40[m
[32m+[m
[32m+[m
[32m+[m[32mMODEL='llama-old-qa'[m
[32m+[m
[32m+[m[32m# python RunOracleSingle.py --dataset NQ --model $MODEL --mode prompt --prompt emotion &[m
[32m+[m[32m# python RunOracleSingle.py --dataset TriviaQA --model $MODEL --mode prompt --prompt emotion &[m
[32m+[m
[32m+[m[32mpython RunOracleSingle.py --dataset hotpotQA --model llama-old-qa --mode prompt --prompt emotion &[m
[32m+[m[32mpython RunOracleSingle.py --dataset NQ --model llama-old-qa --mode prompt --prompt zero-shot-CoT &[m
[32m+[m[32mwait[m
[32m+[m
[32m+[m[32mpython RunOracleSingle.py --dataset TriviaQA --model llama-old-qa --mode prompt --prompt zero-shot-CoT &[m
[32m+[m[32mpython RunOracleSingle.py --dataset hotpotQA --model llama-old-qa --mode prompt --prompt zero-shot-CoT &[m
[32m+[m[32mwait[m
[32m+[m
[32m+[m[32m# OLD_MODEL='gpt-3.5-turbo-0125'[m
[32m+[m[32m# NEW_MODEL='gpt-3-5-turbo'[m
[32m+[m[32m# DATASET='NQ'[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_3shot.json ../data/$DATASET/new_results/Prompt/few-shot_${NEW_MODEL}.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_cot.json ../data/$DATASET/new_results/Prompt/CoT_${NEW_MODEL}.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_least_to_most.json ../data/$DATASET/new_results/Prompt/Least-to-Most_${NEW_MODEL}.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_plan_and_solve.json ../data/$DATASET/new_results/Prompt/Plan-and-Solve_${NEW_MODEL}.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_self-refine.json ../data/$DATASET/new_results/Prompt/self-refine_${NEW_MODEL}.json[m
[32m+[m
[32m+[m[32m# cp ../old_data/$DATASET/results/model_${OLD_MODEL}_n_1_prompt_method_con.json ../data/$DATASET/new_results/Prompt/CoN_${NEW_MODEL}.json[m
\ No newline at end of file[m
[1mdiff --git a/expertiments/test.py b/expertiments/test.py[m
[1mindex e69de29..163de5e 100644[m
[1m--- a/expertiments/test.py[m
[1m+++ b/expertiments/test.py[m
[36m@@ -0,0 +1,50 @@[m
[32m+[m[32mimport tempfile[m
[32m+[m[32mimport os[m
[32m+[m[32mimport subprocess[m
[32m+[m
[32m+[m
[32m+[m[32mdef check_python_code_ruff(code_string):[m
[32m+[m[32m    """[m
[32m+[m[32m    Check Python code string using Ruff for errors only (ignoring warnings)[m
[32m+[m
[32m+[m[32m    Args:[m
[32m+[m[32m        code_string (str): Python code to check[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        bool: True if no errors found, False if errors detected[m
[32m+[m[32m    """[m
[32m+[m[32m    try:[m
[32m+[m[32m        # Create temporary file[m
[32m+[m[32m        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:[m
[32m+[m[32m            temp_file.write(code_string)[m
[32m+[m[32m            temp_filename = temp_file.name[m
[32m+[m
[32m+[m[32m        try:[m
[32m+[m[32m            # Run Ruff with error-only selection[m
[32m+[m[32m            result = subprocess.run([[m
[32m+[m[32m                'ruff', 'check',[m
[32m+[m[32m                '--select=E9,F63,F7,F82',  # Error codes only[m
[32m+[m[32m                temp_filename[m
[32m+[m[32m            ], capture_output=True, text=True, timeout=10)[m
[32m+[m
[32m+[m[32m            # Return True if no errors (exit code 0), False if errors found[m
[32m+[m[32m            # print(result)[m
[32m+[m[32m            return result.returncode == 0[m
[32m+[m
[32m+[m[32m        finally:[m
[32m+[m[32m            # Clean up temporary file[m
[32m+[m[32m            if os.path.exists(temp_filename):[m
[32m+[m[32m                os.unlink(temp_filename)[m
[32m+[m
[32m+[m[32m    except subprocess.TimeoutExpired:[m
[32m+[m[32m        return False[m
[32m+[m[32m    except FileNotFoundError:[m
[32m+[m[32m        return False[m
[32m+[m[32m    except Exception:[m
[32m+[m[32m        return False[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == '__main__':[m
[32m+[m[32m    code = "code\nimport pandas as pd\nimport numpy as np\n\ndef replacing_blank_with_nan(df):\n    # replace field that's entirely space (or empty) with NaN using regex\n    df.replace(r'^\\s*$', np.nan, regex=True, inplace=True)\n    return df\n"[m
[32m+[m[32m    result = check_python_code_ruff(code)[m
[32m+[m[32m    print(result)[m
[1mdiff --git a/generator/pred_eval.py b/generator/pred_eval.py[m
[1mindex 1ffd2e8..b3be4ee 100644[m
[1m--- a/generator/pred_eval.py[m
[1m+++ b/generator/pred_eval.py[m
[36m@@ -1,3 +1,4 @@[m
[32m+[m[32mimport ast[m
 import os.path[m
 import platform[m
 import sys[m
[36m@@ -5,6 +6,8 @@[m [mimport json[m
 import re[m
 import numpy as np[m
 from typing import List[m
[32m+[m[32mimport tempfile[m
[32m+[m[32mimport subprocess[m
 system = platform.system()[m
 if system == 'Darwin':[m
     root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'[m
[36m@@ -161,7 +164,7 @@[m [mdef pandas_numpy_eval_result_process(prompt_type, output, code_prompt, output_be[m
     return pred[m
 [m
 [m
[31m-def process_gene_results(dataset, outputs, prompt_type=None, code_prompt=None, outputs_before=None):[m
[32m+[m[32mdef process_gene_results(qs_id, dataset, outputs, prompt_type=None, code_prompt=None, outputs_before=None):[m
     preds = [][m
     if dataset == 'conala':[m
         for idx, output in enumerate(outputs):[m
[36m@@ -190,11 +193,11 @@[m [mdef process_gene_results(dataset, outputs, prompt_type=None, code_prompt=None, o[m
     elif dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA':[m
         for idx, output in enumerate(outputs):[m
             pred = output[m
[31m-            if prompt_type == 'RaR':[m
[31m-                try: pred = pred.split('Answer:\n')[1][m
[31m-                except: ...[m
[31m-                try: pred = pred.split('the answer')[1][m
[31m-                except: ...[m
[32m+[m[32m            # if prompt_type == 'RaR':[m
[32m+[m[32m            #     try: pred = pred.split('Answer:\n')[1][m
[32m+[m[32m            #     except: ...[m
[32m+[m[32m            #     try: pred = pred.split('the answer')[1][m
[32m+[m[32m            #     except: ...[m
             if prompt_type == 'self-refine':[m
                 if not '<answer>' in output and not '```' in output:[m
                     pred = outputs_before[idx][m
[36m@@ -305,98 +308,429 @@[m [mdef process_ds1000_outputs(pid: str, outputs: List[str], existing_code: str):[m
     return processed_outputs[m
 [m
 [m
[32m+[m
[32m+[m
[32m+[m[32mdef check_python_code_ruff(code_string):[m
[32m+[m[32m    """[m
[32m+[m[32m    Check Python code string using Ruff for errors only (ignoring warnings)[m
[32m+[m
[32m+[m[32m    Args:[m
[32m+[m[32m        code_string (str): Python code to check[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        bool: True if no errors found, False if errors detected[m
[32m+[m[32m    """[m
[32m+[m[32m    common_imports = {[m
[32m+[m[32m        'pd', 'pandas',  # pandas[m
[32m+[m[32m        'np', 'numpy',  # numpy[m
[32m+[m[32m        'tf', 'tensorflow',  # tensorflow[m
[32m+[m[32m        'torch', 'pytorch',  # pytorch[m
[32m+[m[32m        'sk', 'sklearn',  # scikit-learn[m
[32m+[m[32m        'sns', 'seaborn',  # seaborn[m
[32m+[m[32m        'json', 'pickle',  # serialization[m
[32m+[m[32m        'os', 'sys', 'pathlib',  # system[m
[32m+[m[32m        'time', 'datetime',  # time[m
[32m+[m[32m        're', 'regex',  # regex[m
[32m+[m[32m        'math', 'statistics',  # math[m
[32m+[m[32m        'collections', 'itertools',  # collections[m
[32m+[m[32m        'typing', 'dataclasses',  # typing[m
[32m+[m[32m        # todo: weird function name for conala[m
[32m+[m[32m        'array_equal', 'indexOf', 'parse_datetime_string', 'set_warn_always', 'find_duplicate', 'file_exists', '_record_count', '_remove_dups', 'get', 'write_bytes',[m
[32m+[m[32m        'makePickle', 'formatweekday', 'intersection', 'split', 'timedelta', 'today', 'ensure_decoded', 'encode_base64', 'to_list', 'atoi', '_record_count',[m
[32m+[m[32m        '_follow_symlinks', 'split_outside_bracket', '_sort_dump_data_by', 'itervalues', 'setvar', 'find', 'unhex', '_delimited_splitter', 'get_attrs', 'mean', 'utcnow', 'concatenate',[m
[32m+[m[32m        'unhexlify', 'b64encode', 'dedent', '_munge_whitespace', 'timezone', 'unquote_plus', 'fromisoformat', 'builtins', 'DataFrame', 'Sequence', 'extract_bool_array', 'set_array',[m
[32m+[m[32m        'eliminate_zeros', '_find_valid_index', '_remove_vertex', 'set_charmap', 'itn', 'previous_friday', 'unquote', 'before_nearest_workday', 'plain', 'countOf', 'drive_files',[m
[32m+[m[32m        'unescape', 'encode', '_parse_date', 'datestr', 'date_format', 'strftime', '_convert_strls', 'size', 'file_exists', 'contains', 'temp_setattr', 'set_charmap', 'remove_axis',[m
[32m+[m[32m        'set_char', 'ishex', 'splitattr', 'get_intersection', 'BooleanArray', 'unique_key', 'test', 'reverse_dict', '_normalise_json_ordered', '_list_of_dict_to_arrays', '_sorted',[m
[32m+[m[32m        'valfilter', '_matrix_vector_product_of_stacks', 'get_lastbday', '_prev_opening_time', 'before_nearest_workday', 'Int2AP', 'add_object_type_line', '_log_normalize',[m
[32m+[m[32m        '_prepare_categoricals', '_replace_nans', '_convert_strls', '_shape_common', 'samestat', '_datetime_to_stata_elapsed_vec', 'previous_workday', 'duplicated', '_filter',[m
[32m+[m[32m        'fast_unique_multiple', 'strip_newsgroup_footer', 'strip_newsgroup_quoting', '_checkpoint_exists', '_flatten_dims_0_and_1', '_lookup_reduction', 'replace_list', 'filter_sources',[m
[32m+[m[32m        'newer_pairwise', 'targets', 'multicolumn', '_get_json_content_from_openml_api', '_find_lteq', 'random', 'a', 'b', 'abracadabra', 'newFile', 'newFileBytes', '_create_block_3_diagonal_matrix',[m
[32m+[m[32m        'd', '_multi_dot_three', 'subspace_angles', '_matrix_vector_product_of_stacks', '_remove_zero_rows', 'indices_to_mask', 'asof_locs', '_all_string_prefixes', 'validate_strlist',[m
[32m+[m[32m        'isalnum', 'nearest_workday', 'after_nearest_workday', '_format_multicolumn', '_Flatten', 'df', 's', '_abc_registry_clear', 'clean', 'prune', 'difference', 'l1', 'l2', 'stopwords',[m
[32m+[m[32m        'remove_vertex', 'unescaped', 'previous_workday', 'StataStrLWriter', 'matmul', '_remove_zero_rows', 'remove', 'match_extensions', 'what', '_guess_quote_and_delimiter', 'splitattr', 'as_hex',[m
[32m+[m[32m        'resetwarnings', 'warning', 'exception', '_missing_warn', 'warn_explicit', 'set_warn_always', 'resetwarnings', 'match', 'symmetric_difference', 'tolil', '_format_multirow', '_Flatten', 'floor',[m
[32m+[m[32m        'load_data', 'sklearn_model', 'data', 'float16', 'result', 'x', 'y', 'index', 'columns', 'sparse_csr_matrix_ops', 'solve_ivp', 'sparse', 'X', 'train_test_split', 'data_matrix', 'X_train', 'y_train',[m
[32m+[m[32m        'GradientBoostingClassifier', 'my_map_func', 'NormalDistro', 'train_size', 'features_dataframe', '_reshape_2D', '_to_tensor', 'tensor', 'standard_scale', 'make_np', 'predict',[m
[32m+[m[32m        'sparse_matrix_sparse_mat_mul', 'RandomForestRegressor', 'X_test', 'dataset', 'scaled', 'fitted_model', 'W', 't', 'exp', 'df_a', 'df_b', '_unpack', 'sciopt', 'e', 'pmin', 'pmax',[m
[32m+[m[32m        'scipy', 'points', 'extraPoints', 'model', 'LinearSVC', 'vectorizer', 'preprocessing', 'SelectFromModel', 'clf', 'TfidfVectorizer', 'cv', 'logreg', 'km', 'hid_dim', 'softmax_output',[m
[32m+[m[32m        'rotate_around', 'make_friedman1', 'scores', 'predicted_t_scaled', '_export_model_variables', 'make_friedman2', 'datasets', 'GradientBoostingClassifier', 'clean_data', 'time_span', 'N0', 'integrate',[m
[32m+[m[32m        'transform_output', 'df_origin', 'new_data', 'C', 'D', 'rotate_deg_around', 'load_target', 'features_dataframe', 'regression_model', 'strs', 'example_df', '_find_numeric_cols', 'DataArray',[m
[32m+[m[32m        'percentile', 'self', 'r', 'c', 'im', 'rankdata', 'N', 'orthogonal_procrustes', '_sparse_manhattan', '_quadratic_assignment_2opt', 'assertNDArrayNear', 'contiguous_regions',[m
[32m+[m[32m        'assertNDArrayNear', 'is_evenly_distributed_thresholds', 'f_classif', 'query', 'apply_2d', '_feature_to_dtype', '_python_apply_general', 'split_training_and_validation_data', '_do_convert_categoricals',[m
[32m+[m[32m        '_sanitize_ndim', '_convert_to_tensors_or_sparse_tensors', 'stacked_matmul', '_cast_tensor_to_floatx', 'convert', 'make_np', '_cast_tensor_to_floatx', 'sanitize_masked_array', 'data_rot', 'suppmach',[m
[32m+[m[32m        'x_test', 'suppmach', 'X_train_num', 'car', 'values', 'names', 'times', 'regressor', 'insert', 'column_names', 'cosine_similarity', 'get_term_frequency_inverse_data_frequency', 'example_df',[m
[32m+[m[32m        'example_dict', 'corr', 'thresh', 'post', 'distance', 'fill_zeros', 'factorize', 'nonzero', 'A', 'img', 'threshold', 'dN1_dt_simple', 'solve_ivp', 'tfidf', 'queries', 'objective', 'pad_width',[m
[32m+[m[32m        'points1', 'points2', 'features', 'diagonal', 'condensed_dist_matrix', 'joblib', 'something', 'array', 'nlargest', 'einsum', 'search', 'where', 'ensure_string_array',[m
[32m+[m[32m        'ensure_float', 'remove_na_arraylike', 'nanargmin', 'vec', 'can_hold_element', 'z_score', 'isnaobj', 'maybe_fill', 'nanops', 'Normalize', '_try_cast',[m
[32m+[m[32m        'is_inferred_bool_dtype', 'condition', 'axis_slice', 'get_geometry', 'max_len', 'z', 'target_series', 'source_series', '_sort_tuples', 'arr', 'make_sparse',[m
[32m+[m[32m        'downcast_intp_index', '_from_backing_data', 'can_hold_element', '_from_backing_data', 'find_repeats', 'is_empty_indexer', 'is_sparse', 'is_scipy_sparse',[m
[32m+[m[32m        'remove_na_arraylike', 'maybe_cast_to_integer_array', 'Index', 'IndexLabel', 'astype_nansafe', 'maybe_fill', 'unpack_zerodim_and_defer', '_get_dataframe_dtype_counts',[m
[32m+[m[32m        '_set_noconvert_dtype_columns', '_try_convert_data', 'rfn', '_to_matrix_vectorized', 'rgb_to_hsv', 'NpDtype', 'f', 'emails', '_get_na_values', 'col', '_validate_names', '_check_column_names',[m
[32m+[m[32m        'loc', 'column', 'value', 'col_names', 'Samples', 's1', 's2', 'col_name', 'df1', 'df2'