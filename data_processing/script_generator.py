files = ['run.py', 'generate.py', 'transfer']
file = files[2]

actions = ['gene_prompts', 'gene_responses', 'eval_pred']
action = actions[2]

models = ['gpt-3.5-turbo-0125', 'codellama-13b-instruct', 'llama2-13b-chat']
model = models[1]

datasets = ['NQ', 'TriviaQA', 'hotpotQA', 'conala', 'DS1000', 'pandas_numpy_eval']
qa_datasets = datasets[3:]
code_datasets = datasets[:3]
# if model == models[2]: datasets = qa_datasets
# elif model == models[1]: datasets = code_datasets

retrievers = ['BM25', 'miniLM', 'openai-embedding', 'codeT5']
retriever = retrievers[2]

analysis_types = ['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_selection', 'prompt_method']
analysis_type = analysis_types[3]
# analysis_type_paras = ['ret_acc', 'ret_doc_type', 'doc_selection_type']
# analysis_type_para = analysis_type_paras[1]

ns = [1, 10]
n = ns[0]

batches = [True, False]
batch = batches[1]


if file == 'generate.py':
    # script = ''
    # for retriever in retrievers:
    #     script_temp = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n {n} --dataset {dataset} "
    #                 f"--retriever {retriever} --analysis_type {analysis_type} --ret_doc_type none")
    #     print(script_temp)
    #     if batch: script += script_temp + ' --batch  &    '
    #     else: script += script_temp + ';
    for dataset in datasets:
        script = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n {n} --dataset {dataset} "
                  f"--retriever {retriever} --analysis_type {analysis_type} --doc_selection_type top_5")
        print(script)
elif file == 'run.py':
    script = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n 1 --dataset {dataset} "
              f"--retriever {retriever} --analysis_type {analysis_type}")
elif file == 'transfer':
    for dataset in datasets:
        if 'llama' in model:
            if dataset in ['NQ', 'TriviaQA', 'hotpotQA']: model = 'llama2-13b-chat'
            else: model = 'codellama-13b-instruct'
        script = (f"scp -P 10389 zhaoshengming@129.128.209.149:~/Code_RAG_Benchmark/data/{dataset}/results/model_{model}_n_1_{analysis_type}_least_to_most.json "
                  f"data/{dataset}/results")
        print(script)
        # script = (f"scp -P 10389 zhaoshengming@129.128.209.149:~/Code_RAG_Benchmark/data/{dataset}/results/model_{model}_n_1_{analysis_type}_3shot.json "
        #           f"data/{dataset}/results")
        # print(script)

