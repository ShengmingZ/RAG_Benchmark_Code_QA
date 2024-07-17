files = ['run.py', 'generate.py']
file = files[0]

actions = ['gene_prompts', 'gene_responses', 'eval_pred']
action = actions[0]

models = ['gpt-3.5-turbo-0125', 'codellama-13b-instruct', 'llama2-13b-chat']
model = models[2]

datasets = ['conala', 'DS1000', 'pandas_numpy_eval', 'NQ', 'TriviaQA', 'hotpotQA']
dataset = datasets[3]

retrievers = ['BM25', 'miniLM', 'openai-embedding', 'codeT5']
retriever = retrievers[2]

analysis_types = ['retrieval_recall', 'retrieval_doc_type', 'retrieval_doc_selection']
analysis_type = analysis_types[0]
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
    #     else: script += script_temp + ';   '
    script = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n {n} --dataset {dataset} "
              f"--retriever {retriever} --analysis_type {analysis_type} --ret_acc 0.6")
else:
    script = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n 1 --dataset {dataset} "
              f"--retriever {retriever} --analysis_type {analysis_type}")

print(script)
