files = ['run.py', 'generate.py']
file = files[1]

actions = ['gene_prompts', 'gene_responses', 'eval_pred']
action = actions[2]

models = ['gpt-3.5-turbo-0125', 'codellama-13b-instruct', 'llama2-13b-chat']
model = models[1]

datasets = ['conala', 'DS1000', 'pandas_numpy_eval']
dataset = datasets[2]

retrievers = ['BM25', 'miniLM', 'openai-embedding', 'codeT5']
retriever = retrievers[2]

batches = [True, False]
batch = batches[1]


if file == 'generate.py':
    script = ''
    for retriever in retrievers:
        script_temp = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n 1 --dataset {dataset} "
                    f"--retriever {retriever} --analysis_type retrieval_doc_selection --doc_selection_type top_5")
        print(script_temp)
        if batch: script += script_temp + ' --batch  &    '
        else: script += script_temp + ';   '
else:
    script = (f"python generator/{file} --action {action} --model {model} --temperature 0 --n 1 --dataset {dataset} "
              f"--retriever {retriever} --analysis_type retrieval_recall")

print(script)
