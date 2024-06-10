LLAMA_SYSTEM_PROMPT = """You are a senior python programmer, given some potential api documents starts with `## Potential documents`, and a unfinished code snippet starts with `## Unfinished Code Snippet`, 
you should first read the potential documents, and then use the knowledge in documents to complete the code snippet according to the comments in the code.
you should only output the uncompleted part of the code snippet, and the output code should starts with <code> and ends with </code>
"""



def llama_0shot_prompt(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Unfinished Code Snippet:
{question}
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
    if model.startswith('llama2') or model.startswith('codellama'):
        prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{user_prompt}[/INST]"""
    elif model.startswith('llama3'):
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>
"""

    return prompt_template


if __name__ == '__main__':
    import sys, platform
    import random

    system = platform.system()
    if system == 'Darwin':
        root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    sys.path.insert(0, root_path)
    from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
    from generator.generate_utils import truncate_docs
    from dataset_utils.corpus_utils import PythonDocsLoader

    loader = PandasNumpyEvalLoader()
    qs_list = loader.load_qs_list()
    oracle_list = loader.load_oracle_list()
    question = qs_list[0]['question']
    output = oracle_list[0]['output']
    print(question)
    print(output)
    docs = PythonDocsLoader().get_docs(oracle_list[0]['oracle_docs'])
    print(llama_0shot_prompt(docs, question, 'codellama-13b-instruct'))
