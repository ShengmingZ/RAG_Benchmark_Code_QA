LLAMA_SYSTEM_PROMPT = """You are a senior python programmer, given some potential api documents starts with `## Potential documents`, a program description starts with `## Problem`, and the unfinished code solution starts with `## Unfinished Code Solution`, 
you should first read the potential documents, and then use the knowledge in documents to complete the code solution according to the problem.
you should only output the completed code solution, and the output code should start with <code> and end with </code>
"""



def llama_0shot_prompt_type1(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Unfinished Code Solution:
{answer}
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
    if model.startswith('llama2') or model.startswith('codellama'):
        prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{user_prompt} [/INST]"""
    elif model.startswith('llama3'):
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>
"""

    return prompt_template