from dataset_utils.corpus_utils import PythonDocsLoader

def get_truncated_docs(api_signs):
    docs = PythonDocsLoader().get_docs(api_signs)
    docs = [doc.replace('\n', ' ') for doc in docs]

    max_length = 1000
    import tiktoken

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    truncated_docs = []
    for doc in docs:
        encoded_doc = encoding.encode(doc)
        if len(encoded_doc) > max_length:
            encoded_doc = encoded_doc[:max_length]
            doc = encoding.decode(encoded_doc)
        print(doc)


def ensemble_prompt(sys_prompt, user_prompt, model, examples=None, answers=None):
    if examples is not None: assert len(examples) == len(answers)
    if model.startswith('llama2') or model.startswith('codellama'):
        if examples is None:
            prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{user_prompt}\n[/INST]"""
        else:
            prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{examples[0]}\n[/INST]{answers[0]}</s>\n"""
            for example, answer in zip(examples, answers):
                prompt_template += f"<s>[INST]{example}\n[/INST]{answer}</s>\n"
            prompt_template += f"<s>[INST]{user_prompt}\n[/INST]"
    elif model.startswith('llama3'):
        prompt_template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n"
        if examples is not None:
            for example, answer in zip(examples, answers):
                prompt_template += f"<|start_header_id|>user<|end_header_id|>{example}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id>{answer}<|eot_id|>\n\n"
        prompt_template += f"<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id>"
    elif model.startswith('gpt'):
        shot_prompt = ''
        if examples is not None:
            shots = ''
            for example, answer in zip(examples, answers):
                shots += f'{example}\n{answer}\n\n'
            shot_prompt += shots
        prompt_template = [sys_prompt, shot_prompt+user_prompt]
    else:
        raise ValueError(f'Unrecognized model: {model}')

    return prompt_template
