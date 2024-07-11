import os, json
import time

import openai
import backoff
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=openai.api_key)


@backoff.on_exception(backoff.expo, openai.APIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def chatgpt(prompts, model, temperature=0.7, max_tokens=500, n=1, stop=None):
    outputs_list, logprobs_list = list(), list()
    for prompt in tqdm(prompts, total=len(prompts)):
        sys_prompt, user_prompt = prompt
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, logprobs=True)
        outputs = [choice.message.content for choice in response.choices]
        logprobs = []
        for choice in response.choices:
            logprobs.append([item.logprob for item in choice.logprobs.content])
        outputs_list.append(outputs)
        logprobs_list.append(logprobs)

    return outputs_list, logprobs_list


def chatgpt_batch(prompt_file_for_batch, prompts, model, temperature=0.7, max_tokens=500, n=1, stop=None):
    requests = list()
    for idx, prompt in enumerate(prompts):
        sys_prompt, user_prompt = prompt
        message = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        request = {"custom_id": str(idx), "method": "POST", "url": "/v1/chat/completions",
                   "body": {"model": model,
                            "messages": message,
                            'temperature': temperature,
                            "max_tokens": max_tokens,
                            "n": n,
                            "stop": stop,
                            "logprobs": True}}
        requests.append(request)
    with open(prompt_file_for_batch, 'w+') as f:
        for request in requests:
            f.write(json.dumps(request)+'\n')

    # if same metadata exists, then just get results, else create batch
    batch_id = None
    # batches = client.batches.list()
    # for data in batches.data:
    #     if data.metadata['description'] == prompt_file_for_batch:
    #         batch_id = data.id
    if batch_id is None:
        batch_input_file = client.files.create(file=open(prompt_file_for_batch, 'rb'), purpose='batch')
        response = client.batches.create(input_file_id=batch_input_file.id,
                                         endpoint='/v1/chat/completions',
                                         completion_window='24h',
                                         metadata={'description': prompt_file_for_batch})
        batch_id = response.id
    print('batch_id: ', batch_id)

    # extract batch results
    status = client.batches.retrieve(batch_id).status
    while status != 'completed':
        time.sleep(300)
        status = client.batches.retrieve(batch_id).status
        if status != 'in_progress': print(status)
    assert client.batches.retrieve(batch_id).request_counts.failed == 0
    output_file_id = client.batches.retrieve(batch_id).output_file_id
    content = client.files.content(output_file_id)
    responses = [json.loads(data) for data in content.text.split('\n') if data != '']
    responses.sort(key=lambda x: int(x['custom_id']))
    outputs_list, logprobs_list = [], []
    for response in responses:
        response = response['response']['body']
        outputs = [choice["message"]["content"] for choice in response["choices"]]
        logprobs = []
        for choice in response["choices"]:
            logprob = choice["logprobs"]
            logprobs.append([item['logprob'] for item in logprob['content']])
        outputs_list.append(outputs)
        logprobs_list.append(logprobs)
    return outputs_list, logprobs_list



def llama(prompts, model_name='llama2-13b-chat', max_new_tokens=100, temperature=0.6, n=1, stop=None):
    """
    :param prompts:
    :param model_name:
    :param max_new_tokens:
    :param temperature:
    :return:
    """
    assert model_name in ['llama2-13b-chat', 'codellama-13b-instruct', 'llama3-8b']
    if model_name == 'llama2-13b-chat':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif model_name == 'codellama-13b-instruct':
        model_name = 'codellama/CodeLlama-13b-Instruct-hf'
    elif model_name == 'llama3-8b':
        model_name = 'meta-llama/Meta-Llama-3-8B'
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=access_token, device_map='auto')
    # model = model.to("cuda")

    def process_naive_output(input_len, outputs, tokenizer):
        gen_sequences = outputs.sequences[:, input_len:].cpu()
        scores = outputs.scores
        probs = torch.stack(scores, dim=1).float().softmax(-1).cpu()
        log_probs = np.log(torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1))
        return [tokenizer.decode(i) for i in gen_sequences][0], log_probs

    texts_list, logprobs_list = [], []
    for prompt in tqdm(prompts, total=len(prompts)):
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to("cuda")
        texts, logprobs = [], []
        for _ in range(n):
            try:
                if temperature == 0:
                    outputs = model.generate(
                        input_ids=input_ids,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                text, logprob = process_naive_output(input_ids.shape[-1], outputs, tokenizer)
            except:
                # use llama api
                raise Exception('out of GPU memory')
            texts.append(text)
            logprobs.append(logprob.tolist())
        texts_list.append(texts)
        logprobs_list.append(logprobs)

    return texts_list, logprobs_list



if __name__ == "__main__":
    # prompts = [['You are a helpful assistant', 'hello']]
    # output_lists, _ = chatgpt(prompts=prompts, model='gpt-3.5-turbo-0125')
    # print(output_lists)
    # print(_)
    ...