import os
import openai
import backoff
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY","")


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chatgpt(prompts, model, temperature=0.7, max_tokens=500, n=1, stop=None):
    outputs_list, logprobs_list = list(), list()
    for prompt in tqdm(prompts, total=len(prompts)):
        sys_prompt, user_prompt = prompt
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, logprobs=True)
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
            print(text)
            texts.append(text)
            logprobs.append(logprob.tolist())
        texts_list.append(texts)
        logprobs_list.append(logprobs)

    return texts_list, logprobs_list



if __name__ == "__main__":
    ...