import os
import openai
import backoff
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY","")


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chatgpt(prompt, model='gpt-3.5-turbo-1106', temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, logprobs=True)
    outputs = [choice["message"]["content"] for choice in response["choices"]]
    logprobs = []
    for choice in response["choices"]:
        logprob = choice["logprobs"]
        logprobs.append([item['logprob'] for item in logprob['content']])

    return outputs, logprobs

def llama(prompts, model_name='llama2-13b-chat', max_new_tokens=1000, temperature=0.7):
    """
    todo: need to use system prompt template or code infill prompt template
    https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/discussions/19
    https://github.com/huggingface/transformers/blob/9e87618f2be1401df55c36ad726629ae201e8e4d/src/transformers/models/code_llama/tokenization_code_llama.py#L465-L466
    https://huggingface.co/blog/codellama#code-infilling
    :param prompts:
    :param model_name:
    :param max_new_tokens:
    :param temperature:
    :return:
    """
    assert model_name in ['llama2-13b-chat', 'codellama-13b-instruct']
    if model_name == 'llama2-13b-chat':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif model_name == 'codellama-13b-instruct':
        model_name = 'codellama/CodeLlama-13b-Instruct-hf'
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=access_token)
    model = model.to("cuda")

    def process_naive_output(input_len, outputs, tokenizer):
        gen_sequences = outputs.sequences[:, input_len:].cpu()
        scores = outputs.scores
        probs = torch.stack(scores, dim=1).float().softmax(-1).cpu()
        log_probs = np.log(torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1))
        return [tokenizer.decode(i) for i in gen_sequences][0], log_probs

    texts, logprobs = [], []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to("cuda")
        outputs = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        # print(outputs)
        text, logprob = process_naive_output(input_ids.shape[-1], outputs, tokenizer)
        texts.append(text)
        logprobs.append(logprob)
        print(text)
        print(logprob)

    return texts, logprobs



if __name__ == "__main__":
    # outputs, logprobs = chatgpt("hello, world")
    # print(outputs)
    # print(logprobs)
    llama(['1+1='])