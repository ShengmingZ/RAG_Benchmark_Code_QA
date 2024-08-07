import os, json
import time
import faiss, h5py
import openai
import backoff
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import numpy as np
from tqdm import tqdm
import platform, sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from generator.generate_utils import get_docs_tokens, _get_generate_func
from retriever.dense_retriever import retrieve
from retriever.dense_encoder import DenseRetrievalEncoder
from retriever.retriever_utils import retriever_config

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
    #     if data.metadata['description'] == prompt_file_for_batch and data.request_counts.failed == 0 and data.status == 'completed':
    #     # if data.metadata['description'] == prompt_file_for_batch:
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
    # assert client.batches.retrieve(batch_id).request_counts.failed == 0
    output_file_id = client.batches.retrieve(batch_id).output_file_id
    content = client.files.content(output_file_id)
    responses = [json.loads(data) for data in content.text.split('\n') if data != '']
    # pad failed responses:
    for idx in range(len(prompts)):
        has_true_flag = False
        for response in responses:
            if int(response['custom_id']) == idx:
                has_true_flag = True
        if not has_true_flag: responses.append(dict(custom_id=str(idx), response=None))
    responses.sort(key=lambda x: int(x['custom_id']))
    outputs_list, logprobs_list = [], []
    for response in responses:
        if response['response'] is None:
            outputs = ['error']
            logprobs = [[-0.000000001]]
        else:
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
            # try:
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
            # except:
            #     # use llama api
            #     raise Exception('out of GPU memory')
            texts.append(text)
            logprobs.append(logprob.tolist())
        del input_ids, outputs
        texts_list.append(texts)
        logprobs_list.append(logprobs)

    return texts_list, logprobs_list


def run_model_for_ir_cot(questions, model, dataset, temperature=0, max_tokens=500, n=1, stop=None):
    max_iter, max_docs = 8, 15
    k = 4 if dataset in ['NQ', 'TriviaQA', 'hotpotQA'] else 2
    assert n==1
    generate_func = _get_generate_func(dataset, no_ret_flag=False, prompt_type='cot')

    # construct retriever
    ret_args = retriever_config(f'--dataset {dataset} --retriever openai-embedding')
    encoder = DenseRetrievalEncoder(ret_args)
    if dataset in ['NQ', 'TriviaQA']:
        def yield_batches_from_hdf5(file_path, dataset_name='wiki_embedding', batch_size=1024):
            with h5py.File(file_path, 'r') as f:
                dataset = f[dataset_name]
                total_size = dataset.shape[0]
                for start in range(0, total_size, batch_size):
                    end = min(start + batch_size, total_size)
                    yield dataset[start:end]

        doc_embed_file = ret_args.corpus_embed_file + '.npy'
        example_embed = next(yield_batches_from_hdf5(doc_embed_file))
        indexer = faiss.IndexFlatIP(example_embed.shape[1])
        for batch in yield_batches_from_hdf5(doc_embed_file):
            indexer.add(batch.astype(np.float32))
    else:
        doc_embed = np.load(ret_args.corpus_embed_file + '.npy').astype(np.float32)
        indexer = faiss.IndexFlatIP(doc_embed.shape[1])
        indexer.add(doc_embed)

    def if_stop(dataset, output):
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            if 'the answer is' in output or output.count('```') > 1:
                return True
        else:
            if 'the code is' in output or ('<code>' in output and '</code>' in output):
                return True
        return False

    output_list, logprobs_list = [], []
    total_outputs_tokens_list, retrieve_times = [], []  # record efficiency
    for question in questions:
        existing_output, existing_logprobs = '', []
        total_output_tokens = 0
        for i in range(1, max_iter+1):
            # retrieve docs
            embedding = encoder.encode(dataset=[question] if existing_output == '' else [existing_output], save_file=None)
            D, I = indexer.search(embedding, k)
            ret_docs = ...
            # ensemble prompts
            prompt = generate_func(ret_docs, question+existing_output, model)
            # run models
            if 'gpt' in model:
                [[output]], [[logprobs]] = chatgpt(prompts=[prompt], model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
            else:
                [[output]], [[logprobs]] = llama(prompts=[prompt], model_name=model, max_new_tokens=max_tokens, temperature=temperature, n=n, stop=stop)
            total_output_tokens += get_docs_tokens(docs=[output], model=model)[0]   # count output tokens of each generation
            output = output.split('.')[0] + '.'  # only take first sentence
            existing_logprobs.extend(logprobs[:get_docs_tokens(docs=[output], model=model)[0]])    # take approximate logprobs of first sentence
            existing_output += output
            if if_stop(dataset, output):
                output_list.append(existing_output)
                logprobs_list.append(existing_logprobs)
                total_outputs_tokens_list.append(total_output_tokens)
                retrieve_times.append(i)




if __name__ == "__main__":
    # prompts = [['You are a helpful assistant', 'hello']]
    # output_lists, _ = chatgpt(prompts=prompts, model='gpt-3.5-turbo-0125')
    # print(output_lists)
    # print(_)
    ...