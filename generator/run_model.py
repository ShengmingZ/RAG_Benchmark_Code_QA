import os, json
import time
import faiss, h5py
import openai
from collections import OrderedDict
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
from generator.generate_utils import get_docs_tokens, _get_generate_func, truncate_docs
from retriever.dense_retriever import retrieve
from retriever.dense_encoder import DenseRetrievalEncoder
from retriever.retriever_utils import retriever_config
from dataset_utils.corpus_utils import WikiCorpusLoader, PythonDocsLoader

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
    # some hyperparameters
    max_iter = 8
    max_docs = 15 if dataset in ['NQ', 'TriviaQA', 'hotpotQA'] else 10
    k = 4 if dataset in ['NQ', 'TriviaQA', 'hotpotQA'] else 2
    assert n == 1
    generate_func = _get_generate_func(dataset, no_ret_flag=False, prompt_type='cot')

    # prepare for faiss search
    if dataset == 'hotpotQA':
        doc_id_list = WikiCorpusLoader().load_wiki_id(dataset)
    elif dataset == 'NQ' or dataset == 'TriviaQA':
        doc_id_list = WikiCorpusLoader().load_wiki_id(dataset)
    elif dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        doc_id_list = PythonDocsLoader().load_api_signs()
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


    def if_stop(dataset, output, retrieve_times, ret_doc_keys): # test if interleaved retrieve should stop
        if retrieve_times >= max_iter or len(ret_doc_keys) >= max_docs:
            return True
        else:
            if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:   # for qa, split first sentence
                output_first_sent = output_this_round.split('.')[0] + '.'
                if 'the answer is' in output_first_sent or output_first_sent.count('```') > 1:
                    return True
            else:   # for code, '.' might appear in generated code
                if ('<code>' in output and '</code>' in output) or (output.count('```') > 1):
                    return True
        return False

    output_list = ['']*len(questions)
    logprobs_list = [[]]*len(questions)
    ret_doc_keys_list = [[]]*len(questions)
    ret_docs_list = [[]]*len(questions)
    prompts_list = [[]]*len(questions)
    output_tokens_list = [[]]*len(questions)
    input_tokens_list = [[]]*len(questions)
    retrieve_times_list = [0]*len(questions)
    stop_list = [False] * len(questions)

    while False in stop_list:
        # first do retrieving for all non-stop samples, update ret_doc_keys_list
        new_ret_doc_keys_list = []
        for idx, stop_flag in enumerate(stop_list):
            if not stop_flag:
                embedding = encoder.encode(dataset=[questions[idx]] if output_list[idx] == '' else [output_list[idx]], save_file=None)
                _, [retrieved_index] = indexer.search(embedding.astype(np.float32), k)
                ret_doc_keys = [doc_id_list[x] for x in retrieved_index]
                new_ret_doc_keys = [item for item in ret_doc_keys if item not in ret_doc_keys_list[idx]][:max_docs-len(ret_doc_keys_list[idx])]
                ret_doc_keys_list[idx].extend(new_ret_doc_keys)
                new_ret_doc_keys_list.append(new_ret_doc_keys)
                retrieve_times_list[idx] += 1
                print(f'{retrieve_times_list[idx]}th retrieve result: ', ret_doc_keys)
        # get ret_docs, update ret_docs_list
        if dataset in ['NQ', 'TriviaQA', 'hotpotQA']:
            new_ret_docs_list = WikiCorpusLoader().get_docs(new_ret_doc_keys_list, dataset, num_procs=8)
        else:
            new_ret_docs_list = [truncate_docs(docs=PythonDocsLoader().get_docs(oracle_docs), model=model, max_length=1000) for oracle_docs in new_ret_doc_keys_list]
        for idx, new_ret_docs in enumerate(new_ret_docs_list):
            ret_docs_list[idx].extend(new_ret_docs)

        # ensemble prompts
        # prompts = [None]*len(questions)
        for idx, stop_flag in enumerate(stop_list):
            if not stop_flag:
                prompts_list[idx].append(generate_func(ret_docs_list[idx], questions[idx]+output_list[idx], model))
                input_tokens_list[idx].append(get_docs_tokens(docs=[prompts_list[idx][-1]], model=model)[0])

        # run models
        for idx, stop_flag in enumerate(stop_list):
            if not stop_flag:
                if 'gpt' in model:
                    [[output_this_round]], [[logprobs_this_round]] = chatgpt(prompts=[prompts_list[idx][-1]], model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
                else:
                    [[output_this_round]], [[logprobs_this_round]] = llama(prompts=[prompts_list[idx][-1]], model_name=model, max_new_tokens=max_tokens, temperature=temperature, n=n, stop=stop)
                output_tokens_list[idx].append(get_docs_tokens(docs=[output_this_round], model=model)[0])   # count output tokens of each generation
                print(f'{retrieve_times_list[idx]}th generate output: ', output_this_round)

                # check if retrieve should stop, update stop_list, output_list, logprobs_list
                if if_stop(dataset, output_this_round, retrieve_times_list[idx], ret_docs_list[idx]):
                    output_list[idx] += ' ' + output_this_round
                    logprobs_list[idx].extend(logprobs_this_round)
                    stop_list[idx] = True
                else:
                    output_first_sent = output_this_round.split('.')[0] + '.'
                    output_list[idx] += ' ' + output_first_sent
                    logprobs_list[idx].extend(logprobs_this_round[:get_docs_tokens([output_first_sent], model=model)[0]])
                print('processed output: ', output_list[idx])

    return output_list, logprobs_list, ret_doc_keys_list, prompts_list, input_tokens_list, output_tokens_list, retrieve_times_list


if __name__ == "__main__":
    # prompts = [['You are a helpful assistant', 'hello']]
    # output_lists, _ = chatgpt(prompts=prompts, model='gpt-3.5-turbo-0125')
    # print(output_lists)
    # print(_)
    ...