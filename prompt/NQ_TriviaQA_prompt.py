from prompt.prompt_utils import ensemble_prompt

LLAMA_SYS_PROMPT = """You are a helpful assistant, given some potential documents starts with `## Potential documents` and a question starts with `## Question`, 
you should first read the potential documents, and then use the knowledge in documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""

LLAMA_SYS_PROMPT_NO_RET = """You are a helpful assistant, given a question starts with `## Question`, you should use your own knowledge to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""

SYS_PROMPT_PRETEND = """You are a helpful assistant, given some potential documents starts with `## Potential documents` and a question starts with `## Question`, 
Your should first pretend that the documents contains useful information to answer the question, then use the knowledge in the documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""

SYS_PROMPT_SELF_GENE = """You are a helpful assistant. Given a question starts with `## Question`, 
your should first use your own knowledge to generate some documents that are helpful to answer the question, the documents should start with <Documents> and end with </Documents>,  
then use these documents to answer the question, the exact answer should start with <answer> and ends with </answer>
"""


def prompt_pretend(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Question: 
{question}
"""
    sys_prompt = SYS_PROMPT_PRETEND
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


def prompt_self_gene(question, model):
    user_prompt = f"""
## Question: 
{question}
"""
    sys_prompt = SYS_PROMPT_SELF_GENE
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


def prompt_self_pad(ellipses, question, model):
    user_prompt = f"""
{ellipses}

## Question: 
{question}
"""
    sys_prompt = LLAMA_SYS_PROMPT_NO_RET
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


def prompt_0shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Question: 
{question}
"""
    sys_prompt = LLAMA_SYS_PROMPT
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


def prompt_0shot_no_ret(question, model, pads=''):
    user_prompt = f"""
{pads}\n
## Question: 
{question}
"""
    sys_prompt = LLAMA_SYS_PROMPT_NO_RET
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


if __name__ == '__main__':
    ...

    # import sys, platform
    # system = platform.system()
    # if system == 'Darwin':
    #     root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    # elif system == 'Linux':
    #     root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    # sys.path.insert(0, root_path)
    # from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
    # from generator.generate_utils import truncate_docs
    # from dataset_utils.corpus_utils import WikiCorpusLoader
    #
    # loader = NQTriviaQAUtils('NQ')
    # qs_list = loader.load_qs_list()
    # oracle_list = loader.load_oracle_list()
    # question = qs_list[0]['question']
    # output = oracle_list[0]['answers']
    # print(question)
    # print(output)
    # docs = WikiCorpusLoader().get_docs([[oracle_list[0]['oracle_doc']]], dataset='NQ')[0]
    # print(docs)
    # print(llama_0shot_prompt(docs, question, 'llama2-13b-chat'))
