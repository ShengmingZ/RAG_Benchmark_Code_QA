import os.path

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

SYS_PROMPT_SELF_GENE_2 = """You are a helpful assistant. Given a question starts with `## Question`, 
You can generate whatever you want at first, then generate the exact answer of the question at the end of your generation, 
the exact answer should start with <answer> and ends with </answer>
"""

SYS_PROMPT_LEAST_TO_MOST = """Follow the examples to solve the last question"""


# SYS_PROMPT_PLAN_AND_SOLVE = """You are a helpful assistant, given a question starts with `## Question` and some potential documents to help you answer the question starts with `## Potential documents`,
# Let’s first understand the problem and devise a plan to solve the problem.
# Then, let’s carry out the plan and solve the problem step by step.
# Finally, output the exact answer tagged with ```
# """


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

def prompt_self_gene_2(question, model):
    user_prompt = f"""
## Question: 
{question}
"""
    sys_prompt = SYS_PROMPT_SELF_GENE_2
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    return prompt_template


# def prompt_self_pad(ellipses, question, model):
#     user_prompt = f"""
# {ellipses}
#
# ## Question:
# {question}
# """
#     sys_prompt = LLAMA_SYS_PROMPT_NO_RET
#     prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
#     return prompt_template


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


NQ_3shot_prompt = """
## Potential documents:
0: of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night

## Question: 
who led the confederate force that captured fort fillmore

## Answer:
```Lieutenant Colonel John R. Baylor```



## Potential documents:
0: in the city and the sun shines on LA.' I didn't like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. 'The bay' fit so nice, 'When the lights go down in the city and the sun shines on the bay.' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single

## Question: 
who sings when the lights go down in the city

## Answer:
```Journey```



## Potential documents:
0: Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell

## Question: 
what type of cell has no nucleus or membrane bound organelles

## Answer:
```prokaryote```
"""

# NQ_3shot_answers = ["""<answer>Lieutenant Colonel John R. Baylor</answer>""", """<answer>Journey</answer>""", """<answer>prokaryote</answer>"""]


NQ_CoT_prompt = """
## Potential documents:
0: of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night

## Question: 
who led the confederate force that captured fort fillmore

## Answer:
The document mentions that a force called the 2nd Texas Mounted that was involved in actions against Union forts.
Lieutenant Colonel John R. Baylor led the 2nd Texas Mounted Rifles.
So the answer is ```Lieutenant Colonel John R. Baylor```



## Potential documents:
0: Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell

## Question: 
what type of cell has no nucleus or membrane bound organelles

## Answer:
The document mentions that a prokaryote is a type of cell that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle.
So the answer is ```prokaryote```



## Potential documents:
0: in the city and the sun shines on LA.' I didn't like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. 'The bay' fit so nice, 'When the lights go down in the city and the sun shines on the bay.' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single

## Question: 
who sings when the lights go down in the city

## Answer:
The document mentioned a person was facing joining "Journey".
The document also mentioned the song "the bay" and "When the lights go down in the city and the sun shines on the bay." was the reason why the person wanted to join Journey
So the answer is ```Journey```
"""


def prompt_3shot(ret_docs, question, model):
    # use same samples for NQ and TriviaQA
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{NQ_3shot_prompt}
\n
## Potential documents:
{potential_docs}
## Question: 
{question}

# Answer:
"""

    # prompt_template = ensemble_prompt(sys_prompt='',
    #                                   user_prompt=user_prompt,
    #                                   model=model,
    #                                   # examples=NQ_3shot_examples,
    #                                   # answers=NQ_3shot_answers
    #                                   )
    if 'gpt' in model: prompt = [SYS_PROMPT_LEAST_TO_MOST, user_prompt]
    else: prompt = user_prompt
    return prompt


def prompt_least_to_most(ret_docs, question, model):
    examples_prompt = """
## Potential documents:
0: of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night

## Question: 
who led the confederate force that captured fort fillmore

## Question Solving:
Step1: To answer the question "who led the confederate force that captured Fort Fillmore" using a decomposition approach, we can break it down into subquestions:
1. What was the confederate force that captured Fort Fillmore?
2. Who led this confederate force?

Step2: We then answer each subquestion:
1. What was the confederate force that captured Fort Fillmore?
   - According to the document, there was a force called the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor that was involved in actions against Union forts.

2. Who led this confederate force?
   - The document specifies that Lieutenant Colonel John R. Baylor led the 2nd Texas Mounted Rifles. It also mentions that Baylor decided to take the initiative and launch an attack of his own, implying his leadership in the operations against Union forts along the Rio Grande.

Step3: solve the final question:
Who led the confederate force that captured Fort Fillmore?
   - Based on the information from the document, it is inferred that the Confederate force that captured Fort Fillmore was under the leadership of Lieutenant Colonel John R. Baylor.

Conclusion: Thus, the answer to the question is ```Lieutenant Colonel John R. Baylor```.



## Potential documents:
0: in the city and the sun shines on LA.' I didn't like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. 'The bay' fit so nice, 'When the lights go down in the city and the sun shines on the bay.' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single

## Question: 
who sings when the lights go down in the city

## Question Solving:
Step1: To answer the question "who sings when the lights go down in the city," we can decompose it into subquestions:
1. What is the title of the song mentioned in the document?
2. Who is associated with the creation or performance of this song?

Step2: We then answer each subquestion:
1. What is the title of the song mentioned in the document?
   - The document includes the phrase, "'When the lights go down in the city and the sun shines on the bay,'" suggesting that "When the lights go down in the city" is a part of the song's title. Therefore, the answer to this subquestion is that the title (or significant part of the title) of the song is "When the lights go down in the city."

2. Who is associated with the creation or performance of this song?
   - According to the document, the speaker was facing joining Journey and refers to the song in this context. Therefore, we can infer that the band Journey is associated with the performance of the song.

Step3: solve the final question:
Who sings "When the lights go down in the city"?
   - Based on the answers to the subquestions, the song "When the lights go down in the city" is performed by Journey.

Conclusion: Thus, the answer to the question is ```Journey```.



## Potential documents:
0: Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell

## Question: 
what type of cell has no nucleus or membrane bound organelles

## Question Solving:
Step1: To answer the question "what type of cell has no nucleus or membrane bound organelles," we can decompose it into subquestions:
1. What type of cell has no nucleus?
2. What type of cell lacks membrane-bound organelles?

Step2: We then answer each subquestion:
1. What type of cell has no nucleus?
   - According to the document, a prokaryote is a type of cell that lacks a membrane-bound nucleus. Therefore, the answer to this subquestion is a prokaryote.

2. What type of cell lacks membrane-bound organelles?
   - The document states that prokaryotes lack mitochondria or any other membrane-bound organelle. Therefore, the answer to this subquestion is again, a prokaryote.

Step3: solve the final question:
What type of cell has no nucleus or membrane-bound organelles?
   - Based on the answers to the subquestions, a prokaryote is a type of cell that has no nucleus or membrane-bound organelles.
   
Conclusion: Thus, the answer to the question is ```prokaryote```.
"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{examples_prompt}
\n
## Potential documents:
{potential_docs}
## Question: 
{question}

## Question Solving
"""


    if 'gpt' in model:
        prompt = [SYS_PROMPT_LEAST_TO_MOST, user_prompt]
    else:
        prompt = user_prompt
    # prompt = ensemble_prompt(sys_prompt=SYS_PROMPT_LEAST_TO_MOST, user_prompt=user_prompt, model=model)
    return prompt


def prompt_plan_and_solve(ret_docs, question, model):
    plan_and_solve_prompt = """Let’s first understand the question and devise a plan to solve the question.
Then, let’s carry out the plan and find the answer to the question step by step.
Finally, let's extract and show the exact answer tagged with <answer>"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Question: 
{question}

## Answer:
{plan_and_solve_prompt}
"""
    # sys_prompt = SYS_PROMPT_PLAN_AND_SOLVE
    # prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt


def prompt_cot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{NQ_CoT_prompt}
\n
## Potential documents:
{potential_docs}
## Question: 
{question}

## Answer:
"""
    # sys_prompt = SYS_PROMPT_PLAN_AND_SOLVE
    # prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
    prompt = [SYS_PROMPT_LEAST_TO_MOST, user_prompt] if 'gpt' in model else user_prompt
    return prompt


# def prompt_ape():
#     import sys, platform
#     import random
#     system = platform.system()
#     if system == 'Darwin':
#         root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
#     elif system == 'Linux':
#         root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
#     sys.path.insert(0, root_path)
#     from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
#     from dataset_utils.corpus_utils import WikiCorpusLoader
#     loader = NQTriviaQAUtils('NQ')
#     qs_list = loader.load_qs_list()
#     oracle_list = loader.load_oracle_list()
#     indices = random.sample(range(len(qs_list)), 30)
#     demo_qs_list = indices


def prompt_con(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
## Question: 
{question}
"""
    SYS_PROMPT_CON = """Task Description:
1. Read the given question and potential documents to gather relevant information.
2. Write reading notes summarizing the key points from these documents.
3. Discuss the relevance of the given question and documents.
4. If some documents are relevant to the given question, provide a brief answer based on the documents.
5. If no documents is relevant, directly provide answer without considering the documents.
6. Extract the exact answer tagged with <answer>
"""
    prompt = ensemble_prompt(SYS_PROMPT_CON, user_prompt, model)
    return prompt


def prompt_RaR(ret_docs, question, model):
    RaR_prompt = """You are a helpful assistant, given some Potential Documents and a Question,
Your task is to first rephrase and expand the question, then provide the answer"""

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

    sys_prompt = RaR_prompt
    prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt



if __name__ == '__main__':
    """get random samples as few shot examples"""
    import sys, platform
    import random
    import json
    system = platform.system()
    if system == 'Darwin':
        root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    sys.path.insert(0, root_path)
    from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
    from dataset_utils.corpus_utils import WikiCorpusLoader


    def get_examples(k=30):
        loader = NQTriviaQAUtils('NQ')
        examples = loader.sample_data(k)
        questions = [item['question'] for item in examples]
        oracle_doc_keys = [item['oracle_doc'] for item in examples]
        oracle_docs = WikiCorpusLoader().get_docs([oracle_doc_keys], 'NQ')[0]
        answers = [item['answers'][0] for item in examples]
        processed_examples = list()
        for question, oracle_doc, answer in zip(questions, oracle_docs, answers):
            processed_examples.append(dict(prompt=prompt_0shot(ret_docs=[oracle_doc], question=question, model='gpt-3.5-turbo-0125')[1], answer=answer))
        with open(os.path.join(root_path, 'data/NQ/examples.json'), 'w') as f:
            json.dump(processed_examples, f, indent=2)

    get_examples()

    def get_few_shots(dataset, k):
        loader = NQTriviaQAUtils(dataset)
        qs_list = loader.load_qs_list()
        question_list = [qs['question'] for qs in qs_list]
        random.seed()
        examples = loader.sample_data(k=k*2)
        few_shots = []
        for example in examples:
            if example['question'] not in question_list:
                few_shots.append(example)
            if len(few_shots) == k: break
        print(few_shots)
        doc_keys = [shot['oracle_doc'] for shot in few_shots]
        few_shots_docs = WikiCorpusLoader().get_docs([doc_keys], dataset)[0]
        for doc in few_shots_docs: print(doc)

    NQ_few_shots = [{'question': 'who led the confederate force that captured fort fillmore', 'answers': ['Lieutenant Colonel John R. Baylor'], 'oracle_doc': '5470008'},
                 {'question': 'who sings when the lights go down in the city', 'answers': ['Journey'], 'oracle_doc': '12810729'},
                 {'question': 'what type of cell has no nucleus or membrane bound organelles', 'answers': ['prokaryote'], 'oracle_doc': '12426381'}]
    NQ_few_shots_docs = ["of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night",
                         'in the city and the sun shines on LA.\' I didn\'t like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. \'The bay\' fit so nice, \'When the lights go down in the city and the sun shines on the bay.\' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single',
                         'Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell']
    for idx in range(len(NQ_few_shots)): NQ_few_shots[idx]['answer'] = NQ_few_shots[idx]['answers'][0]

    TriviaQA_few_shots = [{'question': 'In which 70s musical did Paul Michael Glaser star?', 'answers': ['Fiddler on a Roof', 'Fiddler on the roof', 'Sprintze', 'Fiddler On the Roof', '2 life', 'Fiddler On The Roof', 'The Fiddler on the Roof', 'Fiddler on the Roof', 'Fiddler on the reoof', 'Anatevka'], 'oracle_doc': '4412859'},
                          {'question': 'Who succeeded General Joseph Johnston as the Commander of the Army of Northern Virginia?', 'answers': ['Anne Hill Carter', 'Robert Edward Lee', 'R.E. Lee', 'Lee, Robert Edward', 'Col. Robert E. Lee', 'Gen. Robert E. Lee', 'General Robert E. Lee', 'Robert E. Lee (Confederate general)', 'R E Lee', 'R. E. Lee', 'R.e. lee', 'Robert E. Lee', 'Robert E Lee'], 'oracle_doc': '1916548'},
                          {'question': 'Which of the Canary Islands has the nickname ‘windy island’?', 'answers': ['Fuertaventura', 'Fuerteventura Island', 'Fuerteventura', 'Fuerteventura, Spain', 'Fuerta fentura', 'Fuertoventura'], 'oracle_doc': '14586688'}]
    TriviaQA_few_shots_docs = ["Theatre. He guest starred in an episode of CBS's ""The Mentalist"" on October 1, 2009 titled ""The Scarlet Letter"". In 2013, Glaser revisited ""Fiddler on the Roof"" in a UK stage production on national tour, this time playing the lead character Tevye. In addition to television, film, and theater, Glaser is an avid photographer, writes poetry and is currently working on several children's novels. Glaser has been married twice. He married his first wife, Elizabeth Meyer, in 1980. In August 1981, Meyer contracted HIV through a blood transfusion while giving birth to the couple's first child, Ariel. Meyer did not",
                               "Joseph E. Johnston Joseph Eggleston Johnston (February 3, 1807 – March 21, 1891) was a career United States Army officer, serving with distinction in the Mexican–American War (1846–1848), and Seminole Wars. After Virginia seceded, he entered the Confederate States Army as one of the most senior general officers. (He was unrelated to Confederate general Albert Sidney Johnston, who was killed in early 1862.) Johnston was trained as a civil engineer at the United States Military Academy at West Point, New York, graduating in the same class as Robert E. Lee. He served in Florida, Texas, and Kansas. By 1860 he",
                               "been dismissed by most modern historians, as being based on later forged documents. Evidently drawing from the information provided by Malocello, in 1339 appeared the portolan map by Angelino Dulcert of Majorca showing the Canary island of Lanzarote (named ""Insula de Lanzarotus Marocelus"" and marked by a Genoese shield), as well as the island of ""Forte Vetura"" (Fuerteventura) and ""Vegi Mari"" (Lobos). Although earlier maps had shown fantastical depictions of the ""Fortunate Islands"" (on the basis of their mention in Pliny), this is the first European map where the actual Canary islands make a solid appearance (although Dulcert also includes"]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][7]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][-2]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][2]

    # dataset = 'NQ'
    # get_few_shots(dataset, k=30)
    #
    # dataset = 'NQ'
    # if dataset == 'NQ': few_shots, few_shots_docs = NQ_few_shots, NQ_few_shots_docs
    # else: few_shots, few_shots_docs = TriviaQA_few_shots, TriviaQA_few_shots_docs
    # for shot, doc in zip(few_shots, few_shots_docs):
    #     prompt = prompt_0shot(ret_docs=[doc], question=shot['question'], model='gpt-3.5-turbo-0125')[1]
    #     prompt += f'\n\n## Answer:\n<answer>{shot["answer"]}</answer>'
    #     print(prompt)
