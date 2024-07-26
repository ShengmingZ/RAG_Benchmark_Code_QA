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


NQ_3shot_prompt = """
## Potential documents:
0: of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night

## Question: 
who led the confederate force that captured fort fillmore

## Answer:
<answer>Lieutenant Colonel John R. Baylor</answer>



## Potential documents:
0: in the city and the sun shines on LA.' I didn't like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. 'The bay' fit so nice, 'When the lights go down in the city and the sun shines on the bay.' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single

## Question: 
who sings when the lights go down in the city

## Answer:
<answer>Journey</answer>



## Potential documents:
0: Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell

## Question: 
what type of cell has no nucleus or membrane bound organelles

## Answer:
<answer>prokaryote</answer>
"""

def prompt_3shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{NQ_3shot_prompt}    
    
## Potential documents:
{potential_docs}

## Question: 
{question}

## Answer:
"""

    prompt_template = ensemble_prompt(sys_prompt='', user_prompt=user_prompt, model=model)
    return prompt_template


if __name__ == '__main__':
    """get random samples as few shot examples"""
    import sys, platform
    import random
    system = platform.system()
    if system == 'Darwin':
        root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    sys.path.insert(0, root_path)
    from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
    from dataset_utils.corpus_utils import WikiCorpusLoader

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

    # dataset = 'TriviaQA'
    # get_few_shots(dataset, k=3)

    TriviaQA_few_shots = [{'question': 'In which 70s musical did Paul Michael Glaser star?', 'answers': ['Fiddler on a Roof', 'Fiddler on the roof', 'Sprintze', 'Fiddler On the Roof', '2 life', 'Fiddler On The Roof', 'The Fiddler on the Roof', 'Fiddler on the Roof', 'Fiddler on the reoof', 'Anatevka'], 'oracle_doc': '4412859'},
                          {'question': 'Who succeeded General Joseph Johnston as the Commander of the Army of Northern Virginia?', 'answers': ['Anne Hill Carter', 'Robert Edward Lee', 'R.E. Lee', 'Lee, Robert Edward', 'Col. Robert E. Lee', 'Gen. Robert E. Lee', 'General Robert E. Lee', 'Robert E. Lee (Confederate general)', 'R E Lee', 'R. E. Lee', 'R.e. lee', 'Robert E. Lee', 'Robert E Lee'], 'oracle_doc': '1916548'},
                          {'question': 'Which of the Canary Islands has the nickname ‘windy island’?', 'answers': ['Fuertaventura', 'Fuerteventura Island', 'Fuerteventura', 'Fuerteventura, Spain', 'Fuerta fentura', 'Fuertoventura'], 'oracle_doc': '14586688'}]
    TriviaQA_few_shots_docs = ["Theatre. He guest starred in an episode of CBS's ""The Mentalist"" on October 1, 2009 titled ""The Scarlet Letter"". In 2013, Glaser revisited ""Fiddler on the Roof"" in a UK stage production on national tour, this time playing the lead character Tevye. In addition to television, film, and theater, Glaser is an avid photographer, writes poetry and is currently working on several children's novels. Glaser has been married twice. He married his first wife, Elizabeth Meyer, in 1980. In August 1981, Meyer contracted HIV through a blood transfusion while giving birth to the couple's first child, Ariel. Meyer did not",
                               "Joseph E. Johnston Joseph Eggleston Johnston (February 3, 1807 – March 21, 1891) was a career United States Army officer, serving with distinction in the Mexican–American War (1846–1848), and Seminole Wars. After Virginia seceded, he entered the Confederate States Army as one of the most senior general officers. (He was unrelated to Confederate general Albert Sidney Johnston, who was killed in early 1862.) Johnston was trained as a civil engineer at the United States Military Academy at West Point, New York, graduating in the same class as Robert E. Lee. He served in Florida, Texas, and Kansas. By 1860 he",
                               "been dismissed by most modern historians, as being based on later forged documents. Evidently drawing from the information provided by Malocello, in 1339 appeared the portolan map by Angelino Dulcert of Majorca showing the Canary island of Lanzarote (named ""Insula de Lanzarotus Marocelus"" and marked by a Genoese shield), as well as the island of ""Forte Vetura"" (Fuerteventura) and ""Vegi Mari"" (Lobos). Although earlier maps had shown fantastical depictions of the ""Fortunate Islands"" (on the basis of their mention in Pliny), this is the first European map where the actual Canary islands make a solid appearance (although Dulcert also includes"]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][7]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][-2]
    TriviaQA_few_shots[0]['answer'] = TriviaQA_few_shots[0]['answers'][2]

    dataset = 'NQ'
    if dataset == 'NQ': few_shots, few_shots_docs = NQ_few_shots, NQ_few_shots_docs
    else: few_shots, few_shots_docs = TriviaQA_few_shots, TriviaQA_few_shots_docs
    for shot, doc in zip(few_shots, few_shots_docs):
        prompt = prompt_0shot(ret_docs=[doc], question=shot['question'], model='gpt-3.5-turbo-0125')[1]
        prompt += f'\n\n## Answer:\n<answer>{shot["answer"]}</answer>'
        print(prompt)
