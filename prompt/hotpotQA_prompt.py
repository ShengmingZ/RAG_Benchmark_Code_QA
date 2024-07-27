from prompt.prompt_utils import ensemble_prompt

LLAMA_SYS_PROMPT = """You are a helpful assistant, given some potential documents starts with `## Potential documents` and a question starts with `## Question`, 
you should first read the potential documents, and then use the knowledge in documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""

LLAMA_SYS_PROMPT_NO_RET = """You are a helpful assistant, given a question starts with `## Question`, you should use your own knowledge to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""


hotpotQA_3shot_prompt = """
## Potential documents:
0: Danielle Prendergast (born September 8, 1990), better known by her stage name Elle Royal (formerly known as Patwa), is an independent Hip-Hop artist hailing from The Bronx, New York. Her breakthrough came in 2010 when her video "What Can I Say" went viral after WorldStarHipHop featured her as the “Female Artist of the Week”. Elle Royal later released the mixtape One Gyal Army under Patwa in 2010, followed by the singles “Jammin”, “Lights”, and “Statements” in 2015 under her current stage name, Elle Royal.
1: WorldStarHipHop is a content-aggregating video blog. Founded in 2005, the site averages 528,726 unique visitors a day. Alexa ranks the site 342nd in site traffic in the United States and 1,212th for worldwide traffic. The site, operated by Worldstar, LLC, was founded at age 33 by Lee "Q" O' Denat, a Hollis, Queens-based hip-hop fan and Grover Cleveland High School dropout. Described by "Vibe" as a "remnant of the Geocities generation," the site regularly features public fighting caught on video, music videos and assorted content targeted to young audiences. O'Denat refers to the site as the "CNN of the ghetto." In 2012, Alexa Internet stated "Compared with all Internet users, its users are disproportionately young people and they tend to be childless, moderately educated men 18–21 who browse from school and work."

## Question: 
Elle Royal's video "What Can I Say" went viral after she was featured as “Female Artist of the Week” by a video blog founded in what year?

## Answer:
<answer>2005</answer>



## Potential documents:
0: The 2003 LSU Tigers football team represented Louisiana State University (LSU) during the 2003 NCAA Division I-A football season. Coached by Nick Saban, the LSU Tigers played their home games at Tiger Stadium in Baton Rouge, Louisiana. The Tigers compiled an 11–1 regular season record and then defeated the No. 5 Georgia Bulldogs in the SEC Championship Game, Afterward, LSU was invited to play the Oklahoma Sooners in the Sugar Bowl for the Bowl Championship Series (BCS) national title. LSU won the BCS National Championship Game, the first national football championship for LSU since 1958.
1: The 2004 Nokia Sugar Bowl, the BCS title game for the 2003 college football season, was played on January 4, 2004 at the Louisiana Superdome in New Orleans, Louisiana. The teams were the LSU Tigers and the Oklahoma Sooners. The Tigers won the BCS National Championship, their second championship, defeating the Sooners by a score of 21-14.

## Question: 
What game did the team with an 11-1 regular season record play in for the BCS title game?

## Answer:
<answer>2004 Nokia Sugar Bowl</answer>



## Potential documents:
0: The 2011 Teen Choice Awards ceremony, hosted by Kaley Cuoco, aired live on August 7, 2011 at 8/7c on Fox. This was the first time that the ceremonies were aired live since the 2007 show.
1: Kaley Christine Cuoco ( ; born November 30, 1985) is an American actress. After a series of supporting film and television roles in the late 1990s, she landed her breakthrough role as Bridget Hennessy on the ABC sitcom "8 Simple Rules", on which she starred from 2002 to 2005. Thereafter, Cuoco appeared as Billie Jenkins on the final season of the television series "Charmed" (2005–2006). Since 2007, she has starred as Penny on the CBS sitcom "The Big Bang Theory", for which she has received Satellite, Critics' Choice, and People's Choice Awards. Cuoco's film work includes roles in "To Be Fat like Me" (2007), "Hop" (2011) and "Authors Anonymous" (2014). She received a star on the Hollywood Walk of Fame in 2014.

## Question: 
What show does the host of The 2011 Teen Choice Awards ceremony currently star on?

## Answer:
<answer>The Big Bang Theory</answer>
"""


def prompt_3shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{hotpotQA_3shot_prompt}

## Potential documents:
{potential_docs}
\n
## Question: 
{question}
"""

    prompt_template = ensemble_prompt('', user_prompt, model)
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


def prompt_0shot_no_ret(question, model):
    user_prompt = f"""
## Question: 
{question}
"""
    sys_prompt = LLAMA_SYS_PROMPT_NO_RET
    prompt_template = ensemble_prompt(sys_prompt, user_prompt, model)
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
    from dataset_utils.hotpotQA_utils import HotpotQAUtils
    from dataset_utils.corpus_utils import WikiCorpusLoader

    def get_few_shots(dataset, k):
        loader = HotpotQAUtils()
        qs_list = loader.load_qs_list()
        question_list = [qs['question'] for qs in qs_list]
        random.seed()
        examples = loader.sample_dataset(k=k * 2)
        few_shots = []
        for example in examples:
            if example['question'] not in question_list:
                del example['context']
                few_shots.append(example)
            if len(few_shots) == k: break
        print(few_shots)
        doc_keys_list = []
        for shot in few_shots:
            doc_keys_list.append([sp[0] for sp in shot['supporting_facts']])
        few_shots_docs = WikiCorpusLoader().get_docs(doc_keys_list, dataset)
        for doc in few_shots_docs: print(doc)


    # dataset = 'hotpotQA'
    # get_few_shots(dataset, k=3)


    few_shots = [{'_id': '5a90af865542990a9849369c', 'answer': '2005', 'question': 'Elle Royal\'s video "What Can I Say" went viral after she was featured as “Female Artist of the Week” by a video blog founded in what year?', 'supporting_facts': [['Elle Royal', 1], ['WorldStarHipHop', 1]], 'type': 'bridge', 'level': 'hard'},
                          {'_id': '5a7499eb55429929fddd8470', 'answer': '2004 Nokia Sugar Bowl', 'question': 'What game did the team with an 11-1 regular season record play in for the BCS title game?', 'supporting_facts': [['2003 LSU Tigers football team', 2], ['2004 Sugar Bowl', 0]], 'type': 'bridge', 'level': 'hard'},
                          {'_id': '5ab2b1e05542997061209685', 'answer': 'The Big Bang Theory', 'question': 'What show does the host of The 2011 Teen Choice Awards ceremony currently star on?', 'supporting_facts': [['2011 Teen Choice Awards', 0], ['Kaley Cuoco', 3]], 'type': 'bridge', 'level': 'hard'}]

    few_shots_docs = [
        ['Danielle Prendergast (born September 8, 1990), better known by her stage name Elle Royal (formerly known as Patwa), is an independent Hip-Hop artist hailing from The Bronx, New York. Her breakthrough came in 2010 when her video "What Can I Say" went viral after WorldStarHipHop featured her as the “Female Artist of the Week”. Elle Royal later released the mixtape One Gyal Army under Patwa in 2010, followed by the singles “Jammin”, “Lights”, and “Statements” in 2015 under her current stage name, Elle Royal.',
         'WorldStarHipHop is a content-aggregating video blog. Founded in 2005, the site averages 528,726 unique visitors a day. Alexa ranks the site 342nd in site traffic in the United States and 1,212th for worldwide traffic. The site, operated by Worldstar, LLC, was founded at age 33 by Lee "Q" O\' Denat, a Hollis, Queens-based hip-hop fan and Grover Cleveland High School dropout. Described by "Vibe" as a "remnant of the Geocities generation," the site regularly features public fighting caught on video, music videos and assorted content targeted to young audiences. O\'Denat refers to the site as the "CNN of the ghetto." In 2012, Alexa Internet stated "Compared with all Internet users, its users are disproportionately young people and they tend to be childless, moderately educated men 18–21 who browse from school and work."'],
        ['The 2003 LSU Tigers football team represented Louisiana State University (LSU) during the 2003 NCAA Division I-A football season. Coached by Nick Saban, the LSU Tigers played their home games at Tiger Stadium in Baton Rouge, Louisiana. The Tigers compiled an 11–1 regular season record and then defeated the No. 5 Georgia Bulldogs in the SEC Championship Game, Afterward, LSU was invited to play the Oklahoma Sooners in the Sugar Bowl for the Bowl Championship Series (BCS) national title. LSU won the BCS National Championship Game, the first national football championship for LSU since 1958.',
         'The 2004 Nokia Sugar Bowl, the BCS title game for the 2003 college football season, was played on January 4, 2004 at the Louisiana Superdome in New Orleans, Louisiana. The teams were the LSU Tigers and the Oklahoma Sooners. The Tigers won the BCS National Championship, their second championship, defeating the Sooners by a score of 21-14.'],
        ['The 2011 Teen Choice Awards ceremony, hosted by Kaley Cuoco, aired live on August 7, 2011 at 8/7c on Fox. This was the first time that the ceremonies were aired live since the 2007 show.',
         'Kaley Christine Cuoco ( ; born November 30, 1985) is an American actress. After a series of supporting film and television roles in the late 1990s, she landed her breakthrough role as Bridget Hennessy on the ABC sitcom "8 Simple Rules", on which she starred from 2002 to 2005. Thereafter, Cuoco appeared as Billie Jenkins on the final season of the television series "Charmed" (2005–2006). Since 2007, she has starred as Penny on the CBS sitcom "The Big Bang Theory", for which she has received Satellite, Critics\' Choice, and People\'s Choice Awards. Cuoco\'s film work includes roles in "To Be Fat like Me" (2007), "Hop" (2011) and "Authors Anonymous" (2014). She received a star on the Hollywood Walk of Fame in 2014.']
    ]


    for shot, docs in zip(few_shots, few_shots_docs):
        prompt = prompt_0shot(ret_docs=docs, question=shot['question'], model='gpt-3.5-turbo-0125')[1]
        prompt += f'\n\n## Answer:\n<answer>{shot["answer"]}</answer>'
        print(prompt)
