from prompt.prompt_utils import ensemble_prompt

LLAMA_SYS_PROMPT = """You are a helpful assistant, given some potential documents starts with `## Potential documents` and a question starts with `## Question`, 
you should first read the potential documents, and then use the knowledge in documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""

LLAMA_SYS_PROMPT_NO_RET = """You are a helpful assistant, given a question starts with `## Question`, you should use your own knowledge to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
"""
# original_prompt = """
# ## Potential Documents:
# 1: TV board Independent Filmmaker Project Minnesota. To qualify, films have to be set in the state and/or have a strong Minnesota focus. The film had its official debut at the Lagoon Theater in the Uptown neighborhood of Minneapolis on Friday, March 27, 2015. Colin Covert of the "Star Tribune" gave it 3 out of 4 stars, stating "[Coyle] has given us a well-crafted small budget indie touching some painful, funny truths." The Public Domain (film) The Public Domain is a 2015 Drama film set in Minneapolis, Minnesota. The film follows characters whose lives were impacted by the collapse of the
# 2: weight on the bridge at the time contributed to the catastrophic failure. Help came immediately from mutual aid in the seven-county Minneapolis–Saint Paul metropolitan area and emergency response personnel, charities, and volunteers. Within a few days of the collapse, the Minnesota Department of Transportation (Mn/DOT) planned its replacement with the I-35W Saint Anthony Falls Bridge. Construction was completed rapidly, and it opened on September 18, 2008.<ref name="Mn/DOTbuild"></ref> The bridge was located in Minneapolis, Minnesota\'s largest city and connected the neighborhoods of Downtown East and Marcy-Holmes. The south abutment was northeast of the Hubert H. Humphrey Metrodome, and the north abutment
# ## Question: How many people were killed in the collapse of the bridge featured in the drama film, The Public Domain ?
# ## Answer: 13 people
# ## END
#
#
# ## Potential Documents:
# 1: William Corcoran Eustis William Corcoran Eustis (July 20, 1862 - November 24, 1921) was a captain in the United States Army and the personal assistant to General John J. Pershing during World War I. He was chairman of the inauguration committee for the first inauguration of Woodrow Wilson in 1913 and started the Loudoun Hunt in 1894. He was born on July 20, 1862 in Paris to George Eustis, Jr. (1828–1872) and Louise Morris Corcoran. He was the grandson of banker and philanthropist William Wilson Corcoran. He laid the cornerstone for the Corcoran Gallery of Art on May 10, 1894,
# 2: John J. Pershing General of the Armies John Joseph "Black Jack" Pershing (September 13, 1860 – July 15, 1948) was a senior United States Army officer. His most famous post was when he served as the commander of the American Expeditionary Forces (AEF) on the Western Front in World War I, 1917–18. Pershing rejected British and French demands that American forces be integrated with their armies, and insisted that the AEF would operate as a single unit under his command, although some American divisions fought under British command, and he also allowed all-black units to be integrated with the French
# ## Question: Who was the personal assistant to General of the Armies John Joseph "Black Jack" Pershing during World War I?
# ## Answer: William Corcoran Eustis
# ## END
#
#
# ## Potential Documents:
# 1: Keanu Reeves Keanu Charles Reeves ( ; born September 2, 1964) is a Canadian actor, director, producer, and musician. He gained fame for his starring role performances in several blockbuster films, including comedies from the "Bill and Ted" franchise (1989–1991); action thrillers "Point Break" (1991), "Speed" (1994), and the "John Wick" franchise; psychological thriller "The Devil\'s Advocate" (1997); supernatural thriller "Constantine" (2005); and science fiction/action series "The Matrix" (1999–2003). He has also appeared in dramatic films such as "Dangerous Liaisons" (1988), "My Own Private Idaho" (1991), and "Little Buddha" (1993), as well as the romantic horror "Bram Stoker\'s Dracula" (1992).
# 2: Jeff Tremaine Jeffrey James Tremaine (born September 4, 1966) is an American showrunner, filmmaker and former magazine editor. He is most closely associated with the "Jackass" franchise, having been involved since the inception of the first TV show. Tremaine is the former editor of the skating culture magazine "Big Brother" and a former art director of the influential BMX magazine "GO" as well as a former professional BMX rider. Jeff was the executive producer on the MTV reality series "Rob and Big" and now works as the executive producer of "Rob Dyrdek\'s Fantasy Factory", "Ridiculousness", "Nitro Circus", and Adult Swim\'s
# ## Question: Which jobs do Jeff Tremaine and Keanu Reeves share?
# ## Answer: director, producer
# ## END
#
#
# ## Potential Documents:
# <POTENTIAL DOCUMENTS>
# # Question: <QUESTION>
# # Answer:
# """

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
    ...