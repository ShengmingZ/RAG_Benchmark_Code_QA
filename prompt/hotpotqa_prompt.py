original_prompt_sys = """You are a helpful assistant. Given some potential documents and a question, you should generate the answer of the question based on the documents"""

original_prompt = """
## Potential Documents: 
1: TV board Independent Filmmaker Project Minnesota. To qualify, films have to be set in the state and/or have a strong Minnesota focus. The film had its official debut at the Lagoon Theater in the Uptown neighborhood of Minneapolis on Friday, March 27, 2015. Colin Covert of the "Star Tribune" gave it 3 out of 4 stars, stating "[Coyle] has given us a well-crafted small budget indie touching some painful, funny truths." The Public Domain (film) The Public Domain is a 2015 Drama film set in Minneapolis, Minnesota. The film follows characters whose lives were impacted by the collapse of the
2: weight on the bridge at the time contributed to the catastrophic failure. Help came immediately from mutual aid in the seven-county Minneapolis–Saint Paul metropolitan area and emergency response personnel, charities, and volunteers. Within a few days of the collapse, the Minnesota Department of Transportation (Mn/DOT) planned its replacement with the I-35W Saint Anthony Falls Bridge. Construction was completed rapidly, and it opened on September 18, 2008.<ref name="Mn/DOTbuild"></ref> The bridge was located in Minneapolis, Minnesota\'s largest city and connected the neighborhoods of Downtown East and Marcy-Holmes. The south abutment was northeast of the Hubert H. Humphrey Metrodome, and the north abutment
## Question: How many people were killed in the collapse of the bridge featured in the drama film, The Public Domain ?
## Answer: 13 people
## END


## Potential Documents:
1: William Corcoran Eustis William Corcoran Eustis (July 20, 1862 - November 24, 1921) was a captain in the United States Army and the personal assistant to General John J. Pershing during World War I. He was chairman of the inauguration committee for the first inauguration of Woodrow Wilson in 1913 and started the Loudoun Hunt in 1894. He was born on July 20, 1862 in Paris to George Eustis, Jr. (1828–1872) and Louise Morris Corcoran. He was the grandson of banker and philanthropist William Wilson Corcoran. He laid the cornerstone for the Corcoran Gallery of Art on May 10, 1894,
2: John J. Pershing General of the Armies John Joseph "Black Jack" Pershing (September 13, 1860 – July 15, 1948) was a senior United States Army officer. His most famous post was when he served as the commander of the American Expeditionary Forces (AEF) on the Western Front in World War I, 1917–18. Pershing rejected British and French demands that American forces be integrated with their armies, and insisted that the AEF would operate as a single unit under his command, although some American divisions fought under British command, and he also allowed all-black units to be integrated with the French
## Question: Who was the personal assistant to General of the Armies John Joseph "Black Jack" Pershing during World War I?
## Answer: William Corcoran Eustis
## END


## Potential Documents:
1: Keanu Reeves Keanu Charles Reeves ( ; born September 2, 1964) is a Canadian actor, director, producer, and musician. He gained fame for his starring role performances in several blockbuster films, including comedies from the "Bill and Ted" franchise (1989–1991); action thrillers "Point Break" (1991), "Speed" (1994), and the "John Wick" franchise; psychological thriller "The Devil\'s Advocate" (1997); supernatural thriller "Constantine" (2005); and science fiction/action series "The Matrix" (1999–2003). He has also appeared in dramatic films such as "Dangerous Liaisons" (1988), "My Own Private Idaho" (1991), and "Little Buddha" (1993), as well as the romantic horror "Bram Stoker\'s Dracula" (1992).
2: Jeff Tremaine Jeffrey James Tremaine (born September 4, 1966) is an American showrunner, filmmaker and former magazine editor. He is most closely associated with the "Jackass" franchise, having been involved since the inception of the first TV show. Tremaine is the former editor of the skating culture magazine "Big Brother" and a former art director of the influential BMX magazine "GO" as well as a former professional BMX rider. Jeff was the executive producer on the MTV reality series "Rob and Big" and now works as the executive producer of "Rob Dyrdek\'s Fantasy Factory", "Ridiculousness", "Nitro Circus", and Adult Swim\'s
## Question: Which jobs do Jeff Tremaine and Keanu Reeves share?
## Answer: director, producer
## END


## Potential Documents:
<POTENTIAL DOCUMENTS>
# Question: <QUESTION>
# Answer:
"""


import platform
import sys
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from dataset_utils.corpus_utils import WikiCorpusLoader

if __name__ == '__main__':
    # wiki_loader = WikiCorpusLoader()
    # doc_key_list = ['The Public Domain (film)_1', 'I-35W Mississippi River bridge_1',
    #                 'William Corcoran Eustis_0', 'John J. Pershing_0',
    #                 'Jeff Tremaine_0', 'Keanu Reeves_0']
    # docs = wiki_loader.get_docs(doc_key_list)
    # for item in docs:
    #     print(item)


    qs_list = HotpotQALoader().load_qs_list()
    print(len(qs_list))


    # [['The Public Domain (film)', 1], ['I-35W Mississippi River bridge', 1]]
    #
    # [['William Corcoran Eustis', 0], ['John J. Pershing', 0]]
    #
    # [['Jeff Tremaine', 0], ['Keanu Reeves', 0]]