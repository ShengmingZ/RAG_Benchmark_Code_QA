# RAG_Benchmark_Code_QA

The code and artifacts collection for paper: Understanding the Design Decisions of Retrieval-Augmented Generation Systems
https://arxiv.org/abs/2411.19463v2

## construct_corpus

preparing retrieval corpus for Code datasets, and automatically identify API functions used in authentic solution

### crawl_docs.py
crawl Python API documentations as corpus
output api_doc_builtin.json, api_doc_third_party.json, api_sign_builtin.txt, api_sign_third_party.txt in dir data/python_docs

### get_oracle_docs.py
get Python docs of API function callings in authentic code solutions
output oracle_docs_matched.json in dir: data/DS1000 / data/pandas-numpy-eval / data/conala

### match_oracle_docs.py
match the authentic python docs with API signs
output oracle_docs_matched_processed.json in dir: data/DS1000 / data/pandas-numpy-eval / data/conala
This file would store the authentic API signtures

## retriever

dense_encoder.py utlize dense encoders sentence-transformers, text-embedding-3, contriever, T5Encoder, ... to encode text into vectors
dense_retriever.py embed corpus and questions, using faiss to calculate similarity score to perform the retrieving
sparse_retriever.py perform BM25 retrieving, leveraging ElasticSearch
rerank.py use cohere, ... as reranker to rerank the naive retrieving results
RetrievalProvider.py: the main interface to provide retrieved documents of different retrievers and datasets, and also do the retrieval recall controlling, provide documents with controlled retrieval recalls.

## llms

provide LLM calling interface

## dataset_utils

provide interface to datasets, perform 

## data_processing

results.py store the statistical results of experiments
analyze_results.py compares the difference between 2 prediction distributions
make_graph.py draws all kinds of figures
significance_analysis.py conducts significant tests

## data

Store the dataset samples, retrieval results, experimental results of each dataset

### conala

refer to https://github.com/shuyanzhou/docprompting

### DS1000

refer to https://github.com/xlang-ai/DS-1000

### pandas_numpy_eval

refer to https://github.com/microsoft/PyCodeGPT/tree/main/cert

### hotpotQA

refer to https://github.com/hotpotqa/hotpot

### NQ

refer to https://github.com/google-research-datasets/natural-questions

### TriviaQA

refer to https://github.com/mandarjoshi90/triviaqa

