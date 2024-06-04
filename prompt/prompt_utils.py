from dataset_utils.corpus_utils import PythonDocsLoader

def get_truncated_docs(api_signs):
    docs = PythonDocsLoader().get_docs(api_signs)
    docs = [doc.replace('\n', ' ') for doc in docs]

    max_length = 1000
    import tiktoken

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    truncated_docs = []
    for doc in docs:
        encoded_doc = encoding.encode(doc)
        if len(encoded_doc) > max_length:
            encoded_doc = encoded_doc[:max_length]
            doc = encoding.decode(encoded_doc)
        print(doc)