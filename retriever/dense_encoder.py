import argparse
import shlex
import json, os
import openai
import numpy as np
import torch
import transformers
from transformers import PreTrainedModel, AutoConfig, AutoTokenizer, RobertaModel, AutoModel
from sentence_transformers import SentenceTransformer

openai.api_key = os.environ['OPENAI_API_KEY']

# class RetrievalModel(PreTrainedModel):
#     def __init__(self, config, model_name, tokenizer, model_args, batch_size=64, all_layers=False):
#         super().__init__(config)
#         self.all_layers = all_layers
#         self.model_args = model_args
#         self.batch_size = batch_size
#         self.model_name = model_name
#         self.tokenizer = get_tokenizer(model_name, use_fast=True) if tokenizer is None else tokenizer
#         self.model = get_model(self.model_name)
#
#     def get_pooling_embedding(self, input_ids, attention_mask, lengths, pooling="mean", normalize=False):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.all_layers)
#         if self.all_layers:
#             emb = torch.stack(output[-1], dim=2)
#         else:
#             emb = output[0]
#
#         emb.masked_fill_(~attention_mask.bool().unsqueeze(-1), 2)
#         max_len = max(lengths)
#         base = torch.arange(max_len, dtype=torch.long).expand(len(lengths), max_len).to(lengths.device)
#         pad_mask = base < lengths.unsqueeze(1)
#         emb = (emb * pad_mask.unsqueeze(-1)).sum(dim=1) / pad_mask.sum(-1).unsqueeze(-1)
#         if normalize: emb = emb / emb.norm(dim=1, keepdim=True)
#
#         return emb
#
#
# class CodeT5Retriever:
#     def __init__(self, args):
#         self.args = args
#         self.model_name = self.args.model_name
#         self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
#         # self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
#         self.batch_size = args.batch_size
#         self.normalize_embed = args.normalize_embed
#
#         config = AutoConfig.from_pretrained(self.model_name)
#         class Dummy():
#             def __init__(self, sim_func):
#                 self.sim_func = sim_func
#         model_arg = Dummy(args.sim_func)
#
#         self.model = RetrievalModel(model_name=self.model_name,
#                                     tokenizer=self.tokenizer,
#                                     config=config,
#                                     model_args=model_arg)
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         self.model.eval()
#         self.model.to(self.device)
#
#     def encode(self, text_list, save_file):
#         dataset = text_list
#         print(f"number of sentences to encode: {len(dataset)}")
#
#         with torch.no_grad():
#             all_embeddings = []
#             for i in range(0, len(dataset), self.batch_size):
#                 batch = dataset[i:i + self.batch_size]
#
#                 # pad batch
#                 sent_features = self.tokenizer(batch, add_special_tokens=True,
#                                                    max_length=self.tokenizer.model_max_length, truncation=True)
#                 arr = sent_features['input_ids']
#                 lens = torch.LongTensor([len(a) for a in arr])
#                 max_len = lens.max().item()
#                 padded = torch.ones(len(arr), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
#                 mask = torch.zeros(len(arr), max_len, dtype=torch.long)
#                 for i, a in enumerate(arr):
#                     padded[i, : lens[i]] = torch.tensor(a, dtype=torch.long)
#                     mask[i, : lens[i]] = 1
#                 padded_batch = {'input_ids': padded, 'attention_mask': mask, 'lengths': lens}
#                 for key in padded_batch:
#                     if isinstance(padded_batch[key], torch.Tensor):
#                         padded_batch[key] = padded_batch[key].to(self.device)
#
#                 # get sentence embedding by calc mean embedding
#                 # input_ids, attention_mask, lengths = padded.to(self.device), mask.to(self.device), lens.to(self.device)
#                 input_ids, attention_mask, lengths = padded_batch['input_ids'], padded_batch['attention_mask'], padded_batch['lengths']
#                 output = self.model.model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
#                 emb = output['last_hidden_state']
#                 emb.masked_fill_(~attention_mask.bool().unsqueeze(-1), 0)
#                 # base = torch.arange(max_len, dtype=torch.long).expand(len(lengths), max_len).to(lengths.device)
#                 # pad_mask = base < lengths.unsqueeze(1) # pad token set to false
#                 # emb = (emb*pad_mask.unsqueeze(-1)).sum(dim=1) / pad_mask.sum(-1).unsqueeze(-1)
#                 emb = emb.sum(dim=1) / lengths.unsqueeze(-1)
#                 if self.normalize_embed:
#                     emb = emb / emb.norm(dim=1, keepdim=True)
#                 all_embeddings.append(emb)
#
#             all_embeddings = np.concatenate(all_embeddings, axis=0)
#             print(f"done embedding: {all_embeddings.shape}")
#
#             if not os.path.exists(os.path.dirname(save_file)):
#                 os.makedirs(os.path.dirname(save_file))
#             np.save(save_file, all_embeddings)


class DenseRetrievalEncoder:
    def __init__(self, args):
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.top_k = args.top_k
        self.sim_func = args.sim_func
        self.normalize_embed = args.normalize_embed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # todo: load more type of tokenizer and model
        if 'sentence-transformers' in self.model_name:
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = None
            self.model.to(self.device)
        elif 'text-embedding' in self.model_name:
            self.model = None
            self.tokenizer = None
        else:
            if 't5' in self.model_name:
                self.model = transformers.T5EncoderModel.from_pretrained(self.model_name)
            elif 'roberta' in self.model_name:
                self.model = RobertaModel.from_pretrained(self.model_name)
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)


    # todo: max token lens: 512, now truncate if input text too long
    def encode(self, dataset, save_file):
        import tiktoken
        if 'sentence-transformers' in self.model_name:
            all_embeddings = []
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                all_embeddings.append(self.model.encode(batch))

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"done embedding: {all_embeddings.shape}")
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, all_embeddings)
            return

        if 'text-embedding' in self.model_name:
            OPENAI_TOKENIZER = "cl100k_base"
            OPENAI_MAX_TOKENS = 500
            encoding = tiktoken.get_encoding(OPENAI_TOKENIZER)
            all_embeddings = []
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                # truncate
                for j in range(len(batch)):
                    encoded_doc = encoding.encode(batch[j])[:OPENAI_MAX_TOKENS]
                    batch[j] = encoding.decode(encoded_doc)
                response = openai.Embedding.create(model=self.model_name, input=batch)
                embeds = [data["embedding"] for data in response['data']]
                all_embeddings.append(np.array(embeds))

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"done embedding: {all_embeddings.shape}")
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, all_embeddings)
            return


        with torch.no_grad():
            all_embeddings = []
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]

                # tokenize
                sent_features = self.tokenizer(batch, add_special_tokens=True, max_length=self.tokenizer.model_max_length, truncation=True)
                arr = sent_features['input_ids']

                # pad batch
                lens = torch.LongTensor([len(a) for a in arr])
                max_len = lens.max().item()
                padded = torch.ones(len(arr), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
                mask = torch.zeros(len(arr), max_len, dtype=torch.long)
                for i, a in enumerate(arr):
                    padded[i, : lens[i]] = torch.tensor(a, dtype=torch.long)
                    mask[i, : lens[i]] = 1

                # get embedding
                input_ids, attention_mask, lengths = padded.to(self.device), mask.to(self.device), lens.to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
                if 't5' in self.model_name:
                    emb = output['last_hidden_state']
                elif 'roberta' in self.model_name:
                    emb = output.last_hidden_state

                # pooling token embedding to get sentence embedding
                emb.masked_fill_(~attention_mask.bool().unsqueeze(-1), 0)
                emb = emb.sum(dim=1) / lengths.unsqueeze(-1)
                if self.normalize_embed:
                    emb = emb / emb.norm(dim=1, keepdim=True)
                all_embeddings.append(emb.cpu())

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            print(f"done embedding: {all_embeddings.shape}")

            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, all_embeddings)


    # todo: training
    def train(self):
        pass


if __name__ == '__main__':
    print(openai.api_key)