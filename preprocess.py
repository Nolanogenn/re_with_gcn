"""
Generate graph dataframes for each trex file.
"""

import json
import glob
import pandas as pd
import re
import spacy
import re
import networkx as nx
import numpy as np
from rdflib import URIRef, BNode, Literal, Namespace
from rdflib.namespace import DCTERMS, RDFS
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import cycle, islice

device = torch.device('cpu')
# device = torch.device('cuda')

nlp_en = spacy.load("en_core_web_sm")
layers = [-1]

# we load the model
# we could experiment with other models as well
model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def split_sentences(sample):
    sentence_boundaries = sample['sentences_boundaries']
    sentences = []
    text = sample["text"]
    for boundary in sentence_boundaries:
        start = boundary[0]
        end = boundary[1]
        sentence = text[start:end]
        sentences.append(sentence)
    return sentences, sentence_boundaries


def get_relations(sample):
    sentence_list = []
    sentences, sentence_boundaries = split_sentences(sample)
    triples = sample['triples']

    for i, sentence in enumerate(sentences):
        sentence_dict = {}
        # it looks like some entities do not have boundaries, this would make it difficult to retrieve the tokens
        # let's just not include them for now
        triples_to_get = [x for x in triples if x['sentence_id'] == i and x['object']['boundaries'] and x['subject']['boundaries']]
        if len(triples_to_get) >= 1:
            sentence_dict['sentence'] = sentence
            for rel in triples_to_get:
                if rel['predicate']['boundaries'] is None:
                    rel['predicate']['boundaries'] = sentence_boundaries[i]

            sentence_dict['triples'] = triples_to_get
            sentence_dict['boundaries'] = sentence_boundaries[i]
            sentence_list.append(sentence_dict)
    return sentence_list


# the next two functions are used to extract the embeddings from tokens / sentences
def get_hidden_states(encoded, model, layers):
    with torch.no_grad():
        output = model(**encoded)
    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    return output


def get_words_vector(sent, tokenizer, model, layers):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    # token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, model, layers)


def get_idx(string_list, boundaries, token_offsets):
    ids = []
    for r in range(len(string_list)):
        len_string = len(' '.join(string_list[r:]))
        offset = boundaries[1] - len_string
        ids.append(token_offsets[offset][0])

    return ids


def getDf(sentences):
    df_rel = []

    dict_embeddings = {}

    for enum_sent, sentence in enumerate(sentences):
        try:

            sentence_id = f"sentence_{enum_sent}"

            g = nx.Graph()
            # dict_embeddings = {}
            edge_list = []
            starting_token = 0
            relations_list = []

            sentence_string = sentence['sentence'].replace('  ', ' ')
            re.sub('\W+', ' ', sentence_string).strip()
            sentence_boundaries = sentence['boundaries']
            triples = sentence['triples']
            token_offsets = {}

            sent_embeddings = get_words_vector(sentence_string, tokenizer, model, layers)
            doc_spacy = nlp_en(sentence_string)

            for token in doc_spacy:
                token_offsets[token.idx] = (token.i, token.text)
                token_idx = tokenizer.encode(token.text, add_special_tokens=False)

                token_embeddings = []
                for enum_idx, token_id in enumerate(token_idx):
                    token_embeddings.append(sent_embeddings[starting_token + enum_idx])

                starting_token += 1
                if len(token_embeddings) > 1:
                    token_embeddings = torch.stack(token_embeddings).to(device)
                    token_embeddings = torch.mean(token_embeddings, -2)

                elif len(token_embeddings) == 1:
                    token_embeddings = torch.stack(token_embeddings).to(device)
                else:
                    token_embeddings = torch.rand(1, 768)

                token_embeddings = torch.reshape(token_embeddings, (1, 768))
                start_token = token.idx
                end_token = token.idx + len(token.text)

                g.add_node(token.i,
                           features=token_embeddings,
                           label=token.text,
                           type='token',
                           boundaries=(start_token, end_token
                                       )
                           )
                token_id = f"{sentence_id}_token_{token.i}"
                dict_embeddings[token_id] = token_embeddings

                edge_list.append((token.i, token.head.i, token.dep_))

            for edge in edge_list:
                g.add_edge(edge[0], edge[1], label=edge[2])
            # row_sent = [enum_sent, g, sentence_string]
            # df_sent.append(row_sent)

            for triple in triples:

                subj_uri = triple['subject']['uri']
                subj_string = triple['subject']['surfaceform']
                subj_string_list = subj_string.split()

                subj_boundaries = [x - sentence_boundaries[0] for x in triple['subject']['boundaries']]
                subj_ids = get_idx(subj_string_list, subj_boundaries, token_offsets)

                obj_uri = triple['object']['uri']
                obj_string = triple['object']['surfaceform']
                obj_string_list = obj_string.split()
                obj_boundaries = [x - sentence_boundaries[0] for x in triple['object']['boundaries']]
                obj_ids = get_idx(obj_string_list, obj_boundaries, token_offsets)

                relation_boundary_start = min([subj_boundaries[0], obj_boundaries[0]])
                relation_boundary_end = max([subj_boundaries[1], obj_boundaries[1]])

                ##-----these might be useful in future------
                # rel_boundaries = triple['predicate']['boundaries']
                rel_boundaries = [relation_boundary_start, relation_boundary_end]
                rel_surfaceform = triple['predicate']['surfaceform']
                rel_uri = triple['predicate']['uri']
                relations_list.append((subj_ids, obj_ids, rel_uri))

                #text = data[0][triple['sentence_id']]['text']
                ##----------------------------------------------

                subj_degrees = [g.degree(x) for x in subj_ids]
                subj_id_index = np.argmax(subj_degrees)

                obj_degrees = [g.degree(x) for x in obj_ids]
                obj_id_index = np.argmax(obj_degrees)

                row_rel = [enum_sent, g, sentence_string, rel_uri, rel_boundaries, subj_ids[subj_id_index],
                           obj_ids[obj_id_index], subj_ids, obj_ids]
                df_rel.append(row_rel)
        except:
            pass

    return df_rel


# df_sent = pd.DataFrame(df_sent, columns = df_sent_columns)

if __name__ == '__main__':
    data = [json.load(open(x)) for x in glob.glob('./data/*.json')]
    df_rel_columns = ['id_sentence',
                      'sentence_graph',
                      'sentence_string',
                      'relation_uri',
                      'relation_boundaries',
                      'subj_main_node',  # node with max degree
                      'obj_main_node',  # node with max degree
                      'subj_all_nodes',
                      'obj_all_nodes']

    for file_i, file in enumerate(data):
        print(f"{file_i} / {len(data)}", end='\r')
        df_rel = []
        for doc in file:
            sentences = get_relations(doc)
            to_extend = getDf(sentences)
            df_rel.extend(to_extend)
        df_rel = pd.DataFrame(df_rel, columns=df_rel_columns)
        df_rel.to_csv(f"graph_{file_i}.csv")
