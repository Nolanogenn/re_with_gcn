import os
import pandas as pd
import glob
import pickle
import networkx as nx

files = [f for f in glob.glob('./graphs/enhanced_graph_*.pkl')]
df_test = pd.read_pickle(files[0])
outfile = "./data/trexsemevalformat/trex.semeval"
if os.path.exists(outfile):
    os.remove(outfile)

for i, row in df_test.iterrows():
    sentence_string = row['sentence_string']
    relation_uri = row['relation_uri']
    relation_boundaries = row['relation_boundaries']
    subj_nodes = row['subj_all_nodes']
    starting_subj = subj_nodes[0]
    ending_subj = subj_nodes[-1]

    obj_nodes = row['obj_all_nodes']
    starting_obj = obj_nodes[0]
    ending_obj = obj_nodes[-1]

    sentence_graph = row['sentence_graph']

    e1_start = sentence_graph.nodes[starting_subj]['boundaries'][0]
    e1_end = sentence_graph.nodes[ending_subj]['boundaries'][1]
    e2_start = sentence_graph.nodes[starting_obj]['boundaries'][0]
    e2_end = sentence_graph.nodes[ending_obj]['boundaries'][1]

    sentence_string = f'"{sentence_string[:e1_start]} <e1>{sentence_string[e1_start:e1_end]}</e1> {sentence_string[e1_end:e2_start]} <e2>{sentence_string[e2_start:e2_end]}</e2> {sentence_string[e2_end:]}"'
    line = f"{i+1}\t{sentence_string}\n{relation_uri}\nComment:\n\n"

    with open(outfile, 'a') as f:
        f.write(line)
