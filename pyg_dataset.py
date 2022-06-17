import json
import os.path
import os.path as osp
import glob
import urllib
import urllib.request
import pandas as pd
import spacy
import torch
from random import shuffle
from tqdm import tqdm
from torch.utils.data import IterableDataset
from torch_geometric.data import Dataset, Data, DataLoader, extract_zip
from preprocess import get_relations, getDf


class REDataset(Dataset, IterableDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, debug=True):
        self.name = 'trex'
        self.trex_id = "8768701" if debug else "8760241"
        self.url_trex = f'https://figshare.com/ndownloader/files/{self.trex_id}'
        self.classes = []
        self.num_classes = 0  # number of different relations to predict
        self.nlp_en = spacy.load("en_core_web_sm")
        self.num_relations = len(self.nlp_en.get_pipe("parser").labels)
        self.graph_files = []
        super().__init__(root, transform, pre_transform, pre_filter)

        with open(osp.join(self.processed_dir, 'stats.json'), 'r') as stats_in:
            stats = json.load(stats_in)
            self.name = 'trex'
            self.trex_id = stats['trex_id']
            self.classes = stats['classes']
            self.num_classes = stats['num_classes']
            self.num_relations = stats['num_relations']


    @property
    def raw_file_names(self):
        return ['re-nlg_0-10000.json']  # checking for the first file should be sufficient

    @property
    def processed_file_names(self):
        return [file.split('/')[-1].replace('.json', '.pt') for file in glob.glob(osp.join(self.raw_dir, 're-nlg_*.json'))] + ['relations.json']

    def download(self):
        print('Download dataset from:', self.url_trex)
        urllib.request.urlretrieve(self.url_trex, osp.join(self.raw_dir + f'/{self.trex_id}.zip'))
        print('Unzipping downloaded file')
        extract_zip(osp.join(self.raw_dir + f'/{self.trex_id}.zip'), self.raw_dir)
        print('download')
        pass

    def process(self):
        # create or use a set of all relations which can be predicted
        # this list is used to translate relation URIs into IDs for the classification task
        if not os.path.isfile(osp.join(self.processed_dir, 'relations.json')):
            print('create relations file: relations.json')
            self.classes = []
            triples = []
            for trex_file in glob.glob(osp.join(self.raw_dir, '*re-*.json')):
                with open(trex_file) as trex_in:
                    trex_json = json.load(trex_in)
                    for doc in trex_json:
                        for triple in doc['triples']:
                            try:
                                self.classes .append(triple['predicate']['uri'])
                                triples.append((triple['subject']['uri'],
                                                triple['predicate']['uri'],
                                                triple['object']['uri']))
                            except KeyError:
                                pass
            with open(osp.join(self.processed_dir, 'relations.json'), 'w') as relations_out:
                json.dump(list(set(self.classes)), relations_out, indent=4)
        else:
            print('load relations: relations.json')
            with open(osp.join(self.processed_dir, 'relations.json')) as relations_in:
                self.classes = json.load(relations_in)
        self.num_classes = len(self.classes)
        print(f'Dataset contains unique relations: {len(self.classes)}')

        print('Creating a PyG datasets for each T-REx file...')

        df_rel_columns = ['id_sentence',
                          'sentence_graph',
                          'sentence_string',
                          'relation_uri',
                          'relation_boundaries',
                          'subj_main_node',  # node with max degree
                          'obj_main_node',  # node with max degree
                          'subj_all_nodes',
                          'obj_all_nodes']

        for file_i, file_path in enumerate(tqdm(glob.glob(osp.join(self.raw_dir, 're-nlg_*.json')))):

            # do not process already trex files for which there are already graph files
            if osp.isfile(osp.join(self.processed_dir, file_path.split('/')[-1].replace('.json', '.pt'))):
                print('processed file already exists:', file_path.split('/')[-1].replace('.json', '.pt'))
                continue
            data = json.load(open(file_path))

            # create dataframe
            df_rel = []
            for doc in data:
                sentences = get_relations(doc)
                to_extend = getDf(sentences)
                df_rel.extend(to_extend)
            df_rel = pd.DataFrame(df_rel, columns=df_rel_columns)

            # create PyG data
            graphs = []
            for idx, row in df_rel[['sentence_graph', 'subj_main_node', 'relation_uri', 'obj_main_node']].iterrows():

                g = Data(x=torch.stack(
                    [row['sentence_graph'].nodes[id_node]['features'].flatten(0) for id_node in
                     row['sentence_graph'].nodes]),
                         edge_index=torch.tensor(list(row['sentence_graph'].edges())).T,
                         # head, relation, tail -> objective is to classify the relation right
                         y=torch.tensor([[row['subj_main_node'], self.classes.index(row['relation_uri']), row['obj_main_node']]]),
                         edge_type=torch.tensor([self.nlp_en.get_pipe("parser").labels.index(
                             row['sentence_graph'].get_edge_data(x, y)['label']) for (x, y) in
                                                 row['sentence_graph'].edges()]))
                graphs.append(g)

            torch.save(graphs, osp.join(self.processed_dir, file_path.split('/')[-1].replace('.json', '.pt')))

        with open(osp.join(self.processed_dir, 'stats.json'), 'w') as stats_out:
            json.dump({'name': self.name,
                       'trex_id': self.trex_id,
                       'num_classes': self.num_classes,
                       'classes': self.classes,
                       'num_relations': self.num_relations}, stats_out, indent=4)

    def len(self):
        num_samples = 0
        for file in glob.glob(osp.join(self.processed_dir, f're-nlg_*.pt')):
            graphs = torch.load(file)
            num_samples += len(graphs)
        return num_samples

    def __iter__(self):
        self.graph_files = glob.glob(osp.join(self.processed_dir, f're-nlg_*.pt'))
        shuffle(self.graph_files)
        for graph_file in self.graph_files:
            graphs = torch.load(graph_file)
            shuffle(graphs)
            for g in graphs:
                yield g


if __name__ == '__main__':
    dataset = REDataset('./data/trex', debug=True)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        print(batch)






