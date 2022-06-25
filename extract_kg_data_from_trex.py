import glob
import json
import re
from tqdm import tqdm

"""
Extracts all triples and Wikidata entities from T-REx and writes them to file.
"""

if __name__ == '__main__':
    triples = []
    for trex_file in tqdm(glob.glob('./data/trex/raw/re-*.json')):
        with open(trex_file) as trex_in:
            trex_json = json.load(trex_in)
            for doc in trex_json:
                for triple in doc['triples']:
                    try:
                        h = triple['subject']['uri']
                        r = triple['predicate']['uri']
                        t = triple['object']['uri']
                        triples.append((h, r, t))
                    except KeyError:
                        pass
    print('overall triples in trex:', len(triples))
    with open('./data/trex/processed/triples.json', 'w') as triples_out:
        json.dump(triples, triples_out, indent=4)

    entities = set([triples[0] for triples in triples] + [triples[2] for triples in triples])

    with open('./data/trex/processed/entities.txt', 'w') as f:
        for entity in entities:
            if re.match(r'http:\/\/www\.wikidata\.org\/entity\/Q\d*', entity):
                f.write(entity.replace('www.wikidata.org/entity', 'wd') + '\n')
