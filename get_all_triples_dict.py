import glob
import json

if __name__ == '__main__':
    triples = []
    for trex_file in glob.glob('./data/trex/raw/re-*.json'):
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
    print('overall triples:', len(triples))
    with open('./data/trex/triples.json', 'w') as triples_out:
        json.dump(triple, triples_out, indent=4)
