import json
import re

# triples in trex: 20877472
# triples in wikidata before filtering: 56639376
# triples in wikidata after filtering: 52029643

"""
Reads in triples extracted form T-TEx and removes them from Wikidata.
"""

with open('./data/trex/processed/triples.json') as triples_in:
    triples_to_remove = json.load(triples_in)

    triple_strings = []
    for triple in triples_to_remove:
        if re.match(r'http:\/\/www\.wikidata\.org\/entity\/Q\d*', triple[2]):  # only consider relational triples
            head = triple[0].replace('www.wikidata.org/entity', 'wd')
            relation = triple[1].replace('www.wikidata.org/prop/direct', 'wd')
            tail = triple[2].replace('www.wikidata.org/entity', 'wd')
            triple_strings.append('<' + head + '> <' + relation + '> <' + tail + '> .')
    triple_strings = set(triple_strings)

with open('./data/wikidata-20160328_reduced_by_trex.nt', 'w') as wikidata_out:
    with open('./data/wikidata-20160328_reduced.nt') as wikidata_in:
        for line in wikidata_in.readlines():
            if line.rstrip('\n') not in triple_strings:
                wikidata_out.write(line)

print('Processing done.')
print('Computing statistics:')
print('triples in trex:', len(triple_strings))
print('triples in wikidata before filtering', sum(1 for line in open('./data/wikidata-20160328_reduced.nt')))
print('triples in wikidata after filtering', sum(1 for line in open('./data/wikidata-20160328_reduced_by_trex.nt')))












