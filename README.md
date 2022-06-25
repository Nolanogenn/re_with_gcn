

## Create RDF2Vec Embeddings

### Wikidata
1. Extract Wikidata entities and triples from T-REx: `extract_kg_data_from_trex.py`
2. Remove those triples from Wikidata, s.t. the relations that will be predicted are not implicitly included in the embeddings: `remove_triples_from_wikidata.py`
3. Train RDF2Vec (Lite) with the jRDF2Vec Framework for all relevant entities: `nohup java -Xmx200g -jar jrdf2vec-1.2-SNAPSHOT.jar -graph ~/knowledgegraphs/wikidata-20160328_reduced_by_trex.nt -light ~/re_with_gcn/data/trex/processed/entities.txt -numberOfWalks 100 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./re_with_gcn_wikidata > ~/rdf2vec_re_with_gcn_wikidata.txt &`

### WordNet
1. `nohup java -Xmx200g -jar jrdf2vec-1.2-SNAPSHOT.jar -graph ~/knowledgegraphs/wordnet.nt -numberOfWalks 100 -depth 4 -walkGenerationMode "MID_WALKS" -walkDirectory ./re_with_gcn_wordnet > ~/rdf2vec_re_with_gcn_wordnet.txt &`