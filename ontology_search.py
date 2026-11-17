# %% [markdown]
# Load OWL2Vec* ontology embeddings into FAISS search indices:
# one index over the classes of all ontologies, plus one index per ontology.
# Run cell by cell in VS Code (Python interactive) with the .venv interpreter,
# or as a script: python ontology_search.py

# %% Imports and helpers
import os
import re
from itertools import chain
from typing import Dict, List, Tuple, Union

import faiss
import gensim
import numpy as np
import owlready2
from gensim.models.word2vec import Word2Vec

MODEL_PATH = './output/multi_word2vec'
ONTOLOGY_DIR = './ontologies'


def parse_camel_plus(string: str) -> List[str]:
    """Split a CamelCase / snake_case name into lower-cased words."""
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', string)
    words = [w for m in matches for w in m.group(0).replace('_', ' ').split()]
    return [word.lower() for word in words]


def embed_text(sentence: Union[str, List[str]], model: Word2Vec) -> np.ndarray:
    """Average the embeddings of the words the model knows."""
    if isinstance(sentence, str):
        sentence = sentence.split()
    vectors = [model.wv.get_vector(w) for w in sentence if w in model.wv.key_to_index]
    if not vectors:
        raise ValueError(f"no words recognized in the given sentence: {sentence}")
    return np.mean(vectors, axis=0).astype(np.float32)


# %% Load model and ontologies, embed every class twice (IRI + name)
model = gensim.models.Word2Vec.load(MODEL_PATH)

ontologies: List[owlready2.Ontology] = [
    owlready2.get_ontology(os.path.abspath(os.path.join(ONTOLOGY_DIR, f))).load()
    for f in sorted(os.listdir(ONTOLOGY_DIR))
    if f.endswith(('.owl', '.rdf', '.ttl', '.xml'))
]

# ontology IRI -> list of (class IRI, embedding)
onto_iri2embeds: Dict[str, List[Tuple[str, np.ndarray]]] = dict()
for onto in ontologies:
    entries = onto_iri2embeds.setdefault(onto.base_iri, [])
    for cls in onto.classes():
        if cls.iri in model.wv.key_to_index:
            entries.append((cls.iri, model.wv.get_vector(cls.iri)))
        try:
            entries.append((cls.iri, embed_text(parse_camel_plus(cls.name), model)))
        except ValueError:
            pass
    print(f'{onto.base_iri}: {len(entries)} embeddings for {len(list(onto.classes()))} classes')


# %% Build FAISS indices
class SearchIndex:
    """A FAISS cosine-similarity index plus the class IRI of each row."""

    def __init__(self, embeddings: List[Tuple[str, np.ndarray]], dim: int):
        self.iris = [iri for iri, _ in embeddings]
        vectors = np.stack([vec for _, vec in embeddings]).astype(np.float32)
        faiss.normalize_L2(vectors)          # inner product == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        q = query.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)
        # each class has up to 2 rows, so over-fetch then dedupe by IRI
        scores, ids = self.index.search(q, min(len(self.iris), 2 * k))
        results, seen = [], set()
        for score, i in zip(scores[0], ids[0]):
            if i >= 0 and self.iris[i] not in seen:
                seen.add(self.iris[i])
                results.append((self.iris[i], float(score)))
        return results[:k]


name2faiss_index: Dict[str, SearchIndex] = dict()
name2faiss_index['All'] = SearchIndex(list(chain.from_iterable(onto_iri2embeds.values())), model.vector_size)
for ontology_iri, embeddings in onto_iri2embeds.items():
    name2faiss_index[ontology_iri] = SearchIndex(embeddings, model.vector_size)

print('Indices:', {name: idx.index.ntotal for name, idx in name2faiss_index.items()})


# %% Text search
def search_text(text: str, index_name: str = 'All', k: int = 5) -> List[Tuple[str, float]]:
    words = [w for token in text.split() for w in parse_camel_plus(token)]
    return name2faiss_index[index_name].search(embed_text(words, model), k)


if __name__ == '__main__':
    for query in ['cheese pizza', 'spicy topping', 'vegetarian']:
        print(f'\n{query!r}:')
        for iri, score in search_text(query):
            print(f'  {score:.3f}  {iri}')
