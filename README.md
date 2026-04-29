# BBC News Classification with Cohere

Comparing six different classification approaches on the BBC News dataset using the
Cohere API — embeddings, supervised classifiers, semantic similarity, few-shot chat,
and rerank.

## Dataset

bbc_news_full.csv
2,225 BBC articles across 5 classes:

| class         | count |
|---------------|------:|
| sport         |  511  |
| business      |  510  |
| politics      |  417  |
| tech          |  401  |
| entertainment |  386  |

Train: 20 documents per class (100 total).
Test:  200 documents (stratified).

## Methods compared

1. **LogisticRegression (baseline)** — Cohere `embed-english-v3.0` + sklearn LogReg.
2. **Alternative heads** — same embeddings + kNN, linear SVM, RandomForest.
3. **Semantic similarity** — embed class descriptions, cosine vs test docs, no training.
4. **Command chat (few-shot)** — `command-a-03-2025` with 2 examples per class in prompt.
5. **Rerank (zero-shot)** — `rerank-v3.5` with class descriptions as candidates.



Headline: with 20 training docs/class, **embed + linear classifier** beats every other
approach. Zero-shot methods (rerank, semantic similarity) give a strong floor with no
training data at all.
