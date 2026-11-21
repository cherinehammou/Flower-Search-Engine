# 🌸 Petal: Flower Search Engine

A hybrid search engine over a database of 150 flowers. It combines a **TF-IDF vector
search implemented from scratch** with **semantic reranking by an LLM (Claude)**, so you
can search in natural language, e.g. *"fragrant pink wedding flowers"*.

![Petal demo](docs/demo.gif)

## How it works

1. **Indexing:** each flower's title, description, tags and keywords are tokenized and
   converted into a TF-IDF vector (with smoothed IDF) when the page loads.
2. **Retrieval:** the query is vectorized the same way, and every flower is scored by
   **cosine similarity** on the sparse vectors. The top 12 candidates are kept.
3. **Semantic reranking:** the candidates are sent to the Claude API, which scores each one
   for semantic relevance (0 to 1). The final score blends **35% TF-IDF + 65% semantic score**.
4. **Explainability:** for the top 3 results, Claude highlights the description words most
   related to the query (HIGH / MID / LOW), and each card shows its TF-IDF, semantic and
   blended scores.

If the API call fails, the app falls back gracefully instead of crashing.

## Features

- TF-IDF, cosine similarity and tokenization written by hand, with no ML libraries
- Two-stage retrieval: fast keyword search, then LLM reranking
- Live bar visualization of the query's TF-IDF vector
- Score breakdown and highlighted keywords for each result
- Clickable search suggestions, keyboard accessibility and a responsive layout
- Animated falling-petal background and a cozy hand-written CSS theme

## Project structure

```
index.html   page layout
style.css    styling and animations
app.js       TF-IDF engine, Claude reranking, search pipeline, UI
flowers.js   the 150-flower dataset (title, latin name, description, tags, keywords)
```

## Run it

No build step. Open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000
```

## Note on the AI reranking

The semantic reranking calls the Anthropic API directly from the browser, without an API key.
Outside a Claude environment, those calls will fail, and the app falls back to
TF-IDF-only ranking. A real deployment should send the requests through a small backend
that keeps the API key secret.

## Possible improvements

- Backend proxy for the API key
- Replace TF-IDF with real embeddings 
- Evaluate ranking quality on a set of test queries
- Filters by color, type and season
