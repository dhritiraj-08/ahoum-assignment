# Conversation -> Personality Facet Scorer

Reads a short conversation and scores the personality/behavioural facets it
actually gives evidence for -- out of a 399-facet taxonomy -- using a local
LLM (Ollama + llama3.1), retrieval-first so the model never sees all 399
facets in one prompt, and abstaining explicitly ("insufficient_evidence")
whenever the conversation doesn't support a judgment.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1
python main.py --setup
```

`--setup` checks that the Ollama server is reachable at
`http://localhost:11434` and that `llama3.1` is pulled; it will attempt to
pull it for you if missing.

## Running the pipeline (in order)

```bash
python main.py --audit       # clean/classify the 399 raw facets -> outputs/enriched_facets.csv
python main.py --embed       # build the FAISS retrieval index -> outputs/faiss_index.bin
python main.py --score "I quit my job to backpack across South America with no plan."
python main.py --benchmark   # run the 10-conversation benchmark -> outputs/benchmark_report.json
```

`--audit` and `--embed` only need to be re-run when `data/Facets_Assignment.csv`
changes. `--score` and `--benchmark` reuse the saved index.

## Architecture

```
data/Facets_Assignment.csv
        |
        v
  src/audit.py            -- clean/normalize facet names, classify into 7
        |                     categories, flag observable vs not, assign
        |                     sensitivity + abstention_reason + scoring anchors
        v
outputs/enriched_facets.csv
        |
        v
  src/embeddings.py       -- embed only conversation-observable facets
        |                     (all-MiniLM-L6-v2) into a FAISS index
        v
outputs/faiss_index.bin + observable_facets.json
        |
        v
  src/pipeline.py --------> retrieve_relevant_facets(conversation, top_k=25)
        |                          |
        |                          v
        |                   src/scorer.py -- split into batches of 10,
        |                     one focused Ollama prompt per batch, robust
        |                     JSON parsing, hard-block on medical facets
        v
outputs/pipeline_output_{timestamp}.json
```

Two safety layers keep the system from hallucinating on facets it has no
business judging from a short chat:

1. **Retrieval-time filter**: `audit.py` marks medical/biological,
   spiritual/esoteric, social/demographic, and malformed facets as
   `conversation_observable = False`. `embeddings.py` only ever indexes the
   `True` rows, so those facets are structurally unreachable by retrieval.
2. **Scoring-time filter**: `scorer.py` additionally hard-blocks scoring any
   facet tagged `medical_biological` even if it somehow reaches that stage,
   forcing `"status": "not_observable"` instead of trusting the LLM.

## Known limitations

- Facet classification in `audit.py` is rule-based (keyword/regex
  heuristics), not ML -- it's reproducible and auditable but will
  misclassify some edge-case facet names. Spot-check
  `outputs/enriched_facets.csv` and adjust the keyword lists if you find
  systematic errors (log real ones in `docs/DEBUGGING.md`).
- Retrieval quality depends entirely on `all-MiniLM-L6-v2` embedding
  similarity between the conversation and `"facet name: scoring anchor"`
  text -- a relevant facet whose phrasing is very different from the
  conversation's wording can be missed (a "retrieval miss", tracked
  explicitly in the benchmark report).
- llama3.1 running locally on a 6GB-VRAM GPU is not perfectly reliable at
  emitting clean JSON on every call; `scorer.py` catches parse failures per
  batch and reports them as `"parse_error"` rather than crashing, but a
  parse error still means that batch's facets go unscored for that run.
- Scoring is not deterministic between runs even at low temperature --
  expect small score drift (usually +/-1) if you re-run the same
  conversation.
