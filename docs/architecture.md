# System Architecture
## Author: Monica Kandulapati
## Date: 2026-06-01
## Repository / demo URL: https://github.com/kandulapatimonicavalli/deep-learning-rag-agent — local demo via `uv run streamlit run src/rag_agent/ui/app.py` (not deployed to cloud yet)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (app.py)                                │
│  Sidebar: upload .md/.pdf → ingest    │  Viewer: chunks by source           │
│  Chat: filters + LangGraph invoke      │  session_state + thread_id          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 DocumentChunker          VectorStoreManager      Compiled LangGraph
 (chunker.py)             (store.py / ChromaDB)   (graph.py + nodes.py)
        │                       │                       │
        │  .md / .pdf           │  embed + upsert       │
        ▼                       ▼                       ▼
 ┌──────────────┐        ┌──────────────┐        ┌──────────────────────────┐
 │ Header split │        │ ChromaDB     │        │ query_rewrite → retrieval │
 │ + recursive  │───────►│ Persistent   │◄───────│ → generate | no_context   │
 │ 512 / 50     │ chunks │ cosine HNSW  │ query  │ (MemorySaver threads)   │
 └──────────────┘        └──────────────┘        └──────────────────────────┘
        │                       ▲                       │
        │ SHA-256 chunk_id      │ similarity ≥ 0.3      │ Groq LLM + local
        │ duplicate skip        │                       │ MiniLM embeddings
        └───────────────────────┴───────────────────────┘
```

The diagram shows:
- [x] How a corpus file becomes a chunk — `DocumentChunker` splits by Markdown headers, then `RecursiveCharacterTextSplitter` (512 / 50), prepends topic context for embedding
- [x] How a chunk becomes an embedding — `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) via `VectorStoreManager.ingest()`
- [x] How duplicate detection fires — `generate_chunk_id(source, chunk_text)` → SHA-256; `check_duplicate()` before upsert
- [x] How a user query flows through LangGraph — `query_rewrite` → `retrieval` → conditional → `generation` or `no_context`
- [x] Where the hallucination guard sits — empty retrieval or failed rewrite validation routes to `no_context_node` (skips LLM generation)
- [x] How conversation memory is maintained — `MemorySaver` checkpointer keyed by `thread_id`; `trim_messages` in `generation_node` capped at `MAX_CONTEXT_TOKENS`

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/` (committed markdown); runtime uploads under `data/corpus/uploads/`

- **File formats used:** Markdown (`.md`) for the initial corpus. PDF (`.pdf`) supported by `DocumentChunker` via `PyPDFLoader` but no landmark PDFs ingested yet.

- **Landmark papers ingested:**
  - None as PDFs yet. Papers are **referenced in prose** inside markdown (e.g. LeNet, AlexNet in `cnn_intermediate.md`).
  - Planned: Rumelhart et al. (backprop), LeCun et al. (LeNet), Hochreiter & Schmidhuber (LSTM) when PDF pipeline is used.

- **Chunking strategy:** 512 characters with 50-character overlap (`DocumentChunker.DEFAULT_*`). Markdown is split on `#` / `##` / `###` first so section boundaries stay coherent; sections longer than 512 chars are split recursively. Overlap preserves continuity across splits. Chunks under 80 characters are dropped.

- **Metadata schema:**
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | Canonical label (ANN, CNN, RNN, …) for filtering and citations |
  | difficulty | string | `beginner` / `intermediate` / `advanced` — inferred from filename stem |
  | type | string | e.g. `concept_explanation` — chunk genre for future filtering |
  | source | string | Filename for traceability and document viewer grouping |
  | related_topics | list | Cross-links for multi-topic retrieval and interview questions |
  | is_bonus | bool | Flags GAN, SOM, BoltzmannMachine content when added |

- **Duplicate detection approach:** `chunk_id = SHA-256(f"{source}::{chunk_text}")`. Content hash ensures re-uploading the same file produces identical IDs; filename-only IDs would miss content edits and allow false duplicates across different files with the same name.

- **Corpus coverage:**
  - [x] ANN — `ann_intermediate.md` (4 sections)
  - [x] CNN — `cnn_intermediate.md` (4 sections)
  - [x] RNN — `rnn_intermediate.md` (4 sections)
  - [ ] LSTM — covered in `examples/sample_chunk.json` only; no `lstm_*.md` in corpus yet
  - [ ] Seq2Seq
  - [ ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — `PersistentClient`

- **Local persistence path:** `./data/chroma_db` (`CHROMA_DB_PATH` in `.env`)

- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` / `HuggingFaceEmbeddings` (`EMBEDDING_PROVIDER=local`)

- **Why this embedding model:** Runs fully offline with no API cost; 384-dim vectors are fast on CPU and sufficient for short technical passages. Tradeoff: weaker semantic nuance than `all-mpnet-base-v2` or OpenAI embeddings — acceptable for a study corpus and interview demo.

- **Similarity metric:** Cosine — collection created with `metadata={"hnsw:space": "cosine"}`; scores computed as `1 - cosine_distance`.

- **Retrieval k:** 4 (`RETRIEVAL_K`) — balances context breadth vs. noise and LLM token budget.

- **Similarity threshold:** 0.3 (`SIMILARITY_THRESHOLD`). Starting point from project defaults; calibrated manually during development (scores printed in logs). Below threshold → empty retrieval → hallucination guard.

- **Metadata filtering:** Streamlit selectboxes set `topic_filter` and `difficulty_filter` on `AgentState`; `VectorStoreManager.query()` builds Chroma `where` clauses on `metadata.topic` and `metadata.difficulty`.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Rewrites the latest user message into a keyword-dense query via `QUERY_REWRITE_PROMPT` and Groq |
  | retrieval_node | Embeds rewritten query, retrieves top-k chunks, applies threshold and optional metadata filters; validates rewrite against original query to reduce false positives |
  | generation_node | Builds context block with citations, trims history, invokes LLM with `SYSTEM_PROMPT` + retrieved chunks only |
  | no_context_node | Returns `NO_CONTEXT_RESPONSE` when retrieval fails — no LLM call |

- **Conditional edges:** After `retrieval`, `should_retry_retrieval` returns `"end"` if `no_context_found` → `no_context_node` → `END`; otherwise `"generate"` → `generation_node` → `END`.

- **Hallucination guard:** When no chunk meets the similarity threshold (or rewrite-only match is rejected), the user sees:

```
I was unable to find relevant information in the study corpus for your query.

This may mean:
- The topic is not yet covered in the corpus (check if it is a bonus topic)
- Your query needs to be more specific (try including the exact topic name)
- The corpus needs more content on this area
...
```

(Full text: `NO_CONTEXT_RESPONSE` in `src/rag_agent/agent/prompts.py`.)

- **Query rewriting:**
  - Raw query: `I'm confused about how LSTMs remember things`
  - Rewritten query (typical): `LSTM long-term memory cell state forget gate input output vanishing gradient RNN`

- **Conversation memory:** LangGraph `MemorySaver` persists messages per `thread_id`. `generation_node` uses `trim_messages(..., max_tokens=3000, strategy="last")` so older turns drop first. Memory is in-process only — restarting Streamlit starts a new session unless the user keeps the same `thread_id`.

- **LLM provider:** Groq — `llama-3.1-8b-instant` (default in `.env.example`)

- **Why this provider:** Free tier, low latency, no local GPU required — good for demos and rapid iteration while embeddings stay local for privacy.

---

### Prompt Layer

- **System prompt summary:** Senior ML interviewer persona; must answer only from provided context, cite `[SOURCE: topic | filename]`, refuse to guess, adjust depth to metadata difficulty.

- **Question generation prompt:** Inputs: `{context}` chunk text, `{difficulty}`. Returns JSON with `question`, `model_answer`, `follow_up`, `source_citations`. **Defined in `prompts.py` but not yet wired to the Streamlit UI or graph.**

- **Answer evaluation prompt:** Inputs: `{question}`, `{candidate_answer}`, `{context}`. Returns JSON score 0–10 with rubric, gaps, ideal answer, coaching tip. **Defined but not yet wired to UI.**

- **JSON reliability:** Suffix on JSON prompts: `Respond with the JSON object only. No preamble or explanation.`

- **Failure modes identified:**
  - System prompt: model adds general knowledge → mitigated by strict rules + separate context system message with “Use ONLY the following retrieved study material”
  - Question generation: yes/no or malformed JSON → open-ended requirement + JSON-only instruction (for future UI integration)
  - Answer evaluation: scores too generous → explicit 0–10 rubric in prompt
  - Query rewrite: injects DL terms for off-topic queries → **mitigated in code** by re-querying with original text; if original finds nothing, retrieval is cleared even when rewrite matched

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Local development; Streamlit Community Cloud planned (not deployed)
- **Public URL:** Not deployed — run locally

- **Ingestion panel features:** Sidebar multi-file uploader (`.md`, `.pdf`), save to `data/corpus/uploads/`, chunk + ingest with spinner, success/info for chunks added vs. duplicates skipped, document list with per-source delete

- **Document viewer features:** Dropdown of ingested sources, chunk count caption, expandable chunks with topic/difficulty/type metadata and full `chunk_text`

- **Chat panel features:** Topic and difficulty filters, scrollable history, `st.chat_input`, source expander per assistant message, rewritten-query expander, warning when `no_context_found`, optional confidence when retrieval succeeded

- **Session state keys:**
  | Key | Stores |
  |---|---|
  | chat_history | List of user/assistant dicts (content, sources, flags) |
  | ingested_documents | Cached `list_documents()` from ChromaDB |
  | selected_document | Currently selected source in viewer (if used) |
  | thread_id | LangGraph checkpointer thread (`session-{uuid}`) |
  | last_ingestion_result | Latest `IngestionResult` from upload |
  | topic_filter | Active topic or `None` for “All” |
  | difficulty_filter | Active difficulty or `None` for “All” |

- **Stretch features implemented:** `@st.cache_resource` for embeddings, vector store, graph, and chunker. Streaming responses not implemented.

---

## Design Decisions

1. **Decision:** Chunk size 512 characters with 50-character overlap, header-first Markdown splitting.
   **Rationale:** Headers align chunks with interview-sized concepts; overlap reduces lost context at boundaries. Smaller chunks improve precision but fragment explanations; larger chunks dilute embedding focus.
   **Interview answer:** I split on Markdown sections first, then apply 512/50 recursive splitting so each vector represents one teachable idea while overlap keeps sentences that span a boundary retrievable.

2. **Decision:** Deterministic chunk IDs from `SHA-256(source + "::" + chunk_text)`.
   **Rationale:** Re-ingesting the same material must skip duplicates; content hashing detects identical chunks even if upload order changes. Filename-only IDs would miss edits and allow stale duplicates.
   **Interview answer:** Duplicate detection uses a content hash keyed by source filename so re-uploads are idempotent and chunk identity is stable across sessions.

3. **Decision:** Reject retrieval when only the rewritten query matches off-topic input (original-query validation in `retrieval_node`).
   **Rationale:** Query rewrite can map “capital of France” to RNN-related terms and retrieve false positives; requiring the original wording to also hit the corpus prevents silent hallucination paths.
   **Interview answer:** I treat rewrite as a retrieval hint, not ground truth — if the user’s exact question doesn’t retrieve, I refuse to answer even when a rewritten query would have matched.

4. **Decision:** Local embeddings + Groq for generation.
   **Rationale:** Keeps corpus vectors on-machine; uses a fast hosted LLM for chat without GPU setup. Tradeoff: two-model operational surface vs. single-vendor simplicity.
   **Interview answer:** Embeddings stay local for privacy and cost; the LLM is Groq’s free tier for responsive demos while Chroma stores vectors persistently on disk.

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | e.g. “Explain vanishing gradient” retrieves RNN/ANN chunks; answer cites `[SOURCE: …]` | Pass (manual / design) |
| Off-topic query | No context found message | e.g. “What is the capital of France” → `NO_CONTEXT_RESPONSE`, UI warning | Pass |
| Duplicate ingestion | Second upload skipped | Re-upload same `.md` → `IngestionResult.skipped` > 0, no new IDs | Pass |
| Empty query | Graceful error, no crash | Empty chat input not submitted (`st.chat_input`); blank invoke avoided | Pass |
| Cross-topic query | Multi-topic retrieval | e.g. “How do LSTMs improve on RNNs” — limited until LSTM corpus added; RNN chunks may partial match | Partial |

**Critical failures fixed before Part 3:**
- Rewrite-induced false retrieval for off-topic queries (original-query validation added)
- Streamlit cold-start timeout messaging for Groq / embedding load

**Known issues not fixed (and why):**
- LSTM / Seq2Seq / Autoencoder markdown not authored yet — limits cross-topic and LSTM-specific demos
- `QUESTION_GENERATION_PROMPT` / `ANSWER_EVALUATION_PROMPT` not exposed in UI — time prioritized on core RAG path
- No cloud deployment URL yet

---

## Known Limitations

- Corpus is three intermediate Markdown topics only; no landmark PDFs ingested despite PDF support in code.
- Similarity threshold 0.3 was tuned manually, not with a labeled eval set.
- Conversation memory is in-memory (`MemorySaver`); app restart clears LangGraph threads unless `thread_id` is preserved.
- Question-generation and answer-evaluation flows exist as prompts only, not as graph nodes or UI actions.
- `NO_CONTEXT_RESPONSE` lists topics (LSTM, Seq2Seq, etc.) that are not all present in the on-disk corpus yet.

---

## What I Would Do With More Time

- Add `lstm_intermediate.md`, `seq2seq_intermediate.md`, and `autoencoder_intermediate.md` plus ingest 2–3 landmark PDFs per core topic.
- Wire question-generation and answer-evaluation prompts into Streamlit actions (e.g. “Generate question” / “Grade my answer”).
- Deploy to Streamlit Community Cloud with secrets for `GROQ_API_KEY`.
- Add hybrid retrieval (BM25 + vector) and optional cross-encoder re-ranking.
- Async ingestion with progress for large PDFs so the UI does not block.

---

## Part 3 Interview Questions

**Question 1:** Walk through backpropagation in a feedforward network and explain why vanishing gradients appear in deep stacks.

Model answer: Forward pass computes predictions layer by layer; loss is backpropagated with the chain rule so each weight gets a gradient. In deep networks, gradients to early layers are products of many local derivatives; if those factors are consistently below 1, the signal shrinks (vanishing), slowing or stopping learning in early layers — as covered in the ANN corpus on backpropagation and activation saturation.

**Question 2:** How does the vanishing gradient problem in RNNs differ from feedforward networks, and why do gated architectures help?

Model answer: RNNs unroll over time, so BPTT multiplies Jacobians across many time steps, often more severely than depth in feedforward nets. LSTMs introduce forget, input, and output gates that create additive paths for gradient flow, allowing selective retention of information over long sequences — connecting RNN and LSTM material.

**Question 3:** Why did you choose a 0.3 similarity threshold and what would you change if users reported “no context” too often?

Model answer: Scores are `1 - cosine_distance` on MiniLM embeddings; 0.3 is a conservative floor to prefer abstention over hallucination. If users see too many no-context responses, I’d log score distributions on real queries, lower the threshold slightly, or improve chunking/embeddings rather than removing the guard entirely.

---

## Project Retrospective

**What clicked:**
- LangGraph made the hallucination guard an explicit branch (`no_context_node`) instead of burying it in prompt text alone.
- Header-first chunking produced readable chunks that match how the study notes are written.
- Content-hash duplicate detection made re-ingestion safe during iterative corpus edits.

**What was hardest:**
- Balancing query rewrite (better recall) against off-topic false positives — solved with original-query validation.
- Streamlit lifecycle (reruns, `@st.cache_resource`, first-load embedding latency).

**What I would study next before a real interview:**
- Evaluation metrics for RAG (precision@k, faithfulness judges, threshold calibration).
- Production patterns: hybrid search, re-ranking, and observability for retrieval scores per query.

---

## Phase 1 checkpoint notes (2026-06-01)

1. **What do I have right now?** Working Streamlit app, Chroma persistence, LangGraph RAG pipeline, three-topic markdown corpus, unit tests for vector store.
2. **What do I still need?** Additional corpus topics, PDF ingestion, UI for interview Q&A prompts, deployment, formal integration test log after full manual run.
3. **What is blocking me?** Nothing critical for local demo; expanding corpus is the main quality lever.

---

## Risk assessment

| Risk | Mitigation |
|---|---|
| Incomplete corpus (missing LSTM/Seq2Seq files) | Demo queries scoped to ANN/CNN/RNN; extend `data/corpus/` before claiming multi-topic coverage |
| Rewrite / retrieval false positives | Original-query validation in `retrieval_node`; abstain via `no_context_node` |
| Groq API unavailable at demo time | Document Ollama fallback in `.env`; keep recorded walkthrough as backup |
