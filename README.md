# Deep Learning RAG Interview Prep Agent

A RAG-powered interview preparation agent built with LangChain, LangGraph,
and ChromaDB. Ingest deep learning study material and chat with it to
generate and answer technical interview questions.

---

## Overview

You are building this system end to end as an individual project. The
guide is organized into work areas — corpus, pipeline, UX, prompts,
and QA — each owning a critical piece of the system. The Corpus
Architect workstream is the foundation: without quality study material,
the RAG pipeline has nothing to retrieve. The Pipeline Engineer
workstream is the backbone: ChromaDB, LangChain, and LangGraph wired
into a working system. The UX Lead workstream is the face: the
interface and demo narrative you will present. The Prompt Engineer
workstream is the voice: how the agent thinks, asks questions, and
evaluates answers. The QA Lead workstream is the conscience: stress
testing, failure cases, and quality assurance. No area stands alone —
each depends on the others, and you carry all of them. Treat this as a
solo system design interview and portfolio build. Here is the breakdown:

### Part 1 — Align and Launch 

The goal of Part 1 is to understand the full system before you write
production code.

**Scope the work**
Decide the order you will tackle each work area and what “done” means
for Part 1.

**Sketch the System**
Before heavy implementation, complete `docs/architecture.md`.
Finish it before Part 2 begins.

**Good Chunk vs Bad Chunk Demo**
Understand what a well-formed chunk looks like versus
a poorly formed one. This directly determines retrieval quality. See
`examples/sample_chunk.json` for the canonical reference.

**Parallel Workstream Kickoff**
Start the areas that can move in parallel. Begin drafting corpus content
while you run environment setup and skim the UI and prompt stubs.

---

### Part 2 - The Build 

The goal of Part 2 is a working end-to-end system. Each work area has a
clear milestone and definition of done.

Part 2 is split into two phases with a mandatory checkpoint in the
middle.

**Phase 1 — Parallel Build**
Work toward each Phase 1 milestone: draft core corpus topics,
implement backend stubs in order, build the static three-panel layout,
manually test all prompts, and write the test plan plus demo questions.

**Progress Checkpoint**
Pause and answer three questions: what do I have, what do I still need,
what is blocking me.

**Phase 2 — Integration and Hardening**
Integrate everything: ingest and test the corpus, wire the UI to the
backend, connect prompts inside LangGraph nodes, and run the full test
plan.

**Demo Rehearsal**
One full end-to-end run-through before Part 3. 

See `docs/checklist.md` for the full per-area breakdown of both phases.

---

### Part 3 — Presentations

The goal of Part 3 is to present what you built under interview
conditions — as if walking an interviewer through a technical screen.

**Demo**
Use a live demo structured like a technical interview. No slides — the
running application is your presentation. Show ingestion, duplicate
detection, a successful query with source citation, and the
hallucination guard firing on an off-topic question. For async or
recorded submissions, a 5-minute video walkthrough of the same flow is
fine.

---

### The Thread That Connects All Three Parts

Every decision you make in Part 1 affects what you can build in Part 2.
Every decision you make in Part 2 affects what you can defend in Part 3.
The goal is to explain everything you built and why.

---

## 👋 Navigate the Guide

Use the table below to jump to the work area you are focusing on.
You will touch every section over the course of the project.

| Role | You Own | Go To |
|---|---|---|
| **Corpus Architect** | `data/corpus/` | [→ Corpus Architect](#corpus-architect) |
| **Pipeline Engineer** | `config.py`, `store.py`, `nodes.py`, `graph.py` | [→ Pipeline Engineer](#pipeline-engineer) |
| **UX Lead** | `ui/app.py` | [→ UX Lead](#ux-lead) |
| **Prompt Engineer** | `prompts.py` | [→ Prompt Engineer](#prompt-engineer) |
| **QA Lead** | `tests/`, demo script | [→ QA Lead](#qa-lead) |

---

## Corpus Architect

You can start immediately — in parallel with environment setup — while
the pipeline workstream gets the repo and dependencies ready.

### Your First Task — Draft 3 Topics

Open your preferred editor and begin drafting study material for the
first three core topics. Do not wait for the backend to be ready.
Your content is the foundation everything else depends on.

**Start with these three topics in this order:**
1. Artificial Neural Networks (ANN)
2. Convolutional Neural Networks (CNN)
3. Recurrent Neural Networks (RNN)

**Before writing a single chunk, agree on the metadata schema
with the Pipeline Engineer.** Every chunk must follow this structure:

```json
{
  "chunk_text": "...",
  "metadata": {
    "topic": "ANN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "ann.md",
    "related_topics": ["backpropagation", "activation_functions"],
    "is_bonus": false
  }
}
```

See `examples/sample_chunk.json` for the full schema with validation
rules and a quality checklist.

**Chunk writing rules:**
- One atomic idea per chunk — if it could answer five interview questions, split it
- Minimum 100 words, maximum 300 words per chunk
- If you remove the topic name from the chunk, the content should still
  identify the topic — if not, it is too generic
- Each chunk should stand alone as the basis for exactly one interview question

**File naming convention:**
```
data/corpus/ann_intermediate.md
data/corpus/cnn_intermediate.md
data/corpus/rnn_intermediate.md
```

**An Example of One of the Markdown Files**
```
# Long Short-Term Memory (LSTM)

## The Vanishing Gradient Problem
Long Short-Term Memory networks (LSTMs) solve the vanishing gradient 
problem that affects standard RNNs. They do this through three learned 
gates...

## The Forget Gate
The forget gate decides what information to discard from the cell state.
It takes the previous hidden state and current input, passes them through
a sigmoid function...

## The Input Gate
The input gate decides what new information to store in the cell state.
It consists of two parts: a sigmoid layer that decides which values to
update...
```

**Landmark papers — locate these while you draft:**

| Topic | Paper |
|---|---|
| ANN / Backprop | Rumelhart, Hinton & Williams (1986) |
| CNN | LeCun et al. (1998) LeNet |
| CNN deep | Krizhevsky et al. (2012) AlexNet |
| RNN | Elman (1990) |
| LSTM | Hochreiter & Schmidhuber (1997) |
| Seq2Seq | Sutskever, Vinyals & Le (2014) |
| Autoencoder | Hinton & Salakhutdinov (2006) |
| GAN *(bonus)* | Goodfellow et al. (2014) |
| Boltzmann *(bonus)* | Hinton & Sejnowski (1986) |
| SOM *(bonus)* | Kohonen (1982) |

Place downloaded PDFs in `data/corpus/`.

**Your Phase 1 milestone:**
- [ ] 3 topics drafted with at least 3 chunks each
- [ ] Metadata schema agreed with Pipeline Engineer
- [ ] At least one landmark paper PDF located per topic

---

## Pipeline Engineer

You own environment setup and the core backend. Document decisions as
you go — chunking, embeddings, retrieval, and graph flow are core
system design talking points.

### Step 1 — Clone the Repo

```bash
git clone https://github.com/kandulapatimonicavalli/deep-learning-rag-agent.git
cd deep-learning-rag-agent
```

### Step 2 — Install UV

UV is a fast modern Python package manager. Install it first if you
do not have it.

**Mac / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

### Step 3 — Create Virtual Environment and Install Dependencies

```bash
uv sync
```

This reads `pyproject.toml` and installs all dependencies into a
local `.venv` folder. This is the only command needed — no separate
`pip install` required.

### Step 4 — Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in your LLM provider credentials.
See the [LLM Provider Setup](#llm-provider-setup) section below.

### Step 5 — Verify Setup

```bash
uv run python -c "import chromadb; import langchain; import langgraph; print('All dependencies OK')"
```

If this passes, environment setup is complete.

### Step 6 — Run the App

```bash
uv run streamlit run src/rag_agent/ui/app.py
```

If you see a `NotImplementedError`, a stub is still unimplemented —
follow the stack trace and implement the referenced method. The core
pipeline (`store.py`, `nodes.py`, `graph.py`, `app.py`) is already
implemented in this repository.

**Implementation order:**
1. `EmbeddingFactory._create_local()` in `config.py`
2. `VectorStoreManager._initialise()` in `store.py`
3. `VectorStoreManager.check_duplicate()` in `store.py`
4. `VectorStoreManager.ingest()` in `store.py`
5. `VectorStoreManager.query()` in `store.py`
6. LangGraph nodes in `nodes.py`
7. Graph assembly in `graph.py`

**Your Phase 1 milestone:**
- [ ] `uv sync` completes without errors
- [ ] `.env` configured with working LLM provider
- [ ] `EmbeddingFactory._create_local()` implemented
- [ ] `VectorStoreManager._initialise()` implemented
- [ ] `VectorStoreManager.check_duplicate()` implemented
- [ ] Hello world retrieval returning results

---

## UX Lead

While environment setup runs, read the backend contracts. You do not
need to understand every line of backend code — but you need to
understand what it returns.

### While Watching Setup — Think About These

**On the API contract:**
The methods you will call from `ui/app.py` are already defined
in the codebase. Open these two files now and read them while
setup is running:

- `src/rag_agent/agent/state.py` — all the data models you will
  receive back: `IngestionResult`, `RetrievedChunk`, `AgentResponse`
- `src/rag_agent/vectorstore/store.py` — the methods you will call:
  `ingest()`, `list_documents()`, `get_document_chunks()`, `query()`

You are not waiting for the Pipeline Engineer to define these —
they are already there. What you do need to coordinate with the
Pipeline Engineer is:

- Confirm method signatures have not changed as they implement the stubs
- Agree on how errors are surfaced — does `ingest()` raise an exception
  or return errors inside `IngestionResult`? The answer is in `state.py`
  but make sure you both read it the same way

**On your framework choice:**
Decide now — Streamlit or Gradio. Do not switch mid-sprint.
- **Streamlit** → deploy to [Streamlit Community Cloud](https://share.streamlit.io)
- **Gradio** → deploy to [HuggingFace Spaces](https://huggingface.co/spaces)

**On the three panels you need to build:**
1. Document ingestion — multi-file upload, status display, document list
2. Document viewer — selectable documents, content display, chunk viewer
3. Chat interface — history display, query input, source citations

**On session state:**
Streamlit reruns the entire script on every user interaction.
Everything that needs to persist (chat history, ingested documents,
VectorStoreManager instance) must live in `st.session_state`.
During setup, ask the Pipeline Engineer to walk through `app.py`
line by line — the `initialise_session_state()` function is your
starting point.

**Your Phase 1 milestone:**
- [ ] Framework chosen and recorded in `docs/architecture.md`
- [ ] Data models in `state.py` read and understood
- [ ] Static three-panel layout running locally with placeholder content
- [ ] `st.session_state` keys initialised correctly

---

## Prompt Engineer

While environment setup runs, study the data flow so you can write
prompts that work with real retrieved chunks.

### While Watching Setup — Think About These

**On the system prompt:**
Open `src/rag_agent/agent/prompts.py` now. Read `SYSTEM_PROMPT`
carefully. Ask yourself:
- Does this persona match what we want the agent to do?
- Are the constraints strict enough to prevent hallucination?
- Is the citation instruction clear enough to produce consistent output?

**On the question generation prompt:**
The `QUESTION_GENERATION_PROMPT` asks the model to return JSON.
Think about failure modes — what happens if the model returns
malformed JSON? Talk to the Pipeline Engineer about how to handle
parsing errors gracefully.

**On testing prompts before integration:**
You do not need the backend to be running to test prompts. Open
Claude, ChatGPT, or your chosen LLM in a browser and manually test
each prompt right now while setup is running. Iterate before you
integrate.

There are three prompts in `src/rag_agent/agent/prompts.py` that
you own. Here is what each one does and how to test it manually:

**Prompt 1 — System Prompt (`SYSTEM_PROMPT`)**
This defines the agent's identity, constraints, and behavior. It
instructs the model to answer only from retrieved context, always
cite sources, and indicate when no relevant content is found.

To test: paste `SYSTEM_PROMPT` into your LLM of choice as the
first message, then paste the sample chunk from
`examples/sample_chunk.json` as context, then ask a question about
LSTMs. Verify the model cites the source and does not go beyond
what the chunk says.

Example test input:
```
[System Prompt here]

Context:
[paste sample_chunk.json chunk_text here]

Question: What problem do LSTMs solve and how?
```

Expected behavior: the model answers using only the chunk content
and cites `[SOURCE: LSTM | lstm.md]`. If it adds information not
in the chunk, tighten the system prompt constraints.

---

**Prompt 2 — Question Generation (`QUESTION_GENERATION_PROMPT`)**
Given a retrieved chunk, this prompt generates one interview question
at a specified difficulty level and returns a structured JSON response
with the question, model answer, and a follow-up question.

To test: paste the prompt into your LLM, substitute the sample chunk
for `{context}` and `intermediate` for `{difficulty}`. Verify the
output is valid JSON and the question requires genuine understanding,
not just recall.

Example test input:
```
[QUESTION_GENERATION_PROMPT with placeholders filled in]

{context}: Long Short-Term Memory networks (LSTMs) solve the 
vanishing gradient problem through three learned gates: the forget 
gate, input gate, and output gate...

{difficulty}: intermediate
```

Expected output:
```json
{
  "question": "Explain how the three gates in an LSTM work together
               to solve the vanishing gradient problem",
  "difficulty": "intermediate",
  "topic": "LSTM",
  "model_answer": "...",
  "follow_up": "How would you decide between using an LSTM versus
                a standard RNN for a given task?",
  "source_citations": ["[SOURCE: LSTM | lstm.md]"]
}
```

If the model returns malformed JSON or a trivial yes/no question,
revise the prompt and test again before handing to the Pipeline
Engineer for integration.

---

**Prompt 3 — Answer Evaluation (`ANSWER_EVALUATION_PROMPT`)**
Given a question, a student's answer, and the source chunk, this
prompt evaluates the answer and returns a score out of 10 with
detailed feedback.

To test: paste the prompt with a question from your Prompt 2 test,
write a deliberately incomplete answer, and verify the model
correctly identifies what is missing.

Example test input:
```
[ANSWER_EVALUATION_PROMPT with placeholders filled in]

{question}: Explain how the three gates in an LSTM work together
            to solve the vanishing gradient problem

{candidate_answer}: LSTMs have gates that control what information
                    is kept or forgotten, which helps with long
                    sequences.

{context}: [paste sample chunk here]
```

Expected output: a score around 4-5 out of 10, correctly identifying
that the candidate understood the general concept but did not name
or explain the three gates specifically. If the score seems too
generous or too harsh, adjust the scoring guide in the prompt.

---

The key thing to watch for across all three prompts is
**JSON reliability** — the Pipeline Engineer will be parsing
these responses programmatically. If the model occasionally
returns prose instead of JSON, add this line to the affected
prompts: `Respond with the JSON object only. No preamble,
explanation, or markdown code fences.`

**On difficulty levels:**
How will the system know what difficulty to generate questions at?
Where does that instruction come from — the user, the metadata,
or a default? Decide this with the UX Lead during setup.

**Your Phase 1 milestone:**
- [ ] All three prompts read and understood
- [ ] System prompt reviewed and any changes noted
- [ ] Question generation prompt manually tested against sample chunk
- [ ] Answer evaluation prompt manually tested
- [ ] Failure modes documented for each prompt

---

## QA Lead

While environment setup runs, learn the system well enough to break it
in Phase 2.

### While Watching Setup — Think About These

**On the test plan:**
Read `tests/test_vectorstore.py` now. These are the unit tests
the Pipeline Engineer needs to make pass. Your job in Phase 2
is to run five integration test cases against the full system.
Start writing those test cases during setup:

| Test | Input | Expected |
|---|---|---|
| Normal query | "Explain vanishing gradient" | Relevant chunks, source cited |
| Off-topic query | "History of Rome" | No context found message |
| Duplicate ingestion | Upload same file twice | Second upload skipped |
| Empty query | Blank input | Graceful error, no crash |
| Cross-topic query | "How do LSTMs improve on RNNs" | Multi-topic retrieval |

**On Part 3 questions:**
You are responsible for drafting three technical interview questions
you can use in a demo or portfolio review. Rules:
- Must be answerable from a well-built corpus
- At least one must connect two topics
- Prepare a model answer for each

**Your Phase 1 milestone:**
- [ ] Five integration test cases written
- [ ] Risk assessment completed against rubric
- [ ] Three Part 3 interview questions drafted with model answers
- [ ] Demo script outline started (ingestion → duplicate → query → guard → question)

---

## LLM Provider Setup

### Option 1 — Groq (Recommended: free, fast, no local GPU needed)

1. Create a free account at [console.groq.com](https://console.groq.com)
2. Generate an API key under **API Keys**
3. In `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Available models (free tier): `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`

**Interview talking point:** Groq uses a custom LPU (Language Processing Unit)
chip designed for LLM inference, delivering significantly lower latency
than GPU-based inference APIs.

---

### Option 2 — Ollama (Fully local, no API key, runs offline)

1. Download and install from [ollama.com](https://ollama.com)
2. Pull a model:
```bash
ollama pull llama3.2        # 2B — fast, low memory
ollama pull mistral         # 7B — better quality
ollama pull llama3.1:8b     # 8B — good balance
```
3. Start Ollama before running the app:
```bash
ollama serve
```
4. In `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

**Interview talking point:** Local inference eliminates data privacy
concerns and removes API cost entirely — important for enterprise
deployments with proprietary training data.

---

### Option 3 — LM Studio (Local GUI, OpenAI-compatible API)

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Open LM Studio → **Discover** → download a model
   (recommended: Llama 3.2 3B Instruct or Mistral 7B Instruct)
3. Go to **Local Server** tab → Load model → Start Server
4. In `.env`:
```
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
```

**Interview talking point:** LM Studio exposes an OpenAI-compatible API —
any tooling written for OpenAI works without code changes, just a
`base_url` swap. This is the adapter pattern in practice.

---

## Deployment

### Streamlit Community Cloud (if you chose Streamlit)

1. Push code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select repo, branch, and `src/rag_agent/ui/app.py`
4. Under **Advanced Settings → Secrets**, add your `.env` variables
5. Click **Deploy**

### HuggingFace Spaces (if you chose Gradio, or Streamlit on Spaces)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to **Spaces → Create New Space**
3. Choose **Streamlit** or **Gradio** as the SDK
4. Connect your GitHub repo
5. Add API keys under **Settings → Repository Secrets**

HuggingFace Spaces provides 16GB RAM on the free CPU tier — better
for memory-intensive embedding operations than Streamlit Community Cloud.

---

## Project Structure

```
deep-learning-rag-agent/
├── docs/
│   ├── checklist.md            ← Part 2 guide
│   ├── architecture.md         ← system design doc (filled)
├── data/
│   └── corpus/                 ← Corpus Architect: add .md and .pdf files here
├── examples/
│   └── sample_chunk.json       ← canonical chunk schema reference
├── src/
│   └── rag_agent/
│       ├── config.py           ← Settings, LLMFactory, EmbeddingFactory
│       ├── corpus/
│       │   └── chunker.py      ← DocumentChunker
│       ├── vectorstore/
│       │   └── store.py        ← VectorStoreManager
│       ├── agent/
│       │   ├── state.py        ← AgentState and data models
│       │   ├── prompts.py      ← all LLM prompt templates
│       │   ├── nodes.py        ← LangGraph node functions
│       │   └── graph.py        ← graph assembly
│       └── ui/
│           └── app.py          ← Streamlit application
├── tests/
│   └── test_vectorstore.py     ← unit tests
├── .env.example                ← copy to .env and fill in credentials
├── pyproject.toml              ← UV dependencies
└── README.md
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Submitting Your Work

Publish the finished project to your personal GitHub account when you
are ready to share it.

To publish from VSCode:
1. Open Source Control in the sidebar
2. Click **Publish to GitHub**
3. Choose **Public**
4. Name it something memorable — this goes on your portfolio

A working RAG agent is a strong portfolio piece — document your design
decisions in `docs/architecture.md` so you can discuss them in
interviews.

---

## Presentations

**Live:** Walk through your working demo in person or on a call.

**Recorded:** 5-minute video presentation of your working demo.

---

## Common Issues

**`ModuleNotFoundError: No module named 'rag_agent'`**
Run from the project root using `uv run`, not `python` directly.

**`NotImplementedError`**
Expected — the stubs are waiting for implementation. Follow the
stack trace to find the next method to implement.

**`chromadb.errors.NotEnoughElementsException`**
Your collection has fewer chunks than the requested `k`. Ingest
more content or reduce `RETRIEVAL_K` in `.env`.

**`ollama: connection refused`**
Ollama is not running. Start it with `ollama serve` in a separate terminal.

**Streamlit reruns on every click and loses state**
Wrap persistent objects in `@st.cache_resource`. Store conversation
history and document list in `st.session_state`.

**`data/chroma_db` does not exist**
Create it manually — see Step 5 of Pipeline Engineer setup above.
Once `VectorStoreManager._initialise()` is implemented, it will
create this directory automatically on startup.
