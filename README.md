What this project is

This project is a Retrieval-Augmented Generation (RAG) system built over a small set of public health documents related to influenza vaccination guidance.

The system does not train a machine learning model.
Instead, it retrieves relevant text from documents and uses that text to produce an answer.

⸻

What problem this solves

Large language models can hallucinate answers.

This project avoids that by:
	•	Searching official documents first
	•	Answering only using retrieved content

So every answer is grounded in the source text.

⸻

High-level flow (in simple terms)
	1.	Text documents are loaded
	2.	Documents are split into smaller chunks
	3.	A search index (BM25) is built on those chunks
	4.	A user asks a question
	5.	The system finds the most relevant chunks
	6.	The answer is generated from those chunks
	7.	The retrieved text is shown for transparency

⸻

What “Retrieval-Augmented Generation” means here

Retrieval
→ Find the most relevant document chunks for a query

Augmented
→ Provide those chunks as context

Generation
→ Produce an answer using only that context

No fine-tuning, no retraining, no parameter updates.

⸻

How documents are handled
	•	Input documents are plain .txt files
	•	Each document is split into chunks (~700 characters)
	•	Overlap is used so context is not lost
	•	Each chunk keeps metadata (document ID, position, title)

⸻

Retrieval method used

BM25 (sparse retrieval)
	•	BM25 is a classic information retrieval algorithm
	•	It scores chunks based on keyword relevance
	•	Works well for structured and factual text
	•	Fast and explainable

Why BM25 was chosen
	•	Simple
	•	Transparent
	•	Easy to debug
	•	Sufficient for small public health corpora

(Dense / embedding-based retrieval was intentionally disabled to keep the system lightweight and stable.)

⸻

What happens at query time
	1.	User enters a question
	2.	Query is tokenized
	3.	BM25 scores all chunks
	4.	Top-K chunks are selected
	5.	Answer is constructed using retrieved chunks
	6.	Retrieved sources are displayed

⸻

What the system returns
	•	A natural language answer
	•	The text chunks used to form the answer
	•	Source document references

This makes the system auditable and explainable.

⸻

What this project does NOT do
	•	Does not train a neural network
	•	Does not fine-tune a language model
	•	Does not call OpenAI or external APIs
	•	Does not hallucinate outside the corpus

⸻

Can more data be added?

Yes.

To add more knowledge:
	1.	Add new .txt files to data/sample_corpus/
	2.	Re-run indexing (make index)
	3.	Restart the app

No code changes needed.

⸻

Why this is suitable for an academic project
	•	Demonstrates NLP concepts clearly
	•	Shows retrieval vs generation separation
	•	Avoids black-box models
	•	Easy to explain and justify
	•	Reproducible on any machine

⸻

Current status
	•	Indexing works
	•	Retrieval works
	•	UI works
	•	End-to-end pipeline complete
	•	Ready for presentation and report

⸻

“We implemented a RAG pipeline using BM25-based sparse retrieval over chunked public health documents. Queries retrieve top-K relevant chunks, which are then used to generate grounded answers without model training or external APIs.”