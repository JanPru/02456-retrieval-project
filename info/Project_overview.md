### Idea
- We have a question and thousands of text passages. The goal is to find the passages that answer the question.
- Two ways to do it
	- Lexical (TF-IDF, BM25)
		- We look for the words shared between the question and the document. If the question says "medical images" and the passage says "imatges mèdiques", it no longer works.
	- Dense (DPR)
		- Two transformer encoders turn the question and the passage into a d-dimensional vector each.
		- Then we take the dot product between the two vectors. Since the network learns the vectors, it can relate texts that don't share any words directly.
- Then we evaluate it with several metrics and also visualize the attention maps

### How does it connect to the course notes?
- Exercises 6 --> Applications section, where DPR appears as a "fine-tuned BERT that turns the question and the document into a vector and compares them with a dot product"
	- The project is about actually building this
- Lecture 6 --> Encoder
	- All the notes on transformer layers and so on.
- Lecture 2 --> The loss, which in our case will probably be classification, so cross entropy
	- Answers or doesn't answer --> that's softmax

### What isn't in the course notes?
- TF-IDF and BM25
	- We have the formulas in the project description
- Contrastive learning
- Retrieval metrics
	- We have never covered evaluation metrics
- Vector databases and nearest neighbour search
	- We saw a bit of it in Exercises 6

### What's in the repo?
- The base code, configured with Hydra
	- Almost everything important is marked with TODO or NotImplementedError

| File                                    | What's there                                                                                                                  | What you have to do                                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `src/data_utils/corpus.py`              | Downloads FiQA or SQuAD from HuggingFace, assigns ids `q0…`/`doc0…` and stores which documents are relevant for each query | Nothing (already works for FiQA)                                                                                               |
| `document.py`, `query.py`, `dataset.py` | Simple container classes                                                                                                      | Nothing                                                                                                                        |
| `retrievers/base_retriever.py`          | Abstract class. `Lookup()` calls `CalculateScores()`, sorts and returns the top-k                                             | Nothing. It's slow (sorts every document in Python for each query), and with DPR you'll want to replace it with a vector index |
| `retrievers/tf_idf.py`                  | The skeleton: a sparse documents × vocabulary matrix, and score = query vector · matrixᵀ                                      | Preprocessing, vocabularies, DF, IDF, term counts, filling the TF-IDF matrix and `QueryToVector`                               |
| `retrievers/bm25.py`                    | Inherits from TFIDF. Already computes document lengths and the average D_avg                                                  | Build `self.bm25_matrix` with the saturated TF and the new IDF                                                                 |
| `retrievers/models/encoder.py`          | Skeleton of `TransformerEncoder` with `cls`/`max`/`mean` pooling                                                              | Everything: the encoder and the three pooling variants                                                                         |
| `retrievers/dpr.py`                     | Creates two encoders, one for queries and one for documents (f_θ and g_φ)                                                     | Finish it: index documents, encode queries and compute scores                                                                  |
| `training/loss.py`                      | Empty `DPRLoss`                                                                                                               | The loss described above                                                                                                       |
| `training/trainer.py`                   | Empty                                                                                                                         | The training loop                                                                                                              |
| `evaluation/metrics.py`                 | Only recall                                                                                                                   | Precision@k and MRR                                                                                                            |
| `config/`                               | Dataset, retriever and model (`emb_dim` 128, 6 layers, 4 heads, FFN 768)                                                      | Tune it                                                                                                                        |
- FiQA or SQuAD --> These are the datasets; they contain the passages and the questions
