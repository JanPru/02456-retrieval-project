We have 3 techniques, all three doing the same thing: turning the query and the document into vectors and computing their dot product. What changes is where these vectors come from. In TF-IDF and BM25 they are built by hand by counting words. In DPR a network learns them.

### TF-IDF
Notation:
- There are N documents D
- The query Q has m terms
- f_ij is the number of times term qi appears in document Dj

Term Frequency (TF) -> How much the document talks about this word
$$\text{TF}(q_i, D_j) = \frac{f_{ij}}{\max_k f_{kj}}$$
- The numerator counts how many times the word appears in the document
- The denominator is the count of the most frequent word in that same document. This way TF always stays between 0 and 1

Inverse Document Frequency (IDF) -> How informative the word is
$$\text{IDF}(q_i) = \log\left(\frac{N}{n_{q_i}}\right), \qquad n_{q_i} = |\{D \in \mathcal{D} : q_i \in D\}|$$
- nqi is the number of documents the word appears in. It counts once per document
- If a word appears in every document, like "the", the IDF is log(N/N) = 0. It doesn't help to distinguish the document.
- If it appears in one out of 1000 documents, the log gives 6.9: a very rare word is a very informative word

TF-IDF and score
$$\text{TF-IDF}(q_i, D_j) = \text{TF}(q_i, D_j)\cdot\text{IDF}(q_i), \qquad \text{Score}(Q, D) = \sum_{q\in Q}\text{TF-IDF}(q, D)$$
- A word weighs a lot when it appears often in a document and is also rare

How does the repo do it?
- The sum can be written as a dot product
	- Each document is a vocabulary-sized vector with the TF-IDF of each word. Stacked, they form the tf_idf_matrix of size N x V
	- The query is a vector of counts (QueryToVector)
	- scores = query_vectors * tf_idf_matrix
- Coincidentally, this has the same shape as the attention scores


### BM25
BM25 fixes two things in TF-IDF

**Saturated and length-normalized TF**$$\text{TF}(q_i, D_j) = \frac{f_{ij}\,(\kappa + 1)}{f_{ij} + \kappa\left(1 - b + b\cdot\frac{|D_j|}{D_{avg}}\right)}, \qquad D_{avg} = \frac{1}{N}\sum_i |D_i|$$
- We'll call the term in parentheses L
Saturation --> Take a document of average length, so L = 1. Then TF = f(κ+1)/(f+κ), with κ = 1.5, which is the default

| f (occurrences) | 1    | 2    | 5    | 20   | ∞             |
| --------------- | ---- | ---- | ---- | ---- | ------------- |
| BM25 TF         | 1.00 | 1.43 | 1.92 | 2.33 | **2.5 = κ+1** |
- As we can see, the first occurrence counts a lot and each new occurrence counts less. This way a document that repeats a word many times doesn't win by a landslide
	- **κ → 0**: TF = 1 always. Only whether the word is there or not matters (binary).
	- **κ → ∞**: TF ≈ f/L. Linear, no saturation. That's why the project description says that the larger κ is, the more slowly it saturates.

Length normalization (b)
- A long document has |Dj| > D_avg and therefore L > 1. The denominator grows and the TF drops. We penalize it because a long document is more likely to contain the word by chance
- b = 0: L = 1, no normalization
- b = 1: full normalization. By default we use b = 0.75

**Modified IDF**
$$\text{IDF}(q_i) = \log\left(\frac{N + 1}{n_{q_i} + 0.5}\right)$$
- If the word appears in every document, n = N and the IDF has a small but positive value, not 0
- That's why documents that repeat common words are slightly favoured

### Dense passage retrieval
What if, instead of defining by hand how relevance is measured, we learn it?

The two encoders:
$$f_\theta : \mathcal{V} \to \mathbb{R}^d \;\;(\text{documents}), \qquad g_\phi : \mathcal{V} \to \mathbb{R}^d \;\;(\text{queries})$$
- They are two different networks with different parameters
	- They are the document_encoder and the query_encoder

What happens inside f?
$$\mathbf{x}_n = E[\text{token}_n] + \mathbf{PE}_n \;\to\; \underbrace{\text{Transformer layer}}_{\times K} \;\to\; \mathbf{H} \in \mathbb{R}^{T\times d} \;\to\; \text{pool}(\mathbf{H}) \in \mathbb{R}^d$$
- It's from Lecture 6
	- Embedding + Positional Encoding
	- K transformer layers
		- Each layer has 4 parts
			- Multi-head attention + layernorm + FFN + layernorm
	- Pooling
		- The transformer returns one vector per token and we want one vector for the whole text
		- $$\text{cls}: \mathbf{h}_{\text{[CLS]}} \qquad \text{mean}: \frac{1}{T}\sum_{t=1}^{T}\mathbf{h}_t \qquad \text{max}: \big(\max_t h_{t,c}\big)_{c=1}^{d}$$
The similarity$$s(Q_i, D_j) = f_\theta(D_j)^\top g_\phi(Q_i)$$
- It's the dot product between the two dense vectors of 128 (or whatever size) numbers, instead of two sparse vectors with thousands of dimensions

The Loss --> We have a batch of B (D, Q) pairs and for each pair the following loss: $$\ell(\mathcal{B};\theta,\phi) = -\ln\frac{\exp\big(f_\theta(D_i)^\top g_\phi(Q_i)\big)}{\sum_{j=1}^{|\mathcal{B}|}\exp\big(f_\theta(D_j)^\top g_\phi(Q_i)\big)}$$
- We read it in three steps:
	- Score matrix
		- Stack the B encoded queries into G (B x d) and the B documents into F (B x d). Then S = G * Fᵀ, with Sij = similarity of query i with document j. The diagonal holds the correct pairs and the rest are the incorrect pairs
	- Row-wise softmax
		- For query i it's a probability distribution over which of the B documents is mine
	- -ln of the correct probability --> cross-entropy loss
