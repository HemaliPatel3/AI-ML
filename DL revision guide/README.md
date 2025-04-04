# RAG (Retrieval-Augmented Generation) Cheat Sheet

## Core Concepts

### What is RAG?
- Combines **retrieval** of external knowledge with **generative** AI
- Solves: hallucinations, knowledge cutoff, specialized information needs
- Flow: Query → Retrieve relevant docs → Generate response using retrieved context

### Key Components
1. **Document Corpus** - Knowledge base of text documents
2. **Embedding Model** - Converts text to vector representations
3. **Vector Database** - Stores and searches embeddings efficiently
4. **Retriever** - Finds relevant documents for a query
5. **Generator** - LLM that produces responses using retrieved context
6. **Orchestrator** - Coordinates the entire pipeline

## Technical Foundation

### Embeddings
- Dense vector representations of text capturing semantic meaning
- Similar meanings have similar vector representations
- Common models: OpenAI embeddings, Sentence-BERT, E5, BGE embeddings

### Vector Search
- **Similarity metrics**: Cosine similarity, dot product, Euclidean distance
- **ANN algorithms**: Approximate Nearest Neighbors for efficient search
- **Vector databases**: Pinecone, Weaviate, Chroma, FAISS, Milvus, Qdrant

### Retrieval Approaches
- **Dense retrieval**: Vector similarity (semantic search)
- **Sparse retrieval**: Keyword-based (BM25, TF-IDF)
- **Hybrid retrieval**: Combines dense and sparse approaches
- **Re-ranking**: Second-pass evaluation to improve relevance

## Implementation Essentials

### Document Processing
- **Chunking**: Splitting documents into manageable pieces
  - Too large → irrelevant info
  - Too small → lost context
  - Sweet spot: ~512-1024 tokens with overlap
- **Metadata**: Adding searchable attributes (date, author, category)

### Prompt Engineering for RAG
```
Use ONLY the following context to answer the question. 
If you cannot find the answer in the context, say "I don't have 
information about this." Don't use prior knowledge.

CONTEXT:
{retrieved_documents}

QUESTION: {user_question}
```

### Evaluation Metrics
- **Retrieval quality**: Precision, recall, MRR, NDCG
- **Response quality**: Factual accuracy, relevance, completeness
- **Citation accuracy**: Do citations actually support the claims?
- **System performance**: Latency, throughput, cost

## Advanced Techniques

### Improving Retrieval
- **Query transformation**: Rewriting/expansion for better search
- **Hypothetical Document Embedding (HyDE)**: Generate ideal answer, then retrieve
- **Self-query**: System reformulates original query for better results
- **Recursive retrieval**: Multiple rounds for complex questions

### Handling Context Limitations
- **Document ranking**: Prioritizing most relevant information
- **Summarization**: Condensing retrieved documents
- **Compression**: Removing redundant or irrelevant parts
- **Chunking optimization**: Semantic vs. fixed-length chunking

### Reducing Hallucinations
- Explicit citation requirements in prompts
- Factuality checking of generated content
- Confidence thresholds for retrieval relevance
- Teaching models to say "I don't know" when appropriate

## Common Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Irrelevant retrieval | Improve embeddings, chunking, reranking |
| Slow response time | Caching, tiered retrieval, async processing |
| Hallucinations | Stricter prompting, citation requirements |
| Outdated info | Regular reindexing, timestamp filtering |
| Limited context window | Better ranking, summarization, compression |

## Popular RAG Frameworks & Tools

- **LangChain**: End-to-end framework for LLM applications
- **LlamaIndex**: Data framework for LLM context augmentation
- **Haystack**: Open-source framework for building search pipelines
- **Semantic Kernel**: Microsoft's orchestration framework
- **Vector DBs**: Pinecone, Weaviate, Chroma, Milvus, Qdrant, pgvector

## Common Use Cases

- **Enterprise knowledge bases**: Internal documentation, policies
- **Customer support**: Self-service, agent assistance
- **Legal/compliance**: Contract analysis, regulatory guidance
- **Research assistance**: Literature review, insight generation
- **Personalized learning**: Educational content delivery
- **Product documentation**: Technical support, tutorials

## Implementation Workflow

1. **Data preparation**: Collect, clean, format source documents
2. **Document processing**: Chunk, embed, and index content
3. **Retrieval setup**: Configure search parameters and strategies
4. **LLM integration**: Design effective prompts with retrieved context
5. **Evaluation**: Test with representative queries, measure performance
6. **Optimization**: Refine based on results, implement advanced techniques
7. **Deployment & monitoring**: Track usage patterns and edge cases
