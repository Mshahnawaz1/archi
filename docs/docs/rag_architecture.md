# RAG Architecture in Archi

## Overview

Archi is fundamentally a **Retrieval-Augmented Generation (RAG) framework** designed for research and educational support. This document provides detailed information about how RAG is implemented and used throughout the system.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Model (LLM) responses by:

1. **Retrieving** relevant documents from a knowledge base based on the user's query
2. **Augmenting** the LLM prompt with the retrieved context
3. **Generating** responses that are grounded in the retrieved information

This approach allows the system to provide accurate, source-backed responses without requiring the LLM to memorize all information during training.

## Archi's RAG Architecture

### High-Level Flow

```
User Query
    ↓
Embedding Model (converts query to vector)
    ↓
Vector Store Search (PostgreSQL + pgvector)
    ↓
Document Retrieval (Semantic/Hybrid/Grading)
    ↓
Context Augmentation (add retrieved docs to prompt)
    ↓
LLM Generation (with grounded context)
    ↓
Response (with source citations)
```

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   Web    │  │   Git    │  │   JIRA   │  │ Redmine  │  │  Local   │     │
│  │  Links   │  │  Repos   │  │ Tickets  │  │ Tickets  │  │   Docs   │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │             │             │             │
│       └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                   │                                           │
│                        ┌──────────▼──────────┐                              │
│                        │     Collectors      │                              │
│                        │  (Text Extraction)  │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                           │
│                        ┌──────────▼──────────┐                              │
│                        │   Text Splitter     │                              │
│                        │ (Chunking/Overlap)  │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                           │
│                        ┌──────────▼──────────┐                              │
│                        │  Embedding Model    │                              │
│                        │ (OpenAI/HuggingFace)│                              │
│                        └──────────┬──────────┘                              │
│                                   │                                           │
│                        ┌──────────▼──────────┐                              │
│                        │ PostgreSQL+pgvector │                              │
│                        │   (Vector Store)    │                              │
│                        └─────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐                                                        │
│  │   User Query     │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │ Query Embedding  │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                   │
│           ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │              Retriever Selection                          │               │
│  ├──────────────────────────────────────────────────────────┤               │
│  │                                                           │               │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │               │
│  │  │  Semantic    │  │   Hybrid     │  │   Grading    │  │               │
│  │  │  Retriever   │  │  Retriever   │  │  Retriever   │  │               │
│  │  │              │  │              │  │              │  │               │
│  │  │ - Pure       │  │ - Vector     │  │ - Specialized│  │               │
│  │  │   Vector     │  │   Similarity │  │   for        │  │               │
│  │  │   Search     │  │ + BM25       │  │   Assessment │  │               │
│  │  │ - Conceptual │  │ - Balanced   │  │              │  │               │
│  │  │   Queries    │  │   Approach   │  │              │  │               │
│  │  │              │  │              │  │              │  │               │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │               │
│  │         │                 │                 │           │               │
│  └─────────┴─────────────────┴─────────────────┴───────────┘               │
│                              │                                               │
│                              ▼                                               │
│                  ┌───────────────────────┐                                  │
│                  │ Top-K Documents       │                                  │
│                  │ (with scores/metadata)│                                  │
│                  └───────────┬───────────┘                                  │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────────┐
│                        GENERATION PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │            Context Formatting                             │               │
│  │  - Retrieved documents → formatted context                │               │
│  │  - Source citations                                       │               │
│  │  - Metadata inclusion                                     │               │
│  └──────────────────┬───────────────────────────────────────┘               │
│                     │                                                         │
│                     ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │            Prompt Construction                            │               │
│  │  - System prompt + Instructions                           │               │
│  │  - Retrieved context                                      │               │
│  │  - User query                                             │               │
│  └──────────────────┬───────────────────────────────────────┘               │
│                     │                                                         │
│                     ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │                 LLM Generation                            │               │
│  │  (OpenAI, Anthropic, Local Models, etc.)                 │               │
│  └──────────────────┬───────────────────────────────────────┘               │
│                     │                                                         │
│                     ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │            Response with Citations                        │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION & MONITORING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │                 RAGAS Evaluation                          │               │
│  │  - Answer Relevancy                                       │               │
│  │  - Faithfulness                                           │               │
│  │  - Context Precision                                      │               │
│  │  - Context Relevancy                                      │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │              Performance Metrics                          │               │
│  │  - Retrieval latency                                      │               │
│  │  - Embedding generation time                              │               │
│  │  - LLM response time                                      │               │
│  │  - Source accuracy                                        │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core RAG Components

### 1. Vector Store (PostgreSQL + pgvector)

**Location**: `/src/data_manager/vectorstore/postgres_vectorstore.py`

Archi uses PostgreSQL with the `pgvector` extension as its vector database, replacing the previous ChromaDB implementation. This provides:

- **Native SQL integration** for efficient queries
- **Multiple distance metrics**: cosine similarity, L2 distance, inner product
- **HNSW indexing** for fast approximate nearest neighbor search
- **Hybrid search capabilities** (combining vector + full-text search)

**Key Features**:
```python
class PostgresVectorStore(VectorStore):
    """
    Vector store implementation using PostgreSQL with pgvector extension.
    
    Features:
    - LangChain VectorStore interface compatibility
    - Cosine similarity search via pgvector
    - Optional hybrid search (semantic + BM25 full-text)
    - HNSW index support for fast approximate nearest neighbor
    """
```

**Distance Metrics**:
- `cosine`: Cosine distance (1 - cosine_similarity) - **default**
- `l2`: Euclidean distance
- `inner_product`: Negative inner product

**Configuration Example**:
```yaml
data_manager:
  distance_metric: cosine  # or l2, inner_product
```

### 2. Embedding Models

**Location**: `/src/data_manager/collectors/utils/embedding_utils.py`

Archi supports multiple embedding model providers for converting text into vector representations:

#### OpenAI Embeddings
- **text-embedding-3-small** (default)
- **text-embedding-ada-002**

#### HuggingFace Embeddings
- **all-MiniLM-L6-v2** (lightweight, suitable for local deployment)
- Other HuggingFace models via custom configuration

**Configuration Example**:
```yaml
data_manager:
  embedding_name: OpenAIEmbeddings  # or HuggingFaceEmbeddings
  embedding_params:
    model: text-embedding-3-small   # OpenAI
    # OR
    model_name: all-MiniLM-L6-v2    # HuggingFace
```

**Instruction-Aware Embeddings**:

Some models support instruction-based queries, which can improve retrieval accuracy:
```python
# Models with instruction support
INSTRUCTION_AWARE_MODELS = [
    "instructor-large",
    "instructor-xl",
    # ... others
]
```

### 3. Document Retrieval Strategies

Archi implements three distinct retrieval strategies, each optimized for different use cases:

#### 3.1 Semantic Retriever

**Location**: `/src/data_manager/vectorstore/retrievers/semantic_retriever.py`

Pure vector similarity search using embeddings.

**How it works**:
1. Converts query to embedding vector
2. Performs similarity search in vector store
3. Returns top-k most similar documents

**Configuration**:
```yaml
data_manager:
  retrievers:
    semantic_retriever:
      num_documents_to_retrieve: 5
      instructions: "Represent this query for retrieving relevant documents:"
```

**Use Cases**:
- Conceptual questions
- When keyword matching is insufficient
- When you need semantic understanding

**Code Example**:
```python
class SemanticRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Optional: add instructions for instruction-aware models
        if self.instructions and supported:
            query = make_instruction_query(self.instructions, query)
        
        # Perform similarity search
        similarity_result = self.vectorstore.similarity_search_with_score(
            query, k=self.k
        )
        return similarity_result
```

#### 3.2 Hybrid Retriever

**Location**: `/src/data_manager/vectorstore/retrievers/hybrid_retriever.py`

Combines semantic (vector) search with BM25 full-text search for improved retrieval accuracy.

**How it works**:
1. Performs semantic search using embeddings
2. Performs BM25 keyword search using PostgreSQL full-text search
3. Combines scores with configurable weights
4. Returns top-k results by combined score

**Configuration**:
```yaml
data_manager:
  retrievers:
    hybrid_retriever:
      num_documents_to_retrieve: 5
      bm25_weight: 0.6          # Weight for keyword matching
      semantic_weight: 0.4       # Weight for semantic similarity
```

**Use Cases**:
- Technical documentation with specific terms
- When both exact keywords and semantic meaning matter
- Improved recall across different query types

**Code Example**:
```python
class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Tuple[Document, float]]:
        # Use Postgres-native hybrid search
        results = self.vectorstore.hybrid_search(
            query=query,
            k=self.k,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
        )
        return results
```

**Weight Tuning**:
- Higher `bm25_weight`: Better for keyword-heavy queries
- Higher `semantic_weight`: Better for conceptual queries
- Start with 0.5/0.5 and adjust based on evaluation

#### 3.3 Grading Retriever

**Location**: `/src/data_manager/vectorstore/retrievers/grading_retriever.py`

Specialized retriever for automated grading tasks.

**Use Cases**:
- Course assignment grading
- Comparing student submissions against rubrics
- Finding relevant examples or reference solutions

### 4. Document Chunking

Archi splits long documents into smaller chunks for more precise retrieval:

**Configuration**:
```yaml
data_manager:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks (preserves context)
```

**Why Chunking?**:
- Improves retrieval precision
- Allows embedding models to capture local context
- Enables multiple retrievals from the same document

**Text Splitters**:
- `RecursiveCharacterTextSplitter` (default) - respects paragraph/sentence boundaries
- Custom splitters for code or structured documents

### 5. Agent Tools for RAG

**Location**: `/src/archi/pipelines/agents/tools/retriever.py`

Archi exposes retrieval capabilities as LangChain tools for agent workflows:

**Available Tools**:
1. **Metadata Search**: Find files by name/path/source
2. **Content Search (grep)**: Line-level regex search
3. **Document Fetch**: Retrieve full text by hash
4. **Vectorstore Search**: Semantic retrieval

**Tool Integration**:
```python
from langchain.tools import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=semantic_retriever,
    name="vectorstore_search",
    description="Search for relevant documents using semantic similarity",
)
```

## Data Ingestion Pipeline

### Supported Data Sources

Archi can ingest documents from multiple sources:

1. **Web Links**: Scrape URLs (with SSO support via Selenium)
2. **Git Repositories**: Clone and parse MKDocs repositories
3. **JIRA Tickets**: Import issue descriptions and comments
4. **Redmine Tickets**: Import tickets and attachments
5. **Piazza Posts**: Import Q&A discussions
6. **Local Documents**: PDF, DOCX, TXT, markdown files

### Ingestion Process

```
Source → Collector → Text Extraction → Chunking → Embedding → Vector Store
```

**Configuration Example**:
```yaml
data_manager:
  sources:
    links:
      input_lists:
        - documentation.list
    git:
      repositories:
        - url: https://github.com/example/docs
          branch: main
    local:
      directory: /path/to/documents
```

**Metadata Tracking**:

Each document chunk stores metadata:
- `filename`: Original file name
- `source`: Data source type (link, git, jira, etc.)
- `resource_hash`: Unique identifier for the source document
- `collection`: Logical grouping
- Custom metadata from collectors

## RAG in Different Pipelines

### Classic RAG Pipeline

Traditional RAG flow for chat and Q&A:

1. User submits query
2. Query is embedded
3. Semantic/hybrid retrieval fetches top-k documents
4. Retrieved documents are formatted into context
5. LLM generates response with context

**Prompt Template**:
```python
"""
Answer the following question based on the provided context.

Context:
{retriever_output}

Question: {question}

Answer:
"""
```

### Agentic RAG Pipeline

Multi-step RAG with tool use:

1. Agent receives query
2. Agent decides which tools to use (search, retrieve, fetch)
3. Agent calls tools multiple times if needed
4. Agent synthesizes final response

**Benefits**:
- More flexible retrieval strategies
- Can combine multiple information sources
- Self-correcting (can retrieve again if initial results insufficient)

### Grading Pipeline

RAG for automated assessment:

1. Student submission ingested
2. Retrieve relevant rubric items and examples
3. Compare submission against rubric
4. Generate feedback and score

## RAG Configuration

### Complete Configuration Example

```yaml
data_manager:
  # Embedding configuration
  embedding_name: OpenAIEmbeddings
  embedding_params:
    model: text-embedding-3-small
  
  # Vector store configuration
  distance_metric: cosine
  
  # Chunking configuration
  chunk_size: 1000
  chunk_overlap: 200
  
  # Stemming (for BM25 search)
  stemming:
    enabled: true
    language: english
  
  # Retriever configurations
  retrievers:
    semantic_retriever:
      num_documents_to_retrieve: 5
      instructions: "Represent this query for retrieving relevant documents:"
    
    hybrid_retriever:
      num_documents_to_retrieve: 5
      bm25_weight: 0.6
      semantic_weight: 0.4
    
    grading_retriever:
      num_documents_to_retrieve: 10
  
  # Data sources
  sources:
    links:
      input_lists:
        - documentation.list
    git:
      repositories:
        - url: https://github.com/example/docs
          branch: main
```

## RAG Evaluation and Benchmarking

### RAGAS Mode

Archi integrates the [Ragas RAG evaluator](https://docs.ragas.io/) for systematic evaluation:

**Metrics**:
1. **Answer Relevancy**: How relevant is the answer to the question?
2. **Faithfulness**: Is the answer grounded in the retrieved context?
3. **Context Precision**: How precise is the retrieved context?
4. **Context Relevancy**: How relevant is the context to the question?

**Configuration**:
```yaml
services:
  benchmarking:
    modes:
      - "RAGAS"
    evaluation_dataset: path/to/questions.csv
```

**Files**:
- `/src/bin/service_benchmark.py`: Benchmarking service
- `/scripts/benchmarking/benchmark_handler_functions.py`: Evaluation logic

### Sources Mode

Evaluate whether the system retrieves the correct source documents:

```yaml
services:
  benchmarking:
    modes:
      - "SOURCES"
```

## Document Management

### Enabling/Disabling Documents

Control which documents are included in retrieval:

**States**:
- **Enabled**: Document chunks are included in retrieval (default)
- **Disabled**: Document is excluded from retrieval but remains in database

**Use Cases**:
- Testing retrieval with specific document subsets
- Temporarily excluding outdated content
- A/B testing different document sets

### Document Uploader Interface

Web interface for managing documents:

**Features**:
- Upload new documents
- View existing documents
- Enable/disable documents
- Re-index documents
- View metadata

## Performance Optimization

### Indexing

PostgreSQL with pgvector supports HNSW indexing:

```sql
CREATE INDEX ON document_chunks 
USING hnsw (embedding vector_cosine_ops);
```

**Index Types**:
- HNSW: Fast approximate nearest neighbor (recommended for large datasets)
- IVFFlat: Inverted file index (alternative)

### Caching

Consider implementing caching for:
- Frequently accessed embeddings
- Common query results
- Document chunks

### Monitoring

Track RAG performance metrics:
- Retrieval latency
- Embedding generation time
- Document chunk counts
- Query patterns

## Best Practices

### 1. Choose the Right Retriever

- **Semantic**: Conceptual questions, broad topics
- **Hybrid**: Technical docs, specific terms, mixed queries
- **Grading**: Assessment tasks, rubric-based evaluation

### 2. Tune Chunk Size

- **Smaller chunks** (500-800): More precise, better for specific questions
- **Larger chunks** (1000-1500): More context, better for complex topics
- **Overlap**: 10-20% of chunk size to preserve context

### 3. Configure Weights (Hybrid Retriever)

Test different weight combinations:
- Start with 0.5/0.5
- Increase BM25 weight for technical/keyword-heavy content
- Increase semantic weight for conceptual content

### 4. Use Instructions (When Supported)

Instruction-aware models can improve retrieval:
```yaml
semantic_retriever:
  instructions: "Represent this query for retrieving relevant technical documentation:"
```

### 5. Monitor and Evaluate

Regularly run RAGAS evaluation:
- Track metrics over time
- Identify problematic queries
- Adjust configuration based on results

### 6. Organize Documents

Use metadata effectively:
- Tag documents by topic/category
- Track source and version
- Enable/disable by collection

## Advanced Topics

### Custom Retrievers

Extend `BaseRetriever` for custom logic:

```python
from langchain_core.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Custom retrieval logic
        pass
```

### Reranking

Add a reranking step after initial retrieval:

1. Initial retrieval (semantic/hybrid)
2. Rerank results using cross-encoder
3. Return top-k after reranking

### Multi-Modal RAG

Extend RAG to images/diagrams:

1. Extract text from images (OCR)
2. Generate image embeddings (CLIP)
3. Store in vector store alongside text

### Prompt Engineering for RAG

Optimize prompts for RAG responses:

```python
"""
You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{retriever_output}

Question: {question}

Provide a detailed answer with citations to specific sources.
"""
```

## Troubleshooting

### Low Retrieval Quality

**Symptoms**: Irrelevant documents retrieved

**Solutions**:
1. Check embedding model alignment with document type
2. Adjust chunk size (try smaller chunks)
3. Try hybrid retriever instead of semantic
4. Verify documents are properly indexed
5. Review query formulation

### Slow Retrieval

**Symptoms**: High latency for queries

**Solutions**:
1. Add HNSW index to vector columns
2. Reduce `num_documents_to_retrieve`
3. Use connection pooling for database
4. Consider approximate search (vs exact)

### Inconsistent Results

**Symptoms**: Different results for similar queries

**Solutions**:
1. Ensure consistent distance metric
2. Check for query preprocessing differences
3. Verify embedding model version
4. Review metadata filtering logic

## API Reference

### PostgresVectorStore

```python
store = PostgresVectorStore(
    pg_config={"host": "localhost", "port": 5432, ...},
    embedding_function=embeddings,
    collection_name="docs",
    distance_metric="cosine",
)

# Add documents
ids = store.add_texts(
    texts=["Document 1", "Document 2"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
)

# Search
results = store.similarity_search("query", k=5)
results_with_scores = store.similarity_search_with_score("query", k=5)

# Hybrid search (if supported)
hybrid_results = store.hybrid_search(
    query="query",
    k=5,
    semantic_weight=0.5,
    bm25_weight=0.5,
)
```

### Retrievers

```python
from src.data_manager.vectorstore.retrievers import (
    SemanticRetriever,
    HybridRetriever,
    GradingRetriever,
)

# Semantic retriever
semantic = SemanticRetriever(
    vectorstore=store,
    dm_config=config,
    k=5,
    instructions="Represent this query...",
)

# Hybrid retriever
hybrid = HybridRetriever(
    vectorstore=store,
    k=5,
    bm25_weight=0.6,
    semantic_weight=0.4,
)

# Get documents
docs = semantic.get_relevant_documents("query")
```

## Additional Resources

### Documentation
- [User Guide](user_guide.md) - Configuration and usage
- [API Reference](api_reference.md) - Detailed API documentation
- [Developer Guide](developer_guide.md) - Development setup

### External Resources
- [LangChain Documentation](https://python.langchain.com/) - LangChain framework
- [Ragas Documentation](https://docs.ragas.io/) - RAG evaluation framework
- [pgvector GitHub](https://github.com/pgvector/pgvector) - PostgreSQL vector extension

### Code Examples
- `/examples/` - Sample configurations
- `/examples/defaults/prompts/` - Prompt templates
- `/tests/` - Test cases showing usage patterns

## Summary

Archi's RAG implementation provides:

✅ **Flexible retrieval strategies** (semantic, hybrid, grading)  
✅ **PostgreSQL-native vector store** with pgvector  
✅ **Multiple embedding models** (OpenAI, HuggingFace)  
✅ **Configurable chunking and overlap**  
✅ **Hybrid search** (vector + BM25)  
✅ **Agent tool integration** for multi-step workflows  
✅ **Comprehensive evaluation** with RAGAS metrics  
✅ **Document management** (enable/disable, metadata)  
✅ **Production-ready** with connection pooling and indexing  

This makes Archi a powerful, configurable RAG framework suitable for research and educational applications.
