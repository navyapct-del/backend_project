# Data Orchestration — Backend

Python-based Azure Function App that powers the Data Orchestration platform. Handles document ingestion, OCR, RAG-based querying, chat history, image analysis, and agentic AI workflows — all exposed as HTTP endpoints behind Azure API Management.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │              Azure Virtual Network                   │
                        │                                                       │
  React Frontend        │  ┌─────────────────┐      ┌──────────────────────┐  │
  (Static Web App)  ───►│  │  function-subnet │      │ private-endpoint-    │  │
        │               │  │                 │      │ subnet               │  │
        │               │  │  Azure Function │      │                      │  │
        ▼               │  │  App (Python)   │─────►│  ● Storage PE (blob) │  │
  Azure APIM            │  │                 │      │  ● Storage PE (table)│  │
  (Consumption)    ───►│  │  VNET_ROUTE_ALL │      │  ● Key Vault PE      │  │
        │               │  │  = 1            │      │  ● Doc Intelligence  │  │
        │               │  └────────┬────────┘      │    PE                │  │
        │               │           │               └──────────────────────┘  │
        │               │           │ Private DNS                              │
        │               └───────────┼─────────────────────────────────────────┘
        │                           │
        │               ┌───────────▼──────────────────────────────────────────┐
        │               │              Azure Services (Private)                 │
        │               │                                                        │
        │               │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
        │               │  │ Azure Storage│  │  Key Vault   │  │  Document  │  │
        │               │  │  Blob: files │  │  API keys    │  │Intelligence│  │
        │               │  │  Table: meta │  │  conn strings│  │  OCR/Read  │  │
        │               │  └──────────────┘  └──────────────┘  └────────────┘  │
        │               │                                                        │
        │               │  ┌──────────────┐  ┌──────────────┐                  │
        │               │  │ Azure OpenAI │  │  Azure AI    │                  │
        │               │  │  GPT-4       │  │  Search      │                  │
        │               │  │  Embeddings  │  │  Vector+BM25 │                  │
        │               │  └──────────────┘  └──────────────┘                  │
        │               └────────────────────────────────────────────────────────┘
        │
        └──► App Insights + Log Analytics (Monitoring)
```

---

## Infrastructure

| Resource | Type | Purpose |
|---|---|---|
| `dataorch-new-funcapp` | Azure Function App (Linux, Consumption) | Backend API runtime |
| `dataorch-new-apim` | API Management (Consumption) | API gateway, auth, rate limiting |
| `dataorch-new-openai` | Azure OpenAI (S0) | GPT-4 chat, embeddings, summarization |
| `dataorch-new-search` | Azure AI Search (Free) | Vector + hybrid document search |
| `dataorchnewstorage` | Storage Account (Standard LRS) | Blob: files, Table: metadata |
| `dataorch-new-docintel` | Document Intelligence | OCR for PDFs and images |
| `dataorch-new-kv` | Key Vault | Secrets management |
| `dataorch-new-appinsights` | Application Insights | APM and distributed tracing |
| `dataorch-new-law` | Log Analytics Workspace | Log storage and KQL queries |
| `dataorch-new-vnet` | Virtual Network | Network isolation |
| `dataorch-new-nsg` | Network Security Group | Subnet-level traffic rules |

**Networking:**
- `function-subnet` (10.0.2.0/24) — Function App VNet integration, delegated to `Microsoft.App/environments`
- `private-endpoint-subnet` (10.0.3.0/24) — Private endpoints for Storage, Key Vault, Document Intelligence
- `WEBSITE_VNET_ROUTE_ALL=1` — All outbound Function App traffic routes through VNet

---

## Project Structure

```
backend_project/
├── azure_upload_function/
│   ├── function_app.py          # All HTTP route handlers (~1900 lines)
│   ├── requirements.txt         # Python dependencies
│   ├── host.json                # Function App host configuration
│   ├── services/
│   │   ├── rag_pipeline.py      # RAG orchestration pipeline
│   │   ├── query_engine.py      # Query processing and answer generation
│   │   ├── search_service.py    # Azure AI Search integration
│   │   ├── openai_service.py    # Azure OpenAI (chat, embeddings, summary, tags)
│   │   ├── extractor.py         # Document text extraction (PDF, Word, CSV, Excel)
│   │   ├── doc_intelligence_service.py  # OCR via Azure Document Intelligence
│   │   ├── blob_service.py      # Azure Blob Storage operations
│   │   ├── table_service.py     # Azure Table Storage (document metadata)
│   │   ├── chat_storage_service.py      # Chat session persistence
│   │   ├── delete_service.py    # Document deletion (blob + index + metadata)
│   │   ├── image_understanding_service.py  # GPT-4 Vision image analysis
│   │   ├── image_search_service.py      # Image search via embeddings
│   │   ├── intent_classifier.py # Query intent classification
│   │   ├── rag_service.py       # RAG service wrapper
│   │   ├── chunking_service.py  # Document chunking for indexing
│   │   ├── analytics_service.py # Usage analytics
│   │   ├── session_service.py   # Session management
│   │   ├── summary_service.py   # Document summarization
│   │   ├── language_service.py  # Language detection
│   │   ├── router_service.py    # Request routing logic
│   │   ├── cleaner.py           # CSV/Excel data cleaning
│   │   └── config.py            # Environment variable helpers
│   └── tests/
│       ├── test_chat_storage.py
│       └── test_agentic_bot_properties.py
├── apim/
│   └── policy.xml               # APIM inbound/outbound policies
└── .github/
    └── workflows/
        └── deploy.yml           # CI/CD pipeline
```

---

## API Endpoints

### Document Management
| Method | Route | Description |
|---|---|---|
| `POST` | `/upload` | Upload a document — OCR, embed, index, store metadata |
| `GET` | `/documents` | List all documents for a user |
| `GET` | `/download/{id}` | Download a document by ID |
| `DELETE` | `/document/{id}` | Delete document (blob + search index + metadata) |
| `GET` | `/file` | Get file content/URL |
| `POST` | `/reprocess` | Re-extract and re-index an existing document |
| `POST` | `/reset-index` | Reset the search index |

### Query & RAG
| Method | Route | Description |
|---|---|---|
| `GET/POST` | `/query` | RAG query — hybrid search + GPT-4 answer generation |
| `POST` | `/agent/query` | Agentic multi-step query with tool use |
| `POST` | `/agent/ask` | Direct ask without document context |

### Image
| Method | Route | Description |
|---|---|---|
| `POST` | `/agent/image-search` | Search documents using an image |
| `POST` | `/agent/analyze-image` | Analyze image content with GPT-4 Vision |

### Chat History
| Method | Route | Description |
|---|---|---|
| `POST` | `/saveMessage` | Save a chat message to session |
| `GET` | `/getChatHistory` | Get chat history for a session |
| `GET` | `/chatSessions` | List all chat sessions for a user |
| `GET` | `/chatSession/{sessionId}` | Get a specific chat session |
| `DELETE` | `/chatSession/{sessionId}` | Delete a chat session |
| `POST` | `/shareChat` | Share a chat session |
| `GET` | `/chatSession/{sessionId}/shared` | Get shared chat session |
| `POST` | `/cleanup-session` | Clean up expired sessions |

### Auth
| Method | Route | Description |
|---|---|---|
| `POST` | `/register` | Register a new user |
| `POST` | `/login` | User login |

### Utilities
| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/diagnose` | Diagnose service connectivity |
| `GET` | `/agent/diagnose-search` | Diagnose search index |

---

## Document Upload Flow

```
POST /upload
     │
     ├─► Duplicate check (Table Storage — find_by_filename)
     │        └─► 409 Conflict if exists
     │
     ├─► Blob Storage upload (file bytes)
     │
     ├─► Text Extraction (extractor.py)
     │        ├─► PDF → pdfplumber / PyPDF2 / Document Intelligence OCR
     │        ├─► Word → python-docx
     │        ├─► CSV/Excel → pandas + cleaner
     │        └─► Image → Document Intelligence (prebuilt-read)
     │
     ├─► Summary generation (Azure OpenAI GPT-4)
     │
     ├─► Tag generation (Azure OpenAI GPT-4)
     │
     ├─► Embedding generation (Azure OpenAI text-embedding-ada-002)
     │
     ├─► Search index upsert (Azure AI Search — vector + keyword)
     │
     └─► Metadata insert (Azure Table Storage)
              └─► {id, filename, blob_url, summary, tags, file_type, uploaded_by, created_at}
```

## RAG Query Flow

```
POST /query  {question, user_id, doc_ids?}
     │
     ├─► Intent classification (intent_classifier.py)
     │
     ├─► Embedding of question (Azure OpenAI)
     │
     ├─► Hybrid search (Azure AI Search)
     │        ├─► Vector search (semantic similarity)
     │        ├─► BM25 keyword search
     │        └─► doc_id_filter (if specific docs selected)
     │
     ├─► Context assembly (top-k chunks)
     │
     ├─► GPT-4 answer generation with system prompt
     │        └─► Structured response: {answer, sources, type}
     │
     └─► Response + source citations
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `azure-functions` | 1.19.0 | Function App runtime |
| `azure-storage-blob` | 12.23.1 | Blob Storage client |
| `azure-data-tables` | 12.5.0 | Table Storage client |
| `azure-ai-documentintelligence` | 1.0.0 | OCR / Document Intelligence |
| `azure-search-documents` | 11.6.0 | Azure AI Search client |
| `openai` | 1.51.0 | Azure OpenAI client |
| `pandas` | 2.2.3 | CSV/Excel processing |
| `pdfplumber` | 0.11.4 | PDF text extraction |
| `PyPDF2` | 3.0.1 | PDF fallback extraction |
| `python-docx` | 1.1.2 | Word document extraction |
| `Pillow` | 10.4.0 | Image processing |
| `PyJWT` | 2.8.0 | JWT auth tokens |
| `bcrypt` | 4.1.3 | Password hashing |

---

## Environment Variables

All secrets are stored in **Azure Key Vault** and injected at runtime:

| Variable | Description |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Storage Account connection string |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | GPT-4 deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_KEY` | Azure AI Search admin key |
| `AZURE_SEARCH_INDEX` | Search index name |
| `DOC_INTELLIGENCE_ENDPOINT` | Document Intelligence endpoint |
| `DOC_INTELLIGENCE_KEY` | Document Intelligence API key |
| `JWT_SECRET` | JWT signing secret |

---

## Local Development

```bash
cd azure_upload_function

# Install dependencies
pip install -r requirements.txt

# Start locally (requires Azure Functions Core Tools)
func start

# Run tests
pytest tests/
```

---

## CI/CD

Deployed via GitHub Actions (`.github/workflows/deploy.yml`) to Azure Function App on push to `main`.
