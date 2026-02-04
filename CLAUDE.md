# Operative Reports Project

AI-powered medical operative report generation system with RAG (Retrieval-Augmented Generation) capabilities.

## Project Overview

This system helps generate medical operative reports by:
1. **Track 1**: Retrieving similar reports from a database of de-identified medical transcriptions
2. **Track 2**: Generating new reports based on surgeon inputs and RAG context

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OCR Engine    │────▶│    Database     │────▶│   RAG Engine    │
│   (GLM-OCR)     │     │   (SQLite)      │     │   (ChromaDB)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  LLM Generation │
                                                │    (Ollama)     │
                                                └─────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `database.py` | SQLite database operations for reports storage |
| `rag_engine.py` | ChromaDB + sentence-transformers for semantic search |
| `ocr_engine.py` | GLM-OCR via Ollama for text extraction from images/PDFs |
| `load_mtsamples.py` | Script to load MTSamples dataset into database |
| `requirements.txt` | Python dependencies |

## Database Schema

### `reports` table
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `procedure_type` TEXT NOT NULL
- `specialty` TEXT NOT NULL
- `report_name` TEXT
- `report_text` TEXT NOT NULL
- `keywords` TEXT
- `source` TEXT NOT NULL (e.g., 'MTSamples/Kaggle', 'Own Clinical - Philter De-identified')
- `is_deidentified` BOOLEAN DEFAULT TRUE
- `added_at` TIMESTAMP

### `generated_reports` table (Track 2)
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `procedure_type` TEXT NOT NULL
- `surgeon_inputs` TEXT NOT NULL (JSON)
- `generated_report` TEXT NOT NULL
- `user_rating` INTEGER
- `created_at` TIMESTAMP

## Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database and load MTSamples data
python database.py
python load_mtsamples.py

# Build RAG index (takes a few minutes)
python rag_engine.py
```

## Data Sources

- **MTSamples/Kaggle**: ~5000 medical transcription samples, filtered to Surgery and Gastroenterology specialties (~1312 records)
- **Own Clinical**: User-uploaded reports, de-identified using Philter

## Key Classes

### `database.py`
```python
init_db()                      # Initialize tables
add_report(...)                # Add a report
get_report(id)                 # Get by ID
search_reports(...)            # Search with filters
delete_report(id)              # Delete by ID
get_all_reports(limit, offset) # Paginated list
get_report_count_by_source()   # Stats by source
```

### `rag_engine.py`
```python
class RAGEngine:
    add_report(id, text, procedure, specialty)  # Add to ChromaDB
    search_similar(query, n_results=3)          # Semantic search
    get_relevant_context(procedure, findings)   # Formatted RAG context
    rebuild_from_db()                           # Full rebuild from SQLite
    add_single_report(...)                      # Add without rebuild
    delete_report(id)                           # Remove from index
```

### `ocr_engine.py`
```python
class OCREngine:
    process_image(path)  # OCR on .png/.jpg/.jpeg
    process_pdf(path)    # OCR on PDF (page by page)
    process_file(path)   # Auto-detect and process
```

## Environment

- Python 3.10+
- Ollama for local LLM inference
- ChromaDB persisted to `./chroma_db/`
- SQLite database at `./reports.db`

## .gitignore Exclusions

- `own_reports/` subdirectories (PHI data)
- `reports.db`, `chroma_db/` (generated/rebuildable)
- `mtsamples.csv` (large data file)
- `venv/`, `__pycache__/`, `philter/`

## Conventions

- All database functions return dicts (via `sqlite3.Row`)
- RAG engine uses `all-MiniLM-L6-v2` embeddings (384 dimensions)
- OCR returns error strings on failure (doesn't raise exceptions)
- Sources should be one of: `'MTSamples/Kaggle'`, `'Own Clinical - Philter De-identified'`

## Future Components (Planned)

- Gradio web interface for report generation
- De-identification pipeline using Philter
- Report export to DOCX format
- User feedback/rating system for generated reports
