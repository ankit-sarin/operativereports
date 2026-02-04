# Operative Report Generator

## Project Info
- Location: ~/projects/operativereports
- Port: 7863
- URL: operativereports.digitalsurgeon.dev
- Python: use venv (source venv/bin/activate)

## Key Files
- app.py — Main Gradio app (two tabs)
- database.py — SQLite schema and CRUD
- rag_engine.py — ChromaDB embeddings and retrieval
- report_generator.py — Ollama LLM generation
- ocr_engine.py — GLM-OCR wrapper for PDF/image processing
- philter_runner.py — Philter de-identification wrapper
- bulk_import.py — Terminal tool for batch importing reports
- export_report.py — DOCX export
- load_mtsamples.py — One-time Kaggle data loader

## Architecture
- Track 1 (Tab 1):
  - Text paste → Philter → SQLite + ChromaDB
  - PDF/Image upload → GLM-OCR → Philter → SQLite + ChromaDB
- Track 2 (Tab 2): Fill form → RAG finds similar → Ollama generates report
- Bulk path: put files in own_reports/raw/ → python bulk_import.py

## Models (Ollama)
- qwen2.5:32b — Report generation (19GB)
- glm-ocr — OCR for scanned documents (2.2GB)

## Database
- reports.db (SQLite) — all operative reports
- chroma_db/ — vector embeddings for RAG

## PHI Safety
- own_reports/raw/ contains PHI — NEVER commit to git
- All reports in the database are de-identified
- Philter V1.0 (deidstable1_mirror) handles de-identification

## Commands
- Run app: python app.py
- Bulk import: python bulk_import.py
- Rebuild ChromaDB: python rag_engine.py
- Test OCR: python ocr_engine.py
- Service: sudo systemctl status operativereports
