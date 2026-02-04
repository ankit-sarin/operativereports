"""
Bulk Import - Process files from own_reports/raw/ and import into the database.

Workflow:
1. Scan own_reports/raw/ for supported file types
2. Extract text (direct read for .txt, OCR for images/PDFs)
3. De-identify using Philter
4. Save de-identified version to own_reports/deid/
5. Import into reports.db
6. Add to ChromaDB for RAG
7. Move original to own_reports/imported/
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from ocr_engine import OCREngine
from philter_runner import deidentify_text
from database import init_db, add_report
from rag_engine import RAGEngine

# Directories
RAW_DIR = Path("own_reports/raw")
DEID_DIR = Path("own_reports/deid")
IMPORTED_DIR = Path("own_reports/imported")

# Supported file types
TEXT_EXTENSIONS = {'.txt'}
OCR_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}
ALL_EXTENSIONS = TEXT_EXTENSIONS | OCR_EXTENSIONS

# Source identifier for imported reports
SOURCE = "Own Clinical - Philter De-identified"
DEFAULT_SPECIALTY = "Surgery"


def ensure_directories():
    """Create required directories if they don't exist."""
    for directory in [RAW_DIR, DEID_DIR, IMPORTED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def extract_procedure_type(text: str) -> str:
    """
    Extract procedure type from report text.

    Looks for common headers like:
    - PROCEDURE:
    - OPERATION:
    - OPERATIVE PROCEDURE:
    - PROCEDURE PERFORMED:

    Returns extracted procedure type or 'Unknown Procedure'
    """
    # Patterns to look for procedure type
    patterns = [
        r'(?:OPERATIVE\s+)?PROCEDURE(?:\s+PERFORMED)?[:\s]+([^\n]+)',
        r'OPERATION(?:\s+PERFORMED)?[:\s]+([^\n]+)',
        r'SURGERY[:\s]+([^\n]+)',
        r'POSTOPERATIVE\s+DIAGNOSIS[:\s]+([^\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            procedure = match.group(1).strip()
            # Clean up the procedure text
            procedure = re.sub(r'^[,\s]+', '', procedure)  # Remove leading commas/spaces
            procedure = re.sub(r'[,\s]+$', '', procedure)  # Remove trailing commas/spaces
            # Truncate if too long
            if len(procedure) > 200:
                procedure = procedure[:200] + "..."
            if procedure and procedure.lower() not in ['none', 'n/a', '']:
                return procedure

    return "Unknown Procedure"


def extract_specialty(text: str) -> str:
    """
    Try to extract medical specialty from report text.

    Returns extracted specialty or default.
    """
    # Common specialty indicators
    specialties = {
        'gastroenterology': ['gastro', 'endoscopy', 'colonoscopy', 'egd', 'ercp'],
        'orthopedic surgery': ['orthopedic', 'arthroplasty', 'fracture', 'joint'],
        'cardiothoracic surgery': ['cardiothoracic', 'cabg', 'cardiac', 'thoracotomy'],
        'neurosurgery': ['neurosurg', 'craniotomy', 'laminectomy', 'spine'],
        'urology': ['urolog', 'cystoscopy', 'prostatectomy', 'nephrectomy'],
        'gynecology': ['gynecolog', 'hysterectomy', 'oophorectomy'],
        'general surgery': ['appendectomy', 'cholecystectomy', 'hernia', 'laparoscopic'],
    }

    text_lower = text.lower()
    for specialty, keywords in specialties.items():
        for keyword in keywords:
            if keyword in text_lower:
                return specialty.title()

    return DEFAULT_SPECIALTY


def read_text_file(file_path: Path) -> Tuple[str, Optional[str]]:
    """
    Read text from a .txt file.

    Returns (text, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), None
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read(), None
        except Exception as e:
            return "", f"Failed to read file: {e}"
    except Exception as e:
        return "", f"Failed to read file: {e}"


def process_file(
    file_path: Path,
    ocr_engine: OCREngine,
    rag_engine: RAGEngine
) -> Dict:
    """
    Process a single file through the import pipeline.

    Returns a dict with processing results.
    """
    result = {
        'filename': file_path.name,
        'file_type': file_path.suffix.lower(),
        'status': 'pending',
        'error': None,
        'report_id': None,
        'procedure_type': None,
    }

    try:
        # Step 1: Extract text
        ext = file_path.suffix.lower()

        if ext in TEXT_EXTENSIONS:
            raw_text, error = read_text_file(file_path)
            if error:
                result['status'] = 'failed'
                result['error'] = error
                return result
        elif ext in OCR_EXTENSIONS:
            raw_text = ocr_engine.process_file(str(file_path))
            if raw_text.startswith("Error:"):
                result['status'] = 'failed'
                result['error'] = raw_text
                return result
        else:
            result['status'] = 'skipped'
            result['error'] = f"Unsupported file type: {ext}"
            return result

        # Check if we got meaningful text
        if not raw_text or len(raw_text.strip()) < 50:
            result['status'] = 'failed'
            result['error'] = "Extracted text too short or empty"
            return result

        # Step 2: De-identify with Philter
        deid_text = deidentify_text(raw_text)
        if deid_text.startswith("Error:"):
            result['status'] = 'failed'
            result['error'] = f"De-identification failed: {deid_text}"
            return result

        # Step 3: Save de-identified version
        deid_filename = file_path.stem + ".txt"
        deid_path = DEID_DIR / deid_filename
        with open(deid_path, 'w', encoding='utf-8') as f:
            f.write(deid_text)

        # Step 4: Extract metadata
        procedure_type = extract_procedure_type(deid_text)
        specialty = extract_specialty(deid_text)
        result['procedure_type'] = procedure_type

        # Step 5: Add to database
        report_id = add_report(
            procedure_type=procedure_type,
            specialty=specialty,
            report_text=deid_text,
            source=SOURCE,
            report_name=file_path.stem,
            keywords=None,
            is_deidentified=True
        )
        result['report_id'] = report_id

        # Step 6: Add to ChromaDB for RAG
        rag_engine.add_single_report(
            report_id=report_id,
            report_text=deid_text,
            procedure_type=procedure_type,
            specialty=specialty
        )

        # Step 7: Move original to imported/
        imported_path = IMPORTED_DIR / file_path.name
        # Handle duplicate filenames
        counter = 1
        while imported_path.exists():
            imported_path = IMPORTED_DIR / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
        shutil.move(str(file_path), str(imported_path))

        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def bulk_import():
    """
    Run the bulk import process on all files in own_reports/raw/
    """
    print("Bulk Import - Own Clinical Reports")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Ensure directories exist
    ensure_directories()

    # Initialize database
    print("Initializing database...")
    init_db()

    # Initialize engines
    print("Initializing OCR engine...")
    ocr_engine = OCREngine()

    print("Initializing RAG engine...")
    rag_engine = RAGEngine()

    # Scan for files
    print(f"\nScanning {RAW_DIR}...")
    files_to_process = []
    for ext in ALL_EXTENSIONS:
        files_to_process.extend(RAW_DIR.glob(f"*{ext}"))
        files_to_process.extend(RAW_DIR.glob(f"*{ext.upper()}"))

    # Remove duplicates (case-insensitive matching might cause this)
    files_to_process = list(set(files_to_process))
    files_to_process.sort(key=lambda x: x.name.lower())

    if not files_to_process:
        print("\nNo files found to process.")
        print(f"Place files in: {RAW_DIR.absolute()}")
        print(f"Supported types: {', '.join(sorted(ALL_EXTENSIONS))}")
        return

    print(f"Found {len(files_to_process)} file(s) to process")
    print()

    # Process each file
    results = []
    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] Processing: {file_path.name}...", end=" ", flush=True)
        result = process_file(file_path, ocr_engine, rag_engine)
        results.append(result)

        if result['status'] == 'success':
            print(f"✓ (ID: {result['report_id']})")
        elif result['status'] == 'skipped':
            print(f"⊘ Skipped: {result['error']}")
        else:
            print(f"✗ Failed: {result['error']}")

    # Print summary
    print()
    print("=" * 60)
    print("IMPORT SUMMARY")
    print("=" * 60)

    # Count by status
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')

    print(f"\nTotal files:     {len(results)}")
    print(f"  Successful:    {success_count}")
    print(f"  Failed:        {failed_count}")
    print(f"  Skipped:       {skipped_count}")

    # Count by file type
    print("\nBy file type:")
    type_counts = {}
    for r in results:
        ft = r['file_type']
        if ft not in type_counts:
            type_counts[ft] = {'success': 0, 'failed': 0, 'skipped': 0}
        type_counts[ft][r['status']] += 1

    for ft, counts in sorted(type_counts.items()):
        total = sum(counts.values())
        print(f"  {ft}: {total} ({counts['success']} success, {counts['failed']} failed, {counts['skipped']} skipped)")

    # List errors if any
    errors = [r for r in results if r['status'] == 'failed']
    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  - {r['filename']}: {r['error']}")

    # Show procedure types extracted
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        print("\nProcedure types extracted:")
        for r in successful:
            proc = r['procedure_type'][:50] + "..." if len(r['procedure_type']) > 50 else r['procedure_type']
            print(f"  - {r['filename']}: {proc}")

    print()
    print(f"De-identified files saved to: {DEID_DIR.absolute()}")
    print(f"Original files moved to: {IMPORTED_DIR.absolute()}")


if __name__ == "__main__":
    bulk_import()
