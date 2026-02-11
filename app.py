"""
Operative Report Generator - Gradio Web Application

Two-tab interface:
1. Add Report to Database - OCR, de-identify, and store reports
2. Generate Report - AI-powered report generation from surgeon inputs
"""

import gradio as gr
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import os
import json
import ollama

from database import init_db, add_report, get_report, get_all_reports, delete_report, get_report_count_by_source
from rag_engine import RAGEngine
from report_generator import ReportGenerator
from philter_runner import deidentify_text
from ocr_engine import OCREngine
from export_report import export_to_docx

# Initialize components
print("Initializing database...")
init_db()

print("Initializing RAG engine...")
rag_engine = RAGEngine()

print("Initializing Report Generator...")
report_gen = ReportGenerator(rag_engine=rag_engine)

print("Initializing OCR engine...")
ocr_engine = OCREngine()

# Specialty options
SPECIALTY_OPTIONS = [
    "General Surgery",
    "Vascular Surgery",
    "Colorectal Surgery",
    "Surgical Oncology",
    "Trauma Surgery",
    "Pediatric Surgery",
    "Gastroenterology",
    "Other"
]

# Procedure type options for generation
PROCEDURE_OPTIONS = [
    "Laparoscopic Cholecystectomy",
    "Open Cholecystectomy",
    "Laparoscopic Appendectomy",
    "Open Appendectomy",
    "Inguinal Hernia Repair - Open",
    "Inguinal Hernia Repair - Laparoscopic (TEP)",
    "Inguinal Hernia Repair - Laparoscopic (TAPP)",
    "Ventral Hernia Repair",
    "Umbilical Hernia Repair",
    "Other (specify in details)"
]

# Anesthesia options
ANESTHESIA_OPTIONS = [
    "General endotracheal",
    "General with LMA",
    "Spinal",
    "Epidural",
    "Regional block",
    "Local with sedation",
    "Local only"
]


# ============== TAB 1 FUNCTIONS ==============

def process_and_add_report(
    pasted_text: str,
    uploaded_file,
    procedure_type: str,
    specialty: str
) -> Tuple[str, str, str]:
    """
    Process input (text or file), de-identify, and add to database.

    Returns: (ocr_result, deid_text, status_message)
    """
    ocr_result = ""
    raw_text = ""

    try:
        # Determine input source
        if uploaded_file is not None:
            # File uploaded - use OCR
            file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
            ocr_result = ocr_engine.process_file(file_path)

            if ocr_result.startswith("Error:"):
                return ocr_result, "", f"OCR failed: {ocr_result}"

            raw_text = ocr_result
        elif pasted_text and pasted_text.strip():
            # Use pasted text
            raw_text = pasted_text.strip()
        else:
            return "", "", "Error: Please paste text or upload a file."

        # De-identify with Philter
        deid_text = deidentify_text(raw_text)

        if deid_text.startswith("Error:"):
            return ocr_result, deid_text, f"De-identification failed: {deid_text}"

        # Auto-extract procedure type if not provided
        if not procedure_type or not procedure_type.strip():
            procedure_type = extract_procedure_type(deid_text)

        # Add to database
        report_id = add_report(
            procedure_type=procedure_type,
            specialty=specialty,
            report_text=deid_text,
            source="Own Clinical - Philter De-identified",
            report_name=None,
            keywords=None,
            is_deidentified=True
        )

        # Add to RAG index
        rag_engine.add_single_report(
            report_id=report_id,
            report_text=deid_text,
            procedure_type=procedure_type,
            specialty=specialty
        )

        status = f"✓ Report added successfully! ID: {report_id}"
        return ocr_result, deid_text, status

    except Exception as e:
        return ocr_result, "", f"Error: {str(e)}"


def extract_procedure_type(text: str) -> str:
    """Extract procedure type from report text."""
    import re
    patterns = [
        r'(?:OPERATIVE\s+)?PROCEDURE(?:\s+PERFORMED)?[:\s]+([^\n]+)',
        r'OPERATION(?:\s+PERFORMED)?[:\s]+([^\n]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            proc = match.group(1).strip()
            proc = proc.replace('**', '').strip()
            if len(proc) > 100:
                proc = proc[:100] + "..."
            return proc
    return "Unknown Procedure"


def refresh_database_view() -> Tuple[pd.DataFrame, str]:
    """Get all reports and database stats."""
    reports = get_all_reports(limit=500)

    if not reports:
        return pd.DataFrame(), "Database is empty."

    # Build dataframe
    data = []
    for r in reports:
        preview = r.get('report_text', '')[:80] + "..." if len(r.get('report_text', '')) > 80 else r.get('report_text', '')
        data.append({
            'ID': r.get('id'),
            'Procedure Type': r.get('procedure_type', '')[:50],
            'Specialty': r.get('specialty', ''),
            'Source': r.get('source', ''),
            'Added': str(r.get('added_at', ''))[:19],
            'Preview': preview
        })

    df = pd.DataFrame(data)

    # Get stats
    stats = get_report_count_by_source()
    total = sum(stats.values())
    stats_text = f"Total Reports: {total}\n"
    for source, count in stats.items():
        stats_text += f"  - {source}: {count}\n"

    return df, stats_text


def view_full_report(report_id: int) -> str:
    """Get full text of a report by ID."""
    if not report_id:
        return "Please enter a Report ID."

    report = get_report(int(report_id))
    if report:
        return report.get('report_text', 'No text found.')
    return f"Report ID {report_id} not found."


def delete_report_handler(report_id: int) -> str:
    """Delete a report from database and RAG index."""
    if not report_id:
        return "Please enter a Report ID."

    report_id = int(report_id)

    # Delete from database
    deleted = delete_report(report_id)

    if deleted:
        # Delete from RAG index
        rag_engine.delete_report(report_id)
        return f"✓ Report ID {report_id} deleted successfully."
    else:
        return f"Report ID {report_id} not found."


# ============== TAB 2 FUNCTIONS ==============

def extract_from_brief_note(
    pasted_text: str,
    uploaded_file
) -> Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str]:
    """
    Extract structured data from a brief operative note and return field values.

    Returns tuple of:
    (procedure_type, preop_diagnosis, postop_diagnosis, surgeon_name, assistant,
     anesthesia_type, indications, findings, procedure_details, specimens,
     drains, ebl, complications, status_message)
    """
    raw_text = ""
    empty_result = ("", "", "", "", "", "", "", "", "", "", "", "", "", "")

    try:
        # Determine input source
        if uploaded_file is not None:
            # File uploaded - try OCR
            file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
            ocr_result = ocr_engine.process_file(file_path)

            if ocr_result.startswith("Error:"):
                # OCR failed - likely model not installed
                return empty_result[:-1] + ("OCR model not yet available. Please paste the text instead.",)

            raw_text = ocr_result
        elif pasted_text and pasted_text.strip():
            raw_text = pasted_text.strip()
        else:
            return empty_result[:-1] + ("Please paste text or upload a file.",)

        # De-identify with Philter
        deid_text = deidentify_text(raw_text)

        if deid_text.startswith("Error:"):
            return empty_result[:-1] + (f"De-identification failed: {deid_text}",)

        # Build extraction prompt
        extraction_prompt = """Extract the following fields from this brief operative note.
Return ONLY a valid JSON object with these keys:
procedure_type, preop_diagnosis, postop_diagnosis, surgeon_name, assistant,
anesthesia_type, indications, findings, procedure_details, specimens,
drains, ebl, complications.

If a field is not mentioned, use an empty string for that field.

Brief Operative Note:
"""
        extraction_prompt += deid_text

        # Call Ollama to extract fields
        response = ollama.chat(
            model="qwen2.5:32b",
            messages=[
                {"role": "system", "content": "You are a medical data extraction assistant. Extract structured information from operative notes and return valid JSON only."},
                {"role": "user", "content": extraction_prompt}
            ],
            options={
                "temperature": 0.1,  # Low temperature for consistent extraction
                "num_predict": 1024,
            }
        )

        response_text = response['message']['content']

        # Try to parse JSON from response
        # Handle cases where model might wrap JSON in markdown code blocks
        json_text = response_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        extracted = json.loads(json_text.strip())

        # Return extracted values in order
        return (
            extracted.get("procedure_type", ""),
            extracted.get("preop_diagnosis", ""),
            extracted.get("postop_diagnosis", ""),
            extracted.get("surgeon_name", ""),
            extracted.get("assistant", ""),
            extracted.get("anesthesia_type", ""),
            extracted.get("indications", ""),
            extracted.get("findings", ""),
            extracted.get("procedure_details", ""),
            extracted.get("specimens", ""),
            extracted.get("drains", ""),
            extracted.get("ebl", ""),
            extracted.get("complications", ""),
            "Fields extracted successfully. Review and edit as needed."
        )

    except json.JSONDecodeError as e:
        return empty_result[:-1] + (f"Failed to parse extraction response as JSON: {str(e)}",)
    except Exception as e:
        return empty_result[:-1] + (f"Error extracting fields: {str(e)}",)


def generate_report_handler(
    procedure_type: str,
    preop_diagnosis: str,
    postop_diagnosis: str,
    surgeon: str,
    assistant: str,
    anesthesia: str,
    indications: str,
    findings: str,
    procedure_details: str,
    specimens: str,
    drains: str,
    ebl: str,
    complications: str
) -> str:
    """Generate an operative report from inputs."""
    if not procedure_type or not preop_diagnosis or not surgeon:
        return "Error: Please fill in at least Procedure Type, Preop Diagnosis, and Surgeon."

    try:
        report = report_gen.generate_report(
            procedure_type=procedure_type,
            preop_diagnosis=preop_diagnosis,
            postop_diagnosis=postop_diagnosis or preop_diagnosis,
            surgeon_name=surgeon,
            assistant=assistant or "",
            anesthesia_type=anesthesia,
            indications=indications or "",
            findings=findings or "",
            procedure_details=procedure_details or "",
            specimens=specimens or "None",
            drains=drains or "None",
            ebl=ebl or "Minimal",
            complications=complications or "None"
        )
        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"


def export_report_handler(report_text: str) -> Optional[str]:
    """Export report to Word document."""
    if not report_text or report_text.startswith("Error"):
        return None

    try:
        import time
        filename = f"operative_report_{int(time.time())}"
        filepath = export_to_docx(report_text, filename)
        return filepath
    except Exception as e:
        return None


# ============== BUILD GRADIO APP ==============

_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio-theme.css")
with open(_css_path, "r") as f:
    brand_css = f.read()

with gr.Blocks(title="Operative Report Generator", css=brand_css) as app:

    # Header
    gr.Markdown(
        """
        # Operative Report Generator
        ### AI-Powered Surgical Documentation
        """
    )

    with gr.Tabs():

        # ============== TAB 1: Add Report ==============
        with gr.TabItem("Add Report to Database"):
            gr.Markdown("### Add a new report by pasting text or uploading a scanned document")

            # Input row
            with gr.Row():
                with gr.Column():
                    paste_input = gr.Textbox(
                        label="Paste Report Text",
                        placeholder="Paste the raw operative report text here...",
                        lines=15
                    )
                with gr.Column():
                    file_input = gr.File(
                        label="Or Upload PDF/Image",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        type="filepath"
                    )

            gr.Markdown(
                "*Paste text directly OR upload a scanned document. "
                "If both provided, uploaded file takes priority.*"
            )

            # Metadata inputs
            with gr.Row():
                procedure_input = gr.Textbox(
                    label="Procedure Type (optional — will be auto-extracted if empty)",
                    placeholder="e.g., Laparoscopic Cholecystectomy"
                )
                specialty_input = gr.Dropdown(
                    label="Specialty",
                    choices=SPECIALTY_OPTIONS,
                    value="General Surgery"
                )

            # Process button
            add_btn = gr.Button("De-identify & Add to Database", variant="primary", size="lg")

            # Output area
            with gr.Row():
                ocr_output = gr.Textbox(
                    label="OCR Result (if file uploaded)",
                    lines=8,
                    interactive=False
                )
                deid_output = gr.Textbox(
                    label="De-identified Text",
                    lines=8,
                    interactive=False
                )

            status_output = gr.Textbox(label="Status", interactive=False)

            # Wire up the add button
            add_btn.click(
                fn=process_and_add_report,
                inputs=[paste_input, file_input, procedure_input, specialty_input],
                outputs=[ocr_output, deid_output, status_output]
            )

            # Database Management Section
            with gr.Accordion("Database Management", open=False):
                refresh_btn = gr.Button("Refresh Database View")

                db_table = gr.Dataframe(
                    label="Reports in Database",
                    headers=["ID", "Procedure Type", "Specialty", "Source", "Added", "Preview"],
                    interactive=False
                )

                db_stats = gr.Textbox(label="Database Statistics", interactive=False, lines=4)

                refresh_btn.click(
                    fn=refresh_database_view,
                    inputs=[],
                    outputs=[db_table, db_stats]
                )

                gr.Markdown("---")

                # View report
                with gr.Row():
                    view_id_input = gr.Number(label="Report ID", precision=0)
                    view_btn = gr.Button("View Full Report")

                full_report_output = gr.Textbox(
                    label="Full Report Text",
                    lines=10,
                    interactive=False
                )

                view_btn.click(
                    fn=view_full_report,
                    inputs=[view_id_input],
                    outputs=[full_report_output]
                )

                gr.Markdown("---")

                # Delete report
                with gr.Row():
                    delete_id_input = gr.Number(label="Report ID to Delete", precision=0)
                    delete_btn = gr.Button("Delete Report", variant="stop")

                delete_status = gr.Textbox(label="Delete Status", interactive=False)

                delete_btn.click(
                    fn=delete_report_handler,
                    inputs=[delete_id_input],
                    outputs=[delete_status]
                )

        # ============== TAB 2: Generate Report ==============
        with gr.TabItem("Generate Report"):
            gr.Markdown("### Generate a complete operative report from procedure details")

            # Import from Brief Op Note section
            with gr.Accordion("Import from Brief Op Note (auto-fill form)", open=False):
                gr.Markdown("*Paste or upload a brief operative note to automatically extract and fill the form fields below.*")

                with gr.Row():
                    with gr.Column():
                        import_text = gr.Textbox(
                            label="Paste Brief Op Note Text",
                            placeholder="Paste the brief operative note here...",
                            lines=10
                        )
                    with gr.Column():
                        import_file = gr.File(
                            label="Or Upload Brief Op Note (image/PDF)",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                            type="filepath"
                        )

                extract_btn = gr.Button("Extract & Fill Form", variant="secondary")
                extract_status = gr.Textbox(label="Extraction Status", interactive=False)

            gr.Markdown("---")

            with gr.Row():
                # Left column - inputs
                with gr.Column(scale=1):
                    gen_procedure = gr.Dropdown(
                        label="Procedure Type",
                        choices=PROCEDURE_OPTIONS,
                        value="Laparoscopic Cholecystectomy",
                        allow_custom_value=True
                    )

                    with gr.Row():
                        gen_preop = gr.Textbox(label="Preoperative Diagnosis", lines=2)
                        gen_postop = gr.Textbox(label="Postoperative Diagnosis", lines=2)

                    with gr.Row():
                        gen_surgeon = gr.Textbox(label="Surgeon", placeholder="Dr. ")
                        gen_assistant = gr.Textbox(label="Assistant", placeholder="(optional)")

                    gen_anesthesia = gr.Dropdown(
                        label="Anesthesia Type",
                        choices=ANESTHESIA_OPTIONS,
                        value="General endotracheal"
                    )

                    gen_indications = gr.Textbox(
                        label="Indications",
                        placeholder="Brief history and reason for surgery...",
                        lines=3
                    )

                    gen_findings = gr.Textbox(
                        label="Intraoperative Findings",
                        placeholder="What was found during surgery...",
                        lines=3
                    )

                    gen_details = gr.Textbox(
                        label="Procedure Details / Key Steps",
                        placeholder="Important steps, techniques used, structures identified...",
                        lines=5
                    )

                    with gr.Row():
                        gen_specimens = gr.Textbox(label="Specimens", value="None")
                        gen_drains = gr.Textbox(label="Drains", value="None")

                    with gr.Row():
                        gen_ebl = gr.Textbox(label="EBL", value="Minimal")
                        gen_complications = gr.Textbox(label="Complications", value="None")

                    generate_btn = gr.Button("Generate Report", variant="primary", size="lg")

                # Right column - output
                with gr.Column(scale=1):
                    generated_report = gr.Textbox(
                        label="Generated Operative Report",
                        lines=30,
                        interactive=False
                    )

                    export_btn = gr.Button("Export to Word (.docx)")
                    export_file = gr.File(label="Download Report")

            # Wire up generate button
            generate_btn.click(
                fn=generate_report_handler,
                inputs=[
                    gen_procedure, gen_preop, gen_postop,
                    gen_surgeon, gen_assistant, gen_anesthesia,
                    gen_indications, gen_findings, gen_details,
                    gen_specimens, gen_drains, gen_ebl, gen_complications
                ],
                outputs=[generated_report]
            )

            # Wire up export button
            export_btn.click(
                fn=export_report_handler,
                inputs=[generated_report],
                outputs=[export_file]
            )

            # Wire up extract button to fill form fields
            extract_btn.click(
                fn=extract_from_brief_note,
                inputs=[import_text, import_file],
                outputs=[
                    gen_procedure, gen_preop, gen_postop,
                    gen_surgeon, gen_assistant, gen_anesthesia,
                    gen_indications, gen_findings, gen_details,
                    gen_specimens, gen_drains, gen_ebl, gen_complications,
                    extract_status
                ]
            )


# Launch app
if __name__ == "__main__":
    print("\nStarting Gradio app...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False
    )
