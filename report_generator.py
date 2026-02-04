"""
Report Generator - AI-powered operative report generation using RAG and Ollama.
"""

from typing import Optional, Dict, Any, List
import ollama
from rag_engine import RAGEngine

# Default model for generation
DEFAULT_MODEL = "qwen2.5:32b"


class ReportGenerator:
    """
    Generates operative reports using RAG context and LLM generation.
    """

    def __init__(self, rag_engine: Optional[RAGEngine] = None, model: str = DEFAULT_MODEL):
        """
        Initialize the report generator.

        Args:
            rag_engine: RAGEngine instance for retrieving similar reports.
                        If None, a new instance will be created.
            model: Ollama model to use for generation (default: qwen2.5:32b)
        """
        self.rag_engine = rag_engine or RAGEngine()
        self.model = model

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM with the given messages.

        This method is isolated to make future swaps (e.g., TensorRT-LLM)
        a single-method change.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text from the LLM
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 4096,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: LLM generation failed - {str(e)}"

    def generate_report(
        self,
        procedure_type: str,
        preop_diagnosis: str,
        postop_diagnosis: str,
        surgeon_name: str,
        assistant: str = "",
        anesthesia_type: str = "General",
        indications: str = "",
        findings: str = "",
        procedure_details: str = "",
        specimens: str = "None",
        drains: str = "None",
        ebl: str = "Minimal",
        complications: str = "None",
        n_context_reports: int = 3
    ) -> str:
        """
        Generate a complete operative report based on surgeon inputs.

        Args:
            procedure_type: Type of procedure (e.g., "Laparoscopic Cholecystectomy")
            preop_diagnosis: Preoperative diagnosis
            postop_diagnosis: Postoperative diagnosis
            surgeon_name: Name of the operating surgeon
            assistant: Name of assistant surgeon (optional)
            anesthesia_type: Type of anesthesia used
            indications: Indications for the procedure
            findings: Intraoperative findings
            procedure_details: Key details about the procedure performed
            specimens: Specimens sent to pathology
            drains: Drains placed
            ebl: Estimated blood loss
            complications: Any complications encountered
            n_context_reports: Number of similar reports to retrieve for context

        Returns:
            Generated operative report text
        """
        # Get relevant context from RAG
        context = self.rag_engine.get_relevant_context(
            procedure_type=procedure_type,
            findings=f"{indications} {findings} {procedure_details}",
            n_results=n_context_reports
        )

        # Build the system prompt
        system_prompt = """You are an expert medical transcriptionist specializing in operative reports.
Your task is to generate a complete, professional operative report based on the surgeon's inputs.

CRITICAL REQUIREMENTS:
1. Use proper medical terminology throughout
2. Maintain a professional, formal tone appropriate for medical records
3. DO NOT use any placeholders, brackets, or fill-in-the-blank text (e.g., no [DATE], [TIME], etc.)
4. Generate complete sentences and paragraphs
5. Follow standard operative report structure and formatting
6. Include all provided information naturally in the report
7. If a field is marked "None" or empty, either omit it or state appropriately (e.g., "No drains were placed")

The report should include these sections in order:
- PREOPERATIVE DIAGNOSIS
- POSTOPERATIVE DIAGNOSIS
- PROCEDURE PERFORMED
- SURGEON / ASSISTANT
- ANESTHESIA
- INDICATIONS
- FINDINGS
- PROCEDURE IN DETAIL
- SPECIMENS
- DRAINS
- ESTIMATED BLOOD LOSS
- COMPLICATIONS
- DISPOSITION (patient condition at end)"""

        # Build the user prompt with surgeon inputs
        user_prompt = f"""Generate an operative report using the following information:

SURGEON INPUTS:
- Procedure Type: {procedure_type}
- Preoperative Diagnosis: {preop_diagnosis}
- Postoperative Diagnosis: {postop_diagnosis}
- Surgeon: {surgeon_name}
- Assistant: {assistant if assistant else "None"}
- Anesthesia: {anesthesia_type}
- Indications: {indications if indications else "As per diagnosis"}
- Findings: {findings if findings else "As expected for the diagnosis"}
- Procedure Details: {procedure_details}
- Specimens: {specimens}
- Drains: {drains}
- Estimated Blood Loss: {ebl}
- Complications: {complications}

{context}

Based on the surgeon inputs above and using the reference reports for formatting and structure guidance, generate a complete operative report. Write the full report now:"""

        # Build messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate the report
        return self._call_llm(messages)

    def generate_report_from_dict(self, inputs: Dict[str, Any]) -> str:
        """
        Generate a report from a dictionary of inputs.

        Convenience method for use with forms/APIs.

        Args:
            inputs: Dictionary containing report inputs

        Returns:
            Generated operative report text
        """
        return self.generate_report(
            procedure_type=inputs.get('procedure_type', ''),
            preop_diagnosis=inputs.get('preop_diagnosis', ''),
            postop_diagnosis=inputs.get('postop_diagnosis', ''),
            surgeon_name=inputs.get('surgeon_name', ''),
            assistant=inputs.get('assistant', ''),
            anesthesia_type=inputs.get('anesthesia_type', 'General'),
            indications=inputs.get('indications', ''),
            findings=inputs.get('findings', ''),
            procedure_details=inputs.get('procedure_details', ''),
            specimens=inputs.get('specimens', 'None'),
            drains=inputs.get('drains', 'None'),
            ebl=inputs.get('ebl', 'Minimal'),
            complications=inputs.get('complications', 'None'),
            n_context_reports=inputs.get('n_context_reports', 3)
        )


if __name__ == "__main__":
    print("Report Generator - Test")
    print("=" * 60)

    print("\nInitializing RAG Engine...")
    rag_engine = RAGEngine()

    print("Initializing Report Generator...")
    generator = ReportGenerator(rag_engine=rag_engine)

    print("\nGenerating sample cholecystectomy report...")
    print("=" * 60)

    # Sample cholecystectomy case
    report = generator.generate_report(
        procedure_type="Laparoscopic Cholecystectomy",
        preop_diagnosis="Symptomatic cholelithiasis with biliary colic",
        postop_diagnosis="Symptomatic cholelithiasis with chronic cholecystitis",
        surgeon_name="Dr. Smith",
        assistant="Dr. Johnson",
        anesthesia_type="General endotracheal anesthesia",
        indications="Patient is a 45-year-old female with recurrent right upper quadrant pain and ultrasound-confirmed gallstones. Failed conservative management.",
        findings="Gallbladder was distended with omental adhesions. Multiple faceted stones palpable. Cystic duct and artery clearly identified. No common bile duct stones appreciated.",
        procedure_details="Standard 4-port technique. Critical view of safety achieved. Cystic artery and duct clipped and divided. Gallbladder dissected from liver bed using electrocautery. Removed via umbilical port in specimen bag.",
        specimens="Gallbladder with stones sent to pathology",
        drains="None",
        ebl="Less than 20 mL",
        complications="None"
    )

    print("\nGENERATED REPORT:")
    print("-" * 60)
    print(report)
    print("-" * 60)

    print("\nTest complete.")
