"""
RAG Engine for medical report retrieval using ChromaDB and sentence-transformers.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from database import get_all_reports, get_report

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_reports"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine with ChromaDB and sentence-transformers."""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Medical transcription reports"}
        )

    def add_report(
        self,
        report_id: int,
        report_text: str,
        procedure_type: str,
        specialty: str
    ) -> None:
        """
        Add a report to the ChromaDB collection.

        Args:
            report_id: Unique identifier for the report
            report_text: Full text of the medical report
            procedure_type: Type of procedure
            specialty: Medical specialty
        """
        # Generate embedding
        embedding = self.embedding_model.encode(report_text).tolist()

        # Add to collection
        self.collection.add(
            ids=[str(report_id)],
            embeddings=[embedding],
            documents=[report_text],
            metadatas=[{
                "procedure_type": procedure_type,
                "specialty": specialty,
                "report_id": report_id
            }]
        )

    def add_single_report(
        self,
        report_id: int,
        report_text: str,
        procedure_type: str,
        specialty: str
    ) -> None:
        """
        Add a single new report without rebuilding everything.
        Wrapper around add_report for clarity.

        Args:
            report_id: Unique identifier for the report
            report_text: Full text of the medical report
            procedure_type: Type of procedure
            specialty: Medical specialty
        """
        self.add_report(report_id, report_text, procedure_type, specialty)

    def delete_report(self, report_id: int) -> bool:
        """
        Remove a report from ChromaDB.

        Args:
            report_id: ID of the report to remove

        Returns:
            True if deletion was attempted
        """
        try:
            self.collection.delete(ids=[str(report_id)])
            return True
        except Exception:
            return False

    def search_similar(
        self,
        query: str,
        n_results: int = 3,
        specialty_filter: Optional[str] = None,
        procedure_filter: Optional[str] = None
    ) -> List[str]:
        """
        Search for similar reports based on a query.

        Args:
            query: Search query text
            n_results: Number of results to return
            specialty_filter: Optional filter by specialty
            procedure_filter: Optional filter by procedure type

        Returns:
            List of matching report texts
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build where filter if specified
        where_filter = None
        if specialty_filter or procedure_filter:
            conditions = []
            if specialty_filter:
                conditions.append({"specialty": specialty_filter})
            if procedure_filter:
                conditions.append({"procedure_type": procedure_filter})

            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Return documents
        if results and results['documents']:
            return results['documents'][0]
        return []

    def get_relevant_context(
        self,
        procedure_type: str,
        findings: str,
        n_results: int = 3
    ) -> str:
        """
        Get formatted context string for report generation.

        Args:
            procedure_type: Type of procedure being documented
            findings: Clinical findings to match against
            n_results: Number of similar reports to include

        Returns:
            Formatted context string with similar reports
        """
        # Combine procedure type and findings for search
        query = f"{procedure_type}: {findings}"

        # Search for similar reports
        similar_reports = self.search_similar(query, n_results=n_results)

        if not similar_reports:
            return "No similar reports found in the database."

        # Format context string
        context_parts = [
            "=== SIMILAR MEDICAL REPORTS FOR REFERENCE ===\n"
        ]

        for i, report in enumerate(similar_reports, 1):
            # Truncate very long reports for context
            truncated = report[:3000] + "..." if len(report) > 3000 else report
            context_parts.append(f"--- Example Report {i} ---\n{truncated}\n")

        context_parts.append("=== END OF REFERENCE REPORTS ===")

        return "\n".join(context_parts)

    def rebuild_from_db(self) -> Dict[str, int]:
        """
        Rebuild the ChromaDB collection from all reports in the SQLite database.

        Returns:
            Dictionary with statistics about the rebuild
        """
        print("Starting rebuild from database...")

        # Delete existing collection and recreate
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Medical transcription reports"}
        )

        # Get all reports from database
        reports = get_all_reports(limit=10000)
        total = len(reports)
        print(f"Found {total} reports in database")

        # Track statistics
        stats = {
            "total_reports": total,
            "indexed": 0,
            "skipped": 0,
            "specialties": {},
            "procedure_types": {}
        }

        # Process in batches for efficiency
        batch_size = 100
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for i, report in enumerate(reports):
            report_text = report.get('report_text', '')
            if not report_text or not report_text.strip():
                stats["skipped"] += 1
                continue

            procedure_type = report.get('procedure_type', 'Unknown')
            specialty = report.get('specialty', 'Unknown')

            # Track stats
            stats["specialties"][specialty] = stats["specialties"].get(specialty, 0) + 1
            stats["procedure_types"][procedure_type] = stats["procedure_types"].get(procedure_type, 0) + 1

            # Generate embedding
            embedding = self.embedding_model.encode(report_text).tolist()

            # Add to batch
            batch_ids.append(str(report['id']))
            batch_embeddings.append(embedding)
            batch_documents.append(report_text)
            batch_metadatas.append({
                "procedure_type": procedure_type,
                "specialty": specialty,
                "report_id": report['id']
            })

            # Insert batch when full
            if len(batch_ids) >= batch_size:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                stats["indexed"] += len(batch_ids)
                print(f"  Indexed {stats['indexed']}/{total} reports...")

                # Clear batches
                batch_ids = []
                batch_embeddings = []
                batch_documents = []
                batch_metadatas = []

        # Insert remaining batch
        if batch_ids:
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            stats["indexed"] += len(batch_ids)

        print(f"Rebuild complete: {stats['indexed']} reports indexed")
        return stats

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "persist_directory": CHROMA_PERSIST_DIR
        }


if __name__ == "__main__":
    print("Initializing RAG Engine...")
    engine = RAGEngine()

    print("\nRebuilding index from database...")
    stats = engine.rebuild_from_db()

    print("\n" + "=" * 60)
    print("RAG ENGINE STATISTICS")
    print("=" * 60)

    print(f"\nTotal reports indexed: {stats['indexed']}")
    print(f"Reports skipped (empty): {stats['skipped']}")

    print("\nBy specialty:")
    print("-" * 40)
    for specialty, count in sorted(stats['specialties'].items()):
        print(f"  {specialty}: {count}")

    print(f"\nUnique procedure types: {len(stats['procedure_types'])}")

    # Test search
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    test_query = "laparoscopic cholecystectomy gallbladder removal"
    print(f"\nQuery: '{test_query}'")
    results = engine.search_similar(test_query, n_results=2)
    print(f"Found {len(results)} similar reports")

    if results:
        print("\nFirst result preview (first 500 chars):")
        print("-" * 40)
        print(results[0][:500] + "...")

    # Show collection stats
    print("\n" + "=" * 60)
    print("COLLECTION INFO")
    print("=" * 60)
    coll_stats = engine.get_collection_stats()
    for key, value in coll_stats.items():
        print(f"  {key}: {value}")
