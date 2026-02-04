"""
Script to load MTSamples data into the reports database.
"""

import pandas as pd
from collections import Counter
from database import init_db, add_report

# Target specialties to filter
TARGET_SPECIALTIES = {'Surgery', 'General Surgery', 'Gastroenterology'}

def load_mtsamples():
    """Load MTSamples data into the database."""

    # Initialize database
    print("Initializing database...")
    init_db()

    # Read CSV
    print("Reading mtsamples.csv...")
    df = pd.read_csv('mtsamples.csv')

    print(f"Total records in CSV: {len(df)}")

    # Strip whitespace from medical_specialty column
    df['medical_specialty'] = df['medical_specialty'].str.strip()

    # Filter to target specialties
    df_filtered = df[df['medical_specialty'].isin(TARGET_SPECIALTIES)].copy()
    print(f"Records matching target specialties: {len(df_filtered)}")

    # Drop rows where transcription is empty/null
    df_filtered = df_filtered.dropna(subset=['transcription'])
    df_filtered = df_filtered[df_filtered['transcription'].str.strip() != '']
    print(f"Records after removing empty transcriptions: {len(df_filtered)}")

    # Track statistics
    specialty_counts = Counter()
    procedure_counts = Counter()
    loaded_count = 0

    # Insert each record
    print("\nLoading records into database...")
    for _, row in df_filtered.iterrows():
        specialty = row['medical_specialty']
        procedure_type = row['description'] if pd.notna(row['description']) else 'Unknown'
        report_name = row['sample_name'] if pd.notna(row['sample_name']) else None
        report_text = row['transcription']
        keywords = row['keywords'] if pd.notna(row['keywords']) else None

        add_report(
            procedure_type=procedure_type.strip() if isinstance(procedure_type, str) else procedure_type,
            specialty=specialty,
            report_text=report_text,
            source='MTSamples/Kaggle',
            report_name=report_name.strip() if isinstance(report_name, str) and report_name else report_name,
            keywords=keywords.strip() if isinstance(keywords, str) else keywords,
            is_deidentified=True
        )

        specialty_counts[specialty] += 1
        procedure_counts[procedure_type.strip() if isinstance(procedure_type, str) else procedure_type] += 1
        loaded_count += 1

    # Print summary
    print("\n" + "=" * 60)
    print("LOADING COMPLETE")
    print("=" * 60)

    print(f"\nTotal records loaded: {loaded_count}")

    print("\nBreakdown by specialty:")
    print("-" * 40)
    for specialty, count in sorted(specialty_counts.items()):
        print(f"  {specialty}: {count}")

    print("\nTop 10 procedure types:")
    print("-" * 40)
    for procedure, count in procedure_counts.most_common(10):
        # Truncate long procedure descriptions
        display_proc = procedure[:60] + "..." if len(str(procedure)) > 60 else procedure
        print(f"  {display_proc}: {count}")


if __name__ == "__main__":
    load_mtsamples()
