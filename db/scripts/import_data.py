import os
import zipfile
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters from .env
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "vectordb")

"""
Expects a zipfile in a subfolder ../data/rag-data.zip.
If present, looks through for a csv for each target table type
relative to this project. "projects.csv", "files.csv", "chunks.csv"
and imports them to the tables. 

A simple synchronization mechanism across developer machines
or even deployments (if the data is small enough)
"""
def import_data():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Define table import order to respect foreign keys
    table_order = ['projects', 'files', 'chunks']

    # Extract and process zip file
    with zipfile.ZipFile('../data/rag-data.zip', 'r') as zip_ref:
        # Process files in specific order
        for table_name in table_order:
            filename = f"{table_name}.csv"
            if filename not in zip_ref.namelist() or filename.startswith('__'):
                print(f"Skipping {filename} - not found in zip")
                continue

            print(f"Processing {filename}")
            table_name = os.path.splitext(filename)[0]

            # Truncate the table first
            try:
                cur.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
                conn.commit()
                print(f"Truncated table {table_name}")
            except psycopg2.Error as e:
                print(f"Error truncating {table_name}: {str(e)}")
                conn.rollback()
                continue

            try:
                # Special handling for projects table if it doesn't exist in zip
                if table_name == 'projects' and filename not in zip_ref.namelist():
                    print("Creating default project...")
                    cur.execute("""
                        INSERT INTO projects (id, name, description, created_at, updated_at)
                        VALUES (1, 'Default Project', 'Default project for data import', 
                               CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING;
                    """)
                else:
                    # Read the CSV data directly from the zip
                    with zip_ref.open(filename) as file:
                        # Skip the header line
                        file.readline()
                        # Use copy_expert with file-like object
                        cur.copy_expert(
                            f"COPY {table_name} FROM STDIN WITH (FORMAT csv)",
                            file
                        )
                conn.commit()
                print(f"Successfully imported data into {table_name}")
            except psycopg2.Error as e:
                print(f"Error importing into {table_name}: {str(e)}")
                conn.rollback()

    cur.close()
    conn.close()


if __name__ == "__main__":
    import_data()
