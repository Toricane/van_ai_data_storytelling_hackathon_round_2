import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
CSV_FILE_PATH = "Hackathon Round 2_Canada Survey Data.csv"
OUTPUT_CSV_PATH = "hackathon_data_with_embeddings.csv"
COLUMNS_TO_EMBED = [
    "Q5b_Text",
    "Q6_Text",
    "Q7_Other",
    "Q8_Text",
    "Q9_divisions_other",
    "Q10_Other",
    "Q11_Text",
    "Q12_Future_hope",
    "Q13_Future_fear",
]
EMBEDDING_MODEL = "text-embedding-3-large"  # Or "text-embedding-3-small" or "text-embedding-ada-002" for different cost/performance
# Note: text-embedding-3-large default dimensions: 3072. Max input tokens: 8191.


# --- Helper Function for API Calls with Retries ---
def get_embeddings_with_retry(client, texts, model, max_retries=5, delay_seconds=5):
    """
    Gets embeddings for a list of texts with retry logic for API errors.
    OpenAI API can handle lists of texts (batching).
    """
    for attempt in range(max_retries):
        try:
            # Replace None or non-string values with empty strings
            processed_texts = [
                str(text) if pd.notnull(text) and text != "" else " " for text in texts
            ]
            # If all texts are effectively empty after processing, return list of Nones or empty lists
            if not any(t.strip() for t in processed_texts):
                print(
                    f"Warning: All texts in batch are empty or whitespace. Returning list of empty embeddings."
                )
                return [[] for _ in texts]  # Or [None for _ in texts]

            response = client.embeddings.create(input=processed_texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Max retries reached. Raising exception.")
                raise
    return [None] * len(texts)  # Should not be reached if exception is raised


# --- Main Script ---
def main():
    # Load environment variables (for API key)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with OPENAI_API_KEY='your_key_here'")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Load CSV data
    try:
        print(f"Loading data from {CSV_FILE_PATH}...")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Process each specified column for embeddings
    for col_name in COLUMNS_TO_EMBED:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in the CSV. Skipping.")
            continue

        print(f"\nProcessing column: '{col_name}' for embeddings...")

        # Extract text data from the column
        # Handle NaN values by replacing them with empty strings, as embedding models expect strings.
        # An empty string will typically result in a "null" or near-zero embedding.
        texts_to_embed = df[col_name].fillna("").tolist()

        # Check if there's anything to embed (e.g., column might be all NaNs)
        if not texts_to_embed or all(s == "" for s in texts_to_embed):
            print(
                f"Column '{col_name}' contains no text data to embed. Skipping API call."
            )
            df[f"{col_name}_embedding"] = [
                [] for _ in range(len(df))
            ]  # Fill with empty lists
            continue

        print(
            f"Generating embeddings for {len(texts_to_embed)} texts in '{col_name}' using '{EMBEDDING_MODEL}'..."
        )

        try:
            # The OpenAI API can handle a list of texts directly (batching)
            embeddings = get_embeddings_with_retry(
                client, texts_to_embed, model=EMBEDDING_MODEL
            )
            df[f"{col_name}_embedding"] = embeddings
            print(f"Embeddings generated for '{col_name}'.")
            if embeddings and embeddings[0]:
                print(f"Dimension of first embedding: {len(embeddings[0])}")
        except Exception as e:
            print(f"Failed to generate embeddings for column '{col_name}': {e}")
            # Add a column with None or empty lists to maintain DataFrame structure
            df[f"{col_name}_embedding"] = [None] * len(df)

    # Save the DataFrame with embeddings to a new CSV
    try:
        print(f"\nSaving data with embeddings to {OUTPUT_CSV_PATH}...")
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Successfully saved to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


if __name__ == "__main__":
    main()
