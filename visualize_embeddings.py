import ast  # For safely evaluating string representations of lists
import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# --- Configuration ---
INPUT_CSV_PATH = "hackathon_data_with_embeddings.csv"
# Columns that had text and now have corresponding embedding columns
# This should match the COLUMNS_TO_EMBED from the previous script
ORIGINAL_TEXT_COLUMNS = [
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
OUTPUT_HTML_DIR = "tsne_visualizations"  # Directory to save HTML files

# t-SNE Parameters (can be tuned)
TSNE_PERPLEXITY = 30  # Typical values: 5-50. Lower for smaller datasets.
TSNE_N_ITER = 1000  # Number of iterations for optimization.
TSNE_LEARNING_RATE = "auto"  # Recommended starting point for scikit-learn >= 1.2
TSNE_INIT = "pca"  # 'pca' can be more stable and faster than 'random'
RANDOM_STATE = 42  # For reproducibility


# --- Helper function to parse stringified embeddings ---
def parse_embedding_string(embedding_str):
    """
    Safely parses a string representation of a list into a Python list.
    Returns None if parsing fails or input is not a valid string.
    """
    if (
        pd.isna(embedding_str)
        or not isinstance(embedding_str, str)
        or not embedding_str.startswith("[")
    ):
        return None
    try:
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):
        print(
            f"Warning: Could not parse embedding string: {embedding_str[:100]}..."
        )  # Print first 100 chars
        return None


# --- Main Script ---
def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
    print(f"HTML visualizations will be saved in: '{OUTPUT_HTML_DIR}/'")

    # Load the data with embeddings
    try:
        print(f"Loading data from {INPUT_CSV_PATH}...")
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(
            f"Error: File not found at {INPUT_CSV_PATH}. Please run the embedding script first."
        )
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    for original_col_name in ORIGINAL_TEXT_COLUMNS:
        embedding_col_name = f"{original_col_name}_embedding"
        print(
            f"\nProcessing: '{original_col_name}' (embeddings in '{embedding_col_name}')"
        )

        if embedding_col_name not in df.columns:
            print(
                f"  Warning: Embedding column '{embedding_col_name}' not found. Skipping."
            )
            continue

        if original_col_name not in df.columns:
            print(
                f"  Warning: Original text column '{original_col_name}' not found for hover data. Skipping hover text for this plot."
            )
            # We can still proceed without hover text if necessary, or skip the plot
            # For now, let's try to proceed and hover_data will be handled by Plotly
            # (it might just show indices or nothing for that hover item)

        # Parse the string representations of embeddings into actual lists of floats
        # And prepare data for t-SNE
        embeddings_list = []
        hover_texts = []

        # Create a temporary DataFrame with just the original text and its embedding
        # This helps manage NaNs and filtering more easily
        temp_df = df[[original_col_name, embedding_col_name]].copy()
        temp_df["parsed_embedding"] = temp_df[embedding_col_name].apply(
            parse_embedding_string
        )

        # Filter out rows where parsing failed or embeddings are not suitable
        # We need actual list-like embeddings for np.array
        valid_embeddings_df = temp_df.dropna(subset=["parsed_embedding"])
        valid_embeddings_df = valid_embeddings_df[
            valid_embeddings_df["parsed_embedding"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
        ]

        if valid_embeddings_df.empty:
            print(
                f"  No valid embeddings found for '{original_col_name}' after parsing. Skipping t-SNE."
            )
            continue

        embeddings_list = valid_embeddings_df["parsed_embedding"].tolist()
        hover_texts = (
            valid_embeddings_df[original_col_name].fillna("N/A").tolist()
        )  # Use original text for hover

        # Convert list of embeddings to a NumPy array
        embedding_matrix = np.array(embeddings_list)

        if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] == 0:
            print(
                f"  Warning: Embedding matrix for '{original_col_name}' is not valid (shape: {embedding_matrix.shape}). Skipping."
            )
            continue

        # Check if we have enough samples for the chosen perplexity
        # t-SNE perplexity should be less than the number of samples.
        # Typically n_samples > 3 * perplexity.
        # A simpler check: n_samples > perplexity.
        num_samples = embedding_matrix.shape[0]
        current_perplexity = TSNE_PERPLEXITY
        if num_samples <= current_perplexity:
            # Adjust perplexity if too few samples.
            # Perplexity must be at least 1 and less than n_samples.
            current_perplexity = max(1, num_samples - 1)  # Max ensures it's at least 1
            if num_samples <= 1:  # if only 1 or 0 samples, t-SNE can't run
                print(
                    f"  Only {num_samples} valid sample(s) for '{original_col_name}'. Cannot run t-SNE. Skipping."
                )
                continue
            print(
                f"  Adjusted t-SNE perplexity to {current_perplexity} due to low sample size ({num_samples})."
            )

        if num_samples < 5:  # Arbitrary small number, t-SNE might not be meaningful
            print(
                f"  Very few samples ({num_samples}) for '{original_col_name}'. t-SNE results might not be meaningful."
            )

        print(f"  Running t-SNE for '{original_col_name}' on {num_samples} samples...")
        try:
            tsne = TSNE(
                n_components=2,
                random_state=RANDOM_STATE,
                perplexity=current_perplexity,
                n_iter=TSNE_N_ITER,
                learning_rate=TSNE_LEARNING_RATE,
                init=TSNE_INIT,
            )
            tsne_results = tsne.fit_transform(embedding_matrix)
        except Exception as e:
            print(f"  Error during t-SNE for '{original_col_name}': {e}. Skipping.")
            continue

        # Create a DataFrame for Plotly
        plot_df = pd.DataFrame(
            {
                "tsne_1": tsne_results[:, 0],
                "tsne_2": tsne_results[:, 1],
                "text": hover_texts,  # Use the filtered hover_texts
            }
        )

        # Create the interactive scatter plot
        print(f"  Generating Plotly graph for '{original_col_name}'...")
        fig = px.scatter(
            plot_df,
            x="tsne_1",
            y="tsne_2",
            hover_data=["text"],  # Show the original text on hover
            title=f"t-SNE visualization of embeddings for: {original_col_name}",
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(hovermode="closest")

        # Save the plot as an HTML file
        file_name = (
            f"tsne_plot_{original_col_name.replace(' ', '_').replace('/', '_')}.html"
        )
        output_path = os.path.join(OUTPUT_HTML_DIR, file_name)
        try:
            fig.write_html(output_path)
            print(f"  Successfully saved: {output_path}")
        except Exception as e:
            print(f"  Error saving HTML for '{original_col_name}': {e}")

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
