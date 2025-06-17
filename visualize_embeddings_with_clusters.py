import ast
import os
import textwrap  # For wrapping hover text

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --- Configuration ---
INPUT_CSV_PATH = "hackathon_data_with_embeddings.csv"
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
OUTPUT_HTML_DIR = "tsne_visualizations_clustered"

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_LEARNING_RATE = "auto"
TSNE_INIT = "pca"
RANDOM_STATE = 42

# Clustering parameters
KMEANS_MAX_K = 10
KMEANS_MIN_K = 2

# Hover text parameters
HOVER_MAX_LINE_WIDTH = 60  # Max characters per line in hover text


# --- Helper functions ---
def parse_embedding_string(embedding_str):
    if (
        pd.isna(embedding_str)
        or not isinstance(embedding_str, str)
        or not embedding_str.startswith("[")
    ):
        return None
    try:
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):
        return None


def wrap_text_for_hover(text, width):
    """Wraps text with <br> for Plotly hover, if text is a string."""
    if not isinstance(text, str):
        return str(
            text
        )  # Convert non-strings to string, though ideally text is already string
    return "<br>".join(textwrap.wrap(text, width=width))


# --- Main Script ---
def main():
    os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)
    print(f"HTML visualizations will be saved in: '{OUTPUT_HTML_DIR}/'")

    try:
        df_full = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    for original_col_name in ORIGINAL_TEXT_COLUMNS:
        embedding_col_name = f"{original_col_name}_embedding"
        print(
            f"\nProcessing: '{original_col_name}' (embeddings in '{embedding_col_name}')"
        )

        if (
            original_col_name not in df_full.columns
            or embedding_col_name not in df_full.columns
        ):
            print(
                f"  Warning: Missing original text or embedding column for '{original_col_name}'. Skipping."
            )
            continue

        df_question = df_full[[original_col_name, embedding_col_name]].copy()
        df_question["text_is_valid"] = df_question[original_col_name].apply(
            lambda x: isinstance(x, str)
            and x.strip() != ""
            and x.strip().lower() != "n/a"
        )
        df_filtered_text = df_question[df_question["text_is_valid"]].copy()

        if df_filtered_text.empty:
            print(
                f"  No valid text entries for '{original_col_name}' after filtering. Skipping."
            )
            continue

        df_filtered_text["parsed_embedding"] = df_filtered_text[
            embedding_col_name
        ].apply(parse_embedding_string)
        df_valid_embeddings = df_filtered_text.dropna(subset=["parsed_embedding"])
        df_valid_embeddings = df_valid_embeddings[
            df_valid_embeddings["parsed_embedding"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
        ]

        if df_valid_embeddings.empty:
            print(
                f"  No valid embeddings found for '{original_col_name}' after parsing. Skipping."
            )
            continue

        embedding_matrix = np.array(df_valid_embeddings["parsed_embedding"].tolist())

        # Prepare original texts (unwrapped) and wrapped texts for hover
        original_hover_texts = df_valid_embeddings[original_col_name].tolist()
        wrapped_hover_texts = [
            wrap_text_for_hover(text, HOVER_MAX_LINE_WIDTH)
            for text in original_hover_texts
        ]

        num_samples = embedding_matrix.shape[0]

        if num_samples < 5:
            print(
                f"  Too few valid samples ({num_samples}) for '{original_col_name}'. Skipping t-SNE and clustering."
            )
            continue

        current_perplexity = TSNE_PERPLEXITY
        if num_samples <= current_perplexity:
            current_perplexity = max(1, num_samples - 1)
            print(
                f"  Adjusted t-SNE perplexity to {current_perplexity} for {num_samples} samples."
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

        # --- Clustering on t-SNE results ---
        plot_df_data = {
            "tsne_1": tsne_results[:, 0],
            "tsne_2": tsne_results[:, 1],
            "original_text": original_hover_texts,  # Keep original for other uses if needed
            "wrapped_text_hover": wrapped_hover_texts,  # For hover display
        }

        best_k = 1
        cluster_column_for_color = (
            None  # Name of the column used for coloring in px.scatter
        )
        cluster_display_labels = ["All Points"] * num_samples  # Default for hover
        title_suffix = "(single cluster)"

        if (
            num_samples >= KMEANS_MIN_K * 2
        ):  # Need enough points to form at least KMEANS_MIN_K clusters
            max_k_to_test = min(KMEANS_MAX_K, num_samples - 1)
            if max_k_to_test >= KMEANS_MIN_K:
                print(
                    f"  Determining optimal k (from {KMEANS_MIN_K} to {max_k_to_test}) using Silhouette Score..."
                )
                silhouette_scores = {}
                for k_candidate in range(KMEANS_MIN_K, max_k_to_test + 1):
                    kmeans = KMeans(
                        n_clusters=k_candidate, random_state=RANDOM_STATE, n_init="auto"
                    )
                    try:
                        cluster_labels_candidate = kmeans.fit_predict(tsne_results)
                        if (
                            len(set(cluster_labels_candidate)) > 1
                            and len(set(cluster_labels_candidate)) < num_samples
                        ):
                            score = silhouette_score(
                                tsne_results, cluster_labels_candidate
                            )
                            silhouette_scores[k_candidate] = score
                    except Exception:  # Ignore errors for specific k during search
                        pass

                if silhouette_scores:
                    best_k = max(silhouette_scores, key=silhouette_scores.get)
                    print(
                        f"  Best k determined: {best_k} (Silhouette Score: {silhouette_scores[best_k]:.4f})"
                    )
                else:
                    best_k = 1
                    print(
                        f"  Could not determine optimal k via Silhouette. Defaulting to k={best_k}."
                    )
            else:  # Not enough points to test range, so k=1
                best_k = 1
                print(
                    f"  Not enough samples to test a range of clusters. Defaulting to k=1."
                )

        if best_k > 1 and num_samples >= best_k:
            kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
            final_cluster_labels = kmeans.fit_predict(tsne_results)
            plot_df_data["cluster_label"] = [
                f"Cluster {label}" for label in final_cluster_labels
            ]
            cluster_column_for_color = "cluster_label"  # Use this column for coloring
            cluster_display_labels = plot_df_data["cluster_label"]  # For hover
            title_suffix = f"(k={best_k} clusters)"

        plot_df_data["cluster_display_hover"] = cluster_display_labels
        plot_df = pd.DataFrame(plot_df_data)

        print(f"  Generating Plotly graph for '{original_col_name}'...")
        fig = px.scatter(
            plot_df,
            x="tsne_1",
            y="tsne_2",
            color=cluster_column_for_color,  # Will be 'cluster_label' or None
            custom_data=[
                "wrapped_text_hover",
                "cluster_display_hover",
            ],  # Data for hovertemplate
        )

        # Custom hovertemplate
        # %{customdata[0]} refers to the first item in custom_data list (wrapped_text_hover)
        # %{customdata[1]} refers to the second item (cluster_display_hover)
        # <extra></extra> removes the trace information box that Plotly adds by default
        fig.update_traces(
            hovertemplate="<b>Text:</b><br>%{customdata[0]}<br><br>"
            + "<b>Cluster:</b> %{customdata[1]}"
            + "<extra></extra>"
        )

        fig.update_layout(
            title=f"t-SNE: {original_col_name} {title_suffix}",
            hovermode="closest",
            legend_title_text="Cluster ID" if cluster_column_for_color else "",
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))

        file_name = f"tsne_clustered_{original_col_name.replace(' ', '_').replace('/', '_')}.html"
        output_path = os.path.join(OUTPUT_HTML_DIR, file_name)
        try:
            fig.write_html(output_path)
            print(f"  Successfully saved: {output_path}")
        except Exception as e:
            print(f"  Error saving HTML for '{original_col_name}': {e}")

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
