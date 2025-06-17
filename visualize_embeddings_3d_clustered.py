import ast
import os
import textwrap

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
OUTPUT_HTML_DIR = "tsne_visualizations_3d_clustered"  # Updated for 3D

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_LEARNING_RATE = "auto"  # Changed from 200 to 'auto' for scikit-learn >= 1.2
TSNE_INIT = "pca"
RANDOM_STATE = 42

# Clustering parameters
KMEANS_MAX_K = 10
KMEANS_MIN_K = 2

# Hover text parameters
HOVER_MAX_LINE_WIDTH = 60


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
    if not isinstance(text, str):
        return str(text)
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
        original_hover_texts = df_valid_embeddings[original_col_name].tolist()
        wrapped_hover_texts = [
            wrap_text_for_hover(text, HOVER_MAX_LINE_WIDTH)
            for text in original_hover_texts
        ]
        num_samples = embedding_matrix.shape[0]

        if num_samples < 5:  # Need at least a few points for t-SNE/clustering
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

        print(
            f"  Running 3D t-SNE for '{original_col_name}' on {num_samples} samples..."
        )
        try:
            tsne = TSNE(
                n_components=3,  # CHANGED TO 3 for 3D
                random_state=RANDOM_STATE,
                perplexity=current_perplexity,
                n_iter=TSNE_N_ITER,
                learning_rate=TSNE_LEARNING_RATE,
                init=TSNE_INIT,
            )
            tsne_results = tsne.fit_transform(embedding_matrix)
        except Exception as e:
            print(f"  Error during 3D t-SNE for '{original_col_name}': {e}. Skipping.")
            continue

        plot_df_data = {
            "tsne_1": tsne_results[:, 0],
            "tsne_2": tsne_results[:, 1],
            "tsne_3": tsne_results[:, 2],  # ADDED 3rd dimension
            "original_text": original_hover_texts,
            "wrapped_text_hover": wrapped_hover_texts,
        }

        best_k = 1
        cluster_column_for_color = None
        cluster_display_labels = ["All Points"] * num_samples
        title_suffix = "(single cluster)"

        # Silhouette score requires at least 2 clusters and n_samples > n_clusters
        # We need at least KMEANS_MIN_K samples to try clustering.
        if (
            num_samples >= KMEANS_MIN_K * 2
        ):  # A bit more conservative for more stable silhouette
            max_k_to_test = min(
                KMEANS_MAX_K, num_samples - 1
            )  # k cannot be >= num_samples for silhouette
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
                        cluster_labels_candidate = kmeans.fit_predict(
                            tsne_results
                        )  # tsne_results is now 3D
                        if (
                            len(set(cluster_labels_candidate)) > 1
                            and len(set(cluster_labels_candidate)) < num_samples
                        ):
                            score = silhouette_score(
                                tsne_results, cluster_labels_candidate
                            )
                            silhouette_scores[k_candidate] = score
                    except Exception:
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
            else:
                best_k = 1
                print(
                    f"  Not enough samples ({num_samples}) to test a range of clusters. Defaulting to k=1."
                )

        if best_k > 1 and num_samples >= best_k:
            kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto")
            final_cluster_labels = kmeans.fit_predict(tsne_results)
            plot_df_data["cluster_label"] = [
                f"Cluster {label}" for label in final_cluster_labels
            ]
            cluster_column_for_color = "cluster_label"
            cluster_display_labels = plot_df_data["cluster_label"]
            title_suffix = f"(k={best_k} clusters)"

        plot_df_data["cluster_display_hover"] = cluster_display_labels
        plot_df = pd.DataFrame(plot_df_data)

        print(f"  Generating 3D Plotly graph for '{original_col_name}'...")
        fig = px.scatter_3d(  # CHANGED to scatter_3d
            plot_df,
            x="tsne_1",
            y="tsne_2",
            z="tsne_3",  # ADDED z dimension
            color=cluster_column_for_color,
            custom_data=["wrapped_text_hover", "cluster_display_hover"],
        )

        fig.update_traces(
            hovertemplate="<b>Text:</b><br>%{customdata[0]}<br><br>"
            + "<b>Cluster:</b> %{customdata[1]}"
            + "<extra></extra>",
            marker=dict(size=5, opacity=0.8),  # Adjusted size for 3D potentially
        )

        fig.update_layout(
            title=f"3D t-SNE: {original_col_name} {title_suffix}",  # Updated title
            hovermode="closest",
            legend_title_text="Cluster ID" if cluster_column_for_color else "",
            scene=dict(  # Optional: configure 3D scene appearance
                xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"
            ),
        )

        file_name = f"tsne_3d_clustered_{original_col_name.replace(' ', '_').replace('/', '_')}.html"  # Updated filename
        output_path = os.path.join(OUTPUT_HTML_DIR, file_name)
        try:
            fig.write_html(output_path)
            print(f"  Successfully saved: {output_path}")
        except Exception as e:
            print(f"  Error saving HTML for '{original_col_name}': {e}")

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()
