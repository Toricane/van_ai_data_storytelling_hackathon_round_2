# Vancouver AI Data Storytelling Hackathon Round 2: What's Next for Canada?

<iframe src="tsne_visualizations_3d_silhouette_focus_sized/tsne_3d_silhouette_sized_Q6_Text.html" width="100%" height="600px"></iframe>

### Visualization of Hackathon Data with Embeddings

> [!NOTE]
> The final visualizations are in the [`tsne_visualizations_3d_silhouette_focus_sized/`](tsne_visualizations_3d_silhouette_focus_sized/) directory.
> The code used to generate those specific visualizations is in [`visualizations_3d_silhouette_focus_sized.py`](visualizations_3d_silhouette_focus_sized.py).

1. [`embeddings_of_data.py`](embeddings_of_data.py) - Generates embeddings for hackathon data and saves it to [`hackathon_data_with_embeddings.csv`](hackathon_data_with_embeddings.csv). This file contains the original data along with the generated embeddings for each entry.
2. [`visualize_embeddings.py`](visualize_embeddings.py) - Visualizes the embeddings using t-SNE and saves the output as HTML files in [`tsne_visualizations/`](tsne_visualizations/). This script provides a basic visualization of the embeddings without clustering.
3. [`visualize_embeddings_with_clusters.py`](visualize_embeddings_with_clusters.py) - Visualizes the embeddings with clustering and saves the output as HTML files in [`tsne_visualizations_clustered/`](tsne_visualizations_clustered/). This script enhances the visualization by grouping similar embeddings together, making it easier to identify clusters in the data.
4. [`visualize_embeddings_3d_clustered.py`](visualize_embeddings_3d_clustered.py) - Visualizes the embeddings in 3D with clustering and saves the output as HTML files in [`tsne_visualizations_3d_clustered/`](tsne_visualizations_3d_clustered/). This script provides a three-dimensional perspective of the clustered embeddings, allowing for a more comprehensive view of the data distribution.
5. [`visualizations_3d_silhouette_focus.py`](visualizations_3d_silhouette_focus.py) - Creates 3D visualizations focusing on silhouette scores and saves the output as HTML files in [`tsne_visualizations_3d_silhouette_focus/`](tsne_visualizations_3d_silhouette_focus/). This script emphasizes the silhouette scores of clusters, providing insights into the quality of clustering.
6. [`visualizations_3d_silhouette_focus_sized.py`](visualizations_3d_silhouette_focus_sized.py) - Similar to the previous script but with size variations, saving output in [`tsne_visualizations_3d_silhouette_focus_sized/`](tsne_visualizations_3d_silhouette_focus_sized/). This script enhances the 3D visualization by adjusting the size of points based on their text frequency.
