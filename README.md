# Vancouver AI Data Storytelling Hackathon Round 2: What's Next for Canada?

> 1,000 Canadians shared their thoughts on the state of the country and its future. The data is raw, real, and full of tension, nuance, and insight.

> Using a custom dataset gathered with RIVAL’s market research tech, your mission is to transform raw numbers into an experience that’s interactive, visual, or narrative-driven.

> You decide how to make the data speak—whether through dynamic visualizations, compelling storytelling, performative media, or unconventional experimentation.

View my project here: https://prajwal.is-a.dev/van_ai_data_storytelling_hackathon_round_2/

Demo video: [to be added]

> [!NOTE]
> The visuals may take a minute to fully load.

## Pitch

It's hard for us humans to understand long-response survey data. Sure, you could do some sentiment analysis and try to group them in a list, or get an LLM to summarize them, but that doesn't really help us understand the data. What if we could visualize the data in a way that makes it easier to holistically understand the responses, and understand how they relate to each other?

When I was learning about LLMs, I learned how neural networks process text data by converting it into embeddings, which are numerical representations of the text. These matrices of numbers show the semantic meaning of the text, and similar text will have similar embeddings. I thought it would be interesting to apply this technique to the hackathon survey data, and visualize the embeddings in a way that makes it easier to understand the relationships between the responses.

Imagine a 3D scatter plot where each point represents a survey response, and similar responses are clustered together. You could rotate the plot to see how the responses relate to each other, and even zoom in on specific clusters to see the individual responses. This would allow us to understand the data in a qualitative way which is not possible with traditional methods.

These semantic maps allow for a quick visual overview of the data, making it easier to identify patterns and relationships.

## Approach

Inspired by my previous work on https://prajwal.is-a.dev/student_survey/ ([GitHub repository](https://github.com/Toricane/student_survey)), which built 2D visualizations of student survey data, I decided to apply similar techniques to the Vancouver AI Data Storytelling Hackathon Round 2 data. The goal was to create engaging and informative visualizations that highlight key insights from the hackathon survey responses.

I vibe coded this project using Google's Gemini 2.5 Pro. Although I used Gemini to generate the code, it took a lot of trial and error to get the visualizations to look the way I wanted.

-   [Branch of CSV](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221ImYTuHWBhv2lLfe-0ZAMitZKGjAVxaJR%22%5D,%22action%22:%22open%22,%22userId%22:%22112146135410039036896%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)
-   [Canadian Survey Data Visualization](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2210RXVm2LfJUHA83yhapaW23ThgxPCifV8%22%5D,%22action%22:%22open%22,%22userId%22:%22112146135410039036896%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)

The technical approach can be broken down as follows:

1.  **Text Embedding Generation**: Each open-ended survey response was converted into a high-dimensional vector using OpenAI's `text-embedding-3-large` model. These embeddings capture the semantic meaning of the text, allowing for mathematical comparison. The augmented dataset was saved to a new CSV for efficiency.
2.  **Dimensionality Reduction**: To visualize the high-dimensional embeddings, the **t-SNE** (`t-distributed Stochastic Neighbor Embedding`) algorithm was used to reduce the data to three dimensions, preserving the local structure and clustering of the original data.
3.  **Optimal Cluster Identification**: For each question, **K-Means clustering** was applied to the 3D t-SNE coordinates. To programmatically determine the ideal number of clusters (`k`), the **Silhouette Score** was calculated for a range of `k` values. The `k` that yielded the highest score was selected, representing the most distinct and well-separated thematic grouping.
4.  **Aggregation and Volumetric Sizing**: To make the visualization cleaner and more insightful, responses with the exact same text were aggregated into a single point. The size of this point is calculated to be proportional to the **cube root of its frequency**, ensuring its _volume_ in the 3D space directly represents the number of people who gave that response. The point is positioned at the average coordinate of its constituents and colored by its most common cluster ID.
5.  **Interactive Visualization**: The final 3D scatter plots were generated using **Plotly**. This allows for a fully interactive experience where users can rotate, pan, and zoom. Hovering over a point reveals the original text, its cluster, and its frequency.

## Future Improvements

It is awkward to view a 3D visualization in a web browser, and it would be better to use a VR headset to view the visualizations. I would like to explore how to make the visualizations more interactive, such as allowing users to click on points to see the original survey response, or filtering the data based on specific criteria. Last, but not least, I would like to incorporate LLMs to generate summaries and insights from the data, which could be displayed alongside the visualizations to provide additional context and understanding, as well as guide the user to look at important data points.

## Repository Structure

### Visualization of Hackathon Data with Embeddings

> [!NOTE]
> The final visualizations are in the [`tsne_visualizations_3d_silhouette_focus_sized/`](tsne_visualizations_3d_silhouette_focus_sized/) directory.
> The code used to generate those specific visualizations is in [`visualizations_3d_silhouette_focus_sized.py`](visualizations_3d_silhouette_focus_sized.py).

Each script below is a checkpoint for creating better and better visualizations to document my work:

1. [`embeddings_of_data.py`](embeddings_of_data.py) - Generates embeddings for hackathon data and saves it to [`hackathon_data_with_embeddings.csv`](hackathon_data_with_embeddings.csv). This file contains the original data along with the generated embeddings for each entry.
2. [`visualize_embeddings.py`](visualize_embeddings.py) - Visualizes the embeddings using t-SNE and saves the output as HTML files in [`tsne_visualizations/`](tsne_visualizations/). This script provides a basic visualization of the embeddings without clustering.
3. [`visualize_embeddings_with_clusters.py`](visualize_embeddings_with_clusters.py) - Visualizes the embeddings with clustering and saves the output as HTML files in [`tsne_visualizations_clustered/`](tsne_visualizations_clustered/). This script enhances the visualization by grouping similar embeddings together, making it easier to identify clusters in the data.
4. [`visualize_embeddings_3d_clustered.py`](visualize_embeddings_3d_clustered.py) - Visualizes the embeddings in 3D with clustering and saves the output as HTML files in [`tsne_visualizations_3d_clustered/`](tsne_visualizations_3d_clustered/). This script provides a three-dimensional perspective of the clustered embeddings, allowing for a more comprehensive view of the data distribution.
5. [`visualizations_3d_silhouette_focus.py`](visualizations_3d_silhouette_focus.py) - Creates 3D visualizations focusing on silhouette scores and saves the output as HTML files in [`tsne_visualizations_3d_silhouette_focus/`](tsne_visualizations_3d_silhouette_focus/). This script emphasizes the silhouette scores of clusters, providing insights into the quality of clustering.
6. [`visualizations_3d_silhouette_focus_sized.py`](visualizations_3d_silhouette_focus_sized.py) - Similar to the previous script but with size variations, saving output in [`tsne_visualizations_3d_silhouette_focus_sized/`](tsne_visualizations_3d_silhouette_focus_sized/). This script enhances the 3D visualization by adjusting the size of points based on their text frequency.

### Data

-   [`Hackathon Round 2_Canada Survey Data.csv`](<Hackathon Round 2_Canada Survey Data.csv>) - Contains the original data from the hackathon.
-   [`characters_to_be_replaced.txt`](characters_to_be_replaced.txt) - A text file containing characters that had to be replaced in the data for better processing.
-   [`survey_questions.pdf`](survey_questions.pdf) - The original survey document used in the hackathon.
-   `hackathon_data_with_embeddings.csv` - The processed data file that includes the original survey data along with the generated embeddings for each entry.
    -   It serves as the input for the visualization scripts. It is not included in the repository because it is 600 MB in size, but it can be generated by running the [`embeddings_of_data.py`](embeddings_of_data.py) script.

### Other Files

-   [`README.md`](README.md) - This file, providing an overview of the project and instructions.
-   [`requirements.txt`](requirements.txt) - Lists the Python packages required to run the scripts.
    -   `pandas`: For loading, manipulating, and saving the survey data in DataFrames.
    -   `numpy`: For efficient numerical operations, especially handling the embedding vectors and size calculations.
    -   `scikit-learn`: Used for dimensionality reduction (`TSNE`), clustering (`KMeans`), and evaluating cluster quality (`silhouette_score`).
    -   `plotly`: The core library for generating the interactive 3D scatter plots.
    -   `openai`: Required for the initial step of generating text embeddings from the survey responses by calling the OpenAI API.
    -   `python-dotenv`: Used to manage your OpenAI API key securely by loading it from a `.env` file.
-   [`.gitignore`](.gitignore) - Specifies files and directories to be ignored by Git, ensuring that sensitive or unnecessary files are not included in the repository.
-   `.env` - Contains environment variables for the project, such as API keys or configuration settings.

## Getting Started

Follow these steps to set up the project and generate the visualizations on your own machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

This project requires an OpenAI API key to generate embeddings.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your OpenAI API key to this file in the following format:

    ```
    OPENAI_API_KEY="sk-YourSecretApiKeyHere"
    ```

> [!WARNING]
> The `.gitignore` file is already configured to ignore the `.env` file, so your key will not be committed to the repository.

### 5. Run the Scripts

The scripts must be run in the correct order.

1.  **Generate Embeddings**: Run `embeddings_of_data.py` first. This will process the raw survey data, call the OpenAI API, and create the `hackathon_data_with_embeddings.csv` file.
    _This step will incur costs on your OpenAI account and may take a significant amount of time._

    ```bash
    python embeddings_of_data.py
    ```

2.  **Generate Final Visualizations**: Once the embeddings file exists, run the final visualization script. This will perform the t-SNE, clustering, and aggregation to generate the interactive HTML files.

    ```bash
    python visualizations_3d_silhouette_focus_sized.py
    ```

### 6. View the Results

After the final script finishes running, the `tsne_visualizations_3d_silhouette_focus_sized/` directory now contains all of the regenerated HTML files. Open the `index.html` file in your web browser to see the master dashboard.
