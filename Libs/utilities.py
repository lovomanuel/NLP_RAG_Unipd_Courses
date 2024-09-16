import matplotlib.pyplot as plt
import math

# Dictionary of all LLM model_names:full_model_name
llm_models = {
    'llama_13': 'meta-llama/Llama-2-13b-chat-hf',
    'llama_7': 'meta-llama/Llama-2-7b-chat-hf',
    'mistral_7':'mistralai/Mistral-7B-Instruct-v0.3'
    }

# Dictionary of all embedding model_names:full_model_name
embedding_models = {
                'MiniLM': 'sentence-transformers/all-MiniLM-L6-v2',
                'paraphrase': 'sentence-transformers/paraphrase-mpnet-base-v2',
                'distilbert': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
                'allmpnet': 'sentence-transformers/all-mpnet-base-v2',
                'distilbertmean': 'sentence-transformers/distilbert-base-nli-mean-tokens',
                'multiMiniLM': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                'bert': 'sentence-transformers/bert-base-nli-mean-tokens',
                'bge': 'BAAI/bge-small-en-v1.5'
            }


def plot_data_hist(data, title = "data"):
    num_plots = len(data)
    fig, axes = plt.subplots(1, num_plots, figsize=(min(14, 4 * num_plots), 6), sharey=True)

    # In case there's only one plot, axes will not be a list
    if num_plots == 1:
        axes = [axes]

    for ax, (key, documents) in zip(axes, data.items()):
        lengths = [len(doc.page_content) for doc in documents]

        # Determine the number of bins, ensuring it's not excessively large
        num_bins = min(math.ceil(max(lengths) / 250), 50)

        # Define bin edges
        bin_edges = [i * 250 for i in range(num_bins + 1)]

        # Plot the histogram
        ax.hist(lengths, bins=bin_edges, edgecolor='black', color='skyblue')
        ax.set_xlabel('Length of Strings', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{key}', fontsize=12)
        ax.grid(True)

        # Adjust x-ticks to show one over two
        xticks = bin_edges[::2]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45)

    fig.suptitle(f'Histograms of {title} with Sensitivity 250', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the main title
    plt.show()