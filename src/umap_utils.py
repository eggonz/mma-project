import argparse
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    """
    This method computes the umap from the model embeddings.
    """
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.float().cpu().numpy()
    elif isinstance(embeddings, pd.Series):
        embeddings = np.stack(embeddings.values)
    embeddings = embeddings.squeeze()

    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    t1 = time.time()
    umap_embeddings = fit.fit_transform(embeddings)
    t2 = time.time()
    print(f'UMAP elapsed time {datetime.timedelta(seconds=t2 - t1)}')

    return umap_embeddings


def draw_umap(umap_embeddings, title='UMap Visualization of Fromage dataspace'):
    """
    This method draws the umap on the plot and returns the plot
    """
    n_components = umap_embeddings.shape[1]
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(umap_embeddings[:, 0], range(len(umap_embeddings[:, 0])))
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], s=100)
    plt.title(title, fontsize=18)
    return fig


if __name__ == '__main__':
    """
    Can be used as script to add umap coords to existing embeddings dataframe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_pickle(args.embeddings_file)
    print(df.head())
    umap_embeddings = compute_umap(df['embeddings'], n_components=2)
    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]
    print(df.head())
    df.to_pickle(args.output_file)
