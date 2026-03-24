"""
HDMC-Net Visualization Utilities

Visualization functions for attention maps, embeddings, and results.
"""

import wandb
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

matplotlib.use('Agg')
flatui = ["#9b59b6", "#3498db", "orange"]


def plot_attention_weights(attentions):
    """Plot multi-head attention weights as heatmaps."""
    num_layer = len(attentions)
    num_attn = attentions[0].shape[0]
    fig = plt.figure(figsize=(num_attn, num_layer))

    for layer in range(num_layer):
        for head in range(num_attn):
            ax = fig.add_subplot(num_layer, num_attn, (layer * num_attn) + head + 1)
            ax.matshow(attentions[layer][head], cmap='viridis')
            fontdict = {'fontsize': 7}
            ax.set_xlabel('Head {}'.format(head + 1), fontdict={'fontsize': 5})
            ax.axis('off')

    plt.tight_layout()


def tsne(x):
    """Apply t-SNE dimensionality reduction."""
    test_features = x.cpu().numpy()
    tsne_model = TSNE(n_components=2, perplexity=10, n_iter=250)
    tsne_ref = tsne_model.fit_transform(test_features)
    return tsne_ref


def pca(x):
    """Apply PCA dimensionality reduction."""
    test_features = x.cpu().numpy()
    pca_model = PCA(n_components=2)
    pca_model.fit(test_features)
    pcs = pca_model.fit_transform(test_features)
    return pcs


def df_tsne(tsne_ref, label):
    """Create DataFrame from t-SNE results."""
    df = pd.DataFrame(tsne_ref, index=tsne_ref[0:, 1])
    df['x1'] = tsne_ref[:, 0]
    df['x2'] = tsne_ref[:, 1]
    df['label'] = label
    return df


def df_pca(pcs, label):
    """Create DataFrame from PCA results."""
    df = pd.DataFrame(data=pcs, columns=['x1', 'x2'])
    label = pd.DataFrame(label)
    df = pd.concat([df, label], axis=1, join='inner', ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = ['x1', 'x2', 'label']
    return df


def plot_sns_scatter(df, label):
    """Plot scatter plot using seaborn."""
    sns.scatterplot(x="x1", y="x2", hue='label', data=df, legend=True,
                    palette=sns.color_palette("hls", 10), 
                    scatter_kws={"s": 50, "alpha": 0.5})


def plot_sns_lm(df, label):
    """Plot scatter plot using seaborn lmplot."""
    sns.set_palette(flatui)
    sns.lmplot(x="x1", y="x2", data=df, fit_reg=False, legend=True, 
               hue='label', scatter_kws={"s": 50, "alpha": 0.5})


def plot_plt_scatter(tsne_ref, label):
    """Plot scatter plot using matplotlib."""
    f, ax = plt.subplots()
    cmap = sns.color_palette("light:#9b59b6", as_cmap=True)
    points = ax.scatter(tsne_ref[:, 0], tsne_ref[:, 1], c=label, s=50, cmap=cmap)
    f.colorbar(points)


def plot_dr(x_tsne, x_pca, label, i_plot, name):
    """Plot dimensionality reduction results (t-SNE and PCA)."""
    label = label.cpu().numpy()

    def plot_subdr(xdr, drf, plotf, ptitle, wtitle, axis_label=True):
        result = drf(xdr, label)
        plotf(result, label)
        plt.title('{}'.format(ptitle, name), weight='bold').set_fontsize('14')
        if axis_label:
            plt.xlabel('u1', weight='bold').set_fontsize('14')
            plt.ylabel('u2', weight='bold').set_fontsize('14')

    need_df = i_plot != 2
    tsnef = df_tsne if need_df else lambda x, y: x
    pcaf = df_pca if need_df else lambda x, y: x
    plotf = [plot_sns_scatter, plot_sns_lm, plot_plt_scatter][i_plot]
    plot_subdr(x_tsne, tsnef, plotf, 't-SNE: ', 'tsne_', False)
    plot_subdr(x_pca, pcaf, plotf, 'PCA: ', 'pca_')
