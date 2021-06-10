import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def scatterplot_3d(df, level=None, isReal2Syn=True):
    """
    3d scatter plot

    :param df: A pandas df of shape (n_images, 3 + 2) where the last two columns are the filename and data type.
    :param level: feature pyramid level for faster rcnn
    :param isReal2Syn: choosing mapping from real to syn or the other way around.
    :return: A 3d scatterplot of synthetic and real data.
    """
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    df_ = df.iloc[:,:3].astype('float32')
    x = df_.iloc[:,0]
    y = df_.iloc[:,1]
    z = df_.iloc[:,2]
    n = df.shape[0] // 2

    ax = plt.axes(projection='3d')
    ax.scatter(x[:n], y[:n], z[:n], color='yellow', linewidth=0.1, alpha=0.5, label='real data');
    ax.scatter(x[n:], y[n:], z[n:], color='purple', linewidth=0.1, alpha=0.5, label='syn data');
    ax.set_xlim(-3,3); ax.set_ylim(-2,5); ax.set_zlim(-3,3);
    ax.legend(); ax.set_xlabel('Latent Dim 1');ax.set_ylabel('Latent Dim 2');ax.set_zlabel('Latent Dim 3');
    ax.set_title('{1} UMAP 3d clustering visualization, p{0}, isReal2Syn={2}'.format(level, 'CycleGAN', isReal2Syn))
    plt.show()


def scatterplot_2d(df, real_index_lst, syn_index_lst, output_dir):
    """
    2d scatter plot

    :param df: A pandas df of shape (n_images, 2 + 2) where the last two columns are the filename and data type.
    :param real_index_lst: The outlier and centroid index of the image list.
    :param syn_index_lst: The outlier and centroid index of the image list.
    :param output_dir: The output directory for the 2d scatter plot.
    :return:
    """
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()
    df_ = df.iloc[:,:2].astype('float32')
    x = df_.iloc[:,0]
    y = df_.iloc[:,1]
    n = df.shape[0] // 2
    ax.scatter(x[:n], y[:n], color='yellow', linewidth=0.1, alpha=0.5, label='real data');
    ax.scatter(x[n:], y[n:], color='purple', linewidth=0.1, alpha=0.5, label='syn data');
    ax.legend(); ax.set_xlabel('Latent Dim 1');ax.set_ylabel('Latent Dim 2');
    ax.set_title('{1} UMAP 2d clustering visualization 2d, p{0}, isReal2Syn={2}'.format(LEVEL, 'CycleGAN', isReal2Syn))
    for k, i in enumerate(real_index_lst+syn_index_lst):
        plt.text(x[i], y[i], i)
        plt.scatter(x[i], y[i], marker='x', color='black')
    plt.show()
    plt.savefig(output_dir + "/p"+str(LEVEL)+"_scatter2d.pdf", dpi=150)


def getCentroidsAndOutliers(data):
    """
    Get centroids and outliers for the clusters.

    :param data: An df of shape (n_images, 2), where each column is a latent dim.
    :return: real_index_lst, syn_index_lst: A list of outlier or centroid image indexes
    """
    # learn the labels and the means
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)   #data is of shape [1000,]
    labels = kmeans.predict(data)  #labels of shape [1000,] with values 0<= i <= 9
    centroids  = kmeans.cluster_centers_  #means of shape [2,]

    # reformat centroids and predicted labels.
    center_df = pd.DataFrame(np.array([centroids[0].tolist()]*1000), columns=['center_x','center_y'])
    label_df = pd.DataFrame(labels, columns=['pred_labels'])
    out = pd.concat([data, label_df, center_df], axis = 1)

    # concat centroids
    for i in range(1000):
        if out.iloc[i,-3] == 1:
            out.iloc[i,-2] = centroids[1][0]
            out.iloc[i,-1] = centroids[1][1]
    # rename
    out = out.rename(columns={0:'x',1:'y'})
    out = out.astype(np.float32)

    # get distance
    out['distance'] = out['x']
    for i in range(1000):
        out['distance'][i] = abs(out['x'][i]-out['center_x'][i])**2+abs(out['y'][i]-out['center_y'][i])**2
    out = out.sort_values(by=['pred_labels','distance'])

    # real_index_lst    # [center1, center2, outlier1, outlier2]
    # syn_index_lst     # [center1, center2, outlier1, outlier2]
    real_index_lst = out[out['pred_labels']==1.0]['distance'][:2].index.tolist() + out[out['pred_labels']==1.0]['distance'][-2:].index.tolist()
    syn_index_lst = out[out['pred_labels']==0.0]['distance'][:2].index.tolist() + out[out['pred_labels']==0.0]['distance'][-2:].index.tolist()

    return real_index_lst, syn_index_lst


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None, output_dir=''):
    """
    Display the given set of images, optionally with titles.

    :param images:  List or array of image tensors in HWC format.
    :param titles: Optional. A list of titles to display with each image.
    :param cols:  The number of images per row
    :param cmap:  Optional. Color map to use. For example, "Blues".
    :param norm:  Optional. A Normalize instance to map values to colors.
    :param interpolation:  Optional. Image interpolation to use for display.
    :param output_dir: Optional, The output directory of image display.
    :return: A display of images.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
    plt.savefig(output_dir+"/p"+str(LEVEL)+"_imgs.pdf", dpi=150)

