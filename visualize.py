'''

'''

import numpy as np
import matplotlib.pyplot as plt


def scatterplot_3d(df):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    df_ = df.iloc[:,:3].astype('float32')
    x = df_.iloc[:,0]
    y = df_.iloc[:,1]
    z = df_.iloc[:,2]
    n = 500

    ax = plt.axes(projection='3d')
    ax.scatter(x[:n], y[:n], z[:n], color='yellow', linewidth=0.1, alpha=0.5, label='real data');
    ax.scatter(x[n:], y[n:], z[n:], color='purple', linewidth=0.1, alpha=0.5, label='syn data');
    ax.set_xlim(-3,3); ax.set_ylim(-2,5); ax.set_zlim(-3,3);
    ax.legend(); ax.set_xlabel('Latent Dim 1');ax.set_ylabel('Latent Dim 2');ax.set_zlabel('Latent Dim 3');
    ax.set_title('{1} UMAP 3d clustering visualization, p{0}, isReal2Syn={2}'.format(LEVEL, 'CycleGAN', isReal2Syn))
    plt.show()


def scatterplot_2d(df2, real_index_lst=real_index_lst, syn_index_lst=syn_index_lst):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()
    df_ = df2.iloc[:,:2].astype('float32')
    x = df_.iloc[:,0]
    y = df_.iloc[:,1]
    n = 500
    ax.scatter(x[:n], y[:n], color='yellow', linewidth=0.1, alpha=0.5, label='real data');
    ax.scatter(x[n:], y[n:], color='purple', linewidth=0.1, alpha=0.5, label='syn data');
    ax.legend(); ax.set_xlabel('Latent Dim 1');ax.set_ylabel('Latent Dim 2');
    ax.set_title('{1} UMAP 2d clustering visualization 2d, p{0}, isReal2Syn={2}'.format(LEVEL, 'CycleGAN', isReal2Syn))
    for k, i in enumerate(real_index_lst+syn_index_lst):
        plt.text(x[i], y[i], i)
        plt.scatter(x[i], y[i], marker='x', color='black')
    # plt.show()
    plt.savefig("/content/sample_data/p"+str(LEVEL)+"_scatter2d.pdf", dpi=150)

scatterplot_3d(df)
scatterplot_2d(df2)


def getCentroidsAndOutliers():
    # get labels and centroids
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
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
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
    # plt.show()
    plt.savefig("/content/sample_data/p"+str(LEVEL)+"_imgs.pdf", dpi=150)


display_images(images, real_index_lst+syn_index_lst, cols=4)