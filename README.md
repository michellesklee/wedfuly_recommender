## Data Collection:
1. Images of bridal bouquets collected online from florists in Colorado (n = ) split into train and test (80/20)
2. Images from Wedfuly florists (n = ) - holdout data

## Image Processing
1. Cropped bouquets from images
2. Cropped images to square
3. Resized images to 100x100
4. Converted to RGB


## Convolutional Autoencoder

![cnn](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/model1.png)
loss = 0.0258 val_loss = 0.0247

#### Reconstructed Images
![reconstructed](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cnn_first_pass.png)

#### Encoded Layers
Isolate encoding layers:

```python
get_encoded = K.function([cnn_model.layers[0].input], [cnn_model.layers[5].output])
```
Pool filters from the encoded layers to see where model is paying most attention. Taking max() of filters is standard method.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded2.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded3.png)

Next pool up to the last convolutional later in the model. The image on the right is the max of the filters of the decoded image. Bright spots are where the model is activated the most.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/plot_with_attention.png)

## KMeans
Clustering with encoded images - 7 was optimal 7
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/elbow_plot.png)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster_hist.png)

#### Images from each cluster:

Cluster 0 (n = 47)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster0.png)

Cluster 1 (n = 44)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster1.png)

Cluster 2 (n = 49)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster2.png)

Cluster 3 (n = 31)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster3.png)

Cluster 4 (n = 68)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster4.png)

Cluster 5 (n = 34)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster5.png)

Cluster 6 (n = 44)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster6.png)

## Next Steps:
1. Train a neural network to detect bouquets in images
