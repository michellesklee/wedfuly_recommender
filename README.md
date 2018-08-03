[Add Wedfuly image]

<h1 align="center"> Vendor Recommender for Wedfuly </h1>

[Project Description]

## Data Collection:
1. Images of bridal bouquets collected online from florists in Colorado (n = 387) - test set
2. Images from Wedfuly florists (n = 115) - train set
3. Unseen images collected online (n = 40) - validation set

## Feature Engineering
### Image Processing Steps:
1. Cropped bouquets from images
2. Cropped images to square
3. Resized images to 100x100
4. Converted to RGB

## Convolutional Autoencoder

![cnn](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/model1.png)

![loss_plot](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/loss_plot.png)
loss: 0.0140 - val_loss: 0.0142

#### Reconstructed Images
![reconstructed](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cnn_first_pass.png)

#### Building Attention Model
##### Determining where the model is paying the most attention

Isolate encoding layers:

```python
get_encoded = K.function([cnn_model.layers[0].input], [cnn_model.layers[5].output])
```

Pool filters from the encoded layers to see where model is paying most attention. Taking max() of filters is standard method.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded2.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded3.png)

Next pool up to the last convolutional layer in the model. The images on the right is the max of the filters of the decoded image. Bright spots are where the model is activated the most.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/plot_with_attention.png)

## KMeans
Clustering with KMeans - 7 was optimal k
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/elbow_plot.png)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster_hist.png)

#### Images from each cluster:

Cluster 0 (n = 75)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster0.png)

Cluster 1 (n = 38)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster1.png)

Cluster 2 (n = 44)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster2.png)

Cluster 3 (n = 59)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster3.png)

Cluster 4 (n = 52)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster4.png)

Cluster 5 (n = 63)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster5.png)

Cluster 6 (n = 55)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster6.png)

#### Evaluating the model
Since there is no way to validate the model, I randomly chose 40 images googling "bridal bouquet" and tested those clusters against the vendor clusters (and training clusters) - so if you were to add these in your pinterest board, the recommender would recommend the following 3 vendor images

##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster0.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster0.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster1.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster1.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster2.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster2.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster3.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster3.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster4.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster4.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster5.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster5.png)
___
##### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster6.png)

##### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster6.png)
___
## Conclusion
Even with limited training images, a kMeans algorithm on top of a CNN autoencoder has potential to  cluster images for recommendation purposes.

## Next Steps:
1. Include survey data in algorithm (e.g., budget)
2. Extend to other vendors(e.g., photographers)
3. Train a neural network to detect bouquets in images
