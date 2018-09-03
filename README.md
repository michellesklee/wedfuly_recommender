![wedfuly](https://wedfuly.com/wp-content/uploads/2018/04/wedfuly.jpeg)

<h1 align="center"> Vendor Recommender for Wedfuly </h1>
<h4 align="center"> Part I: Cluster Florists with CNN Autoencoder + KMeans </h4>

___


On Wedfuly, clients work online with wedding planners who help with the planning process, including choosing wedding vendors such as florists, photographers, and bakeries.

The ultimate goal of this project is to build a recommender that facilitates the process of wedding planners suggesting vendors to their clients.

For the first part of this project, I focused on florists with the goal of training a model to meaningfully cluster floral arrangements (bridal bouquets specifically).

## Data Collection:
1. Images of bridal bouquets collected online from florists in Colorado (n = 387) - train set
2. Images from Wedfuly florists (n = 115) - test set
3. Unseen images collected online (n = 40) - validation set

## Image Processing Steps:
1. Cropped bouquets from images
2. Centered bouquets and cropped images to square
3. Resized images to 100x100
4. Converted to RGB

## Convolutional Autoencoder

![cnn](https://cdn-images-1.medium.com/max/1818/1*8ixTe1VHLsmKB3AquWdxpQ.png)

![cnn_arch](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/model1.png)

![loss_plot](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/loss_plot.png)

loss: 0.0140 - val_loss: 0.0142

___

### Reconstructed Images
![reconstructed](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cnn_first_pass.png)

### Building Attention Model
#### Determining where the model is paying the most attention

Isolate encoding layers:

```python
get_encoded = K.function([cnn_model.layers[0].input], [cnn_model.layers[5].output])
```

Pool filters from the encoded layers to see where model is paying most attention. Taking max() of filters is standard method.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded2.png)
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/encoded3.png)

Next pool up to the last convolutional layer in the model. The images on the right show the max of the filters. Bright spots are where the model is activated the most -- seemingly bright things and contrasts.

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/plot_with_attention.png)

## Clustering - KMeans
Clustering with KMeans - 7 was optimal k
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/elbow_plot.png)


### Images from each cluster:


# Cluster 0: "Moody and Dark" (38 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster1.png)

# Cluster 1: "Earthy Modern" (55 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster6.png)

# Cluster 2: "Simply Minimal" (59 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster3.png)

# Cluster 3: "Moody and Wild" (63 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster5.png)

# Cluster 4: "Traditional" (75 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster0.png)

# Cluster 5: "Light and Modern" (44 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster2.png)


# Cluster 6: "Colorful and Bold" (52 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster4.png)





## Evaluating the model
To evaluate the model, I googled "bridal bouquet" and chose 40 of the top images to mimic the process of clients adding images to their Pinterest board.

# Cluster 0: "Moody and Dark" (38 images)

# If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster1.png)

# You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster1.png)
___

#### Cluster 1: "Earthy Modern" (55 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster6.png)

#### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster6.png)
___


#### Cluster 2: "Simply Minimal" (59 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster3.png)

#### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster3.png)

___
#### Cluster 3: "Moody and Wild" (63 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster5.png)

#### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster5.png)

___

# Cluster 4: "Traditional" (75 images)

# If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster0.png)

# You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster0.png)
___


#### Cluster 5: "Light and Modern" (44 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster2.png)

#### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster2.png)

___

#### Cluster 6: "Colorful and Bold" (52 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster4.png)

#### You might like these vendors...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/test_cluster4.png)
___



## Conclusion
Even with limited training images, a kMeans algorithm on top of a CNN autoencoder has potential to  cluster images for recommendation purposes.

## Next Steps:
1. Include survey data in algorithm (e.g., budget)
2. Extend to other vendors(e.g., photographers)
3. Train a neural network to detect bouquets in images
4. Started working with a pre-trained model (VGG-19) for feature extraction - may be worth exploring


## References
Wedfuly: https://wedfuly.com/

Convolutional autoencoder image: https://cdn-images-1.medium.com/max/1818/1*8ixTe1VHLsmKB3AquWdxpQ.png

Clustering with autoencoders and attention maps: http://maxcalabro.com/clustering-images-with-autoencoders-and-attention-maps/
