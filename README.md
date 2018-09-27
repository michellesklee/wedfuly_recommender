![wedfuly](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/app_index.png)

___

## Table of Contents
**[Background](#background)**<br>
**[Data Collection](#data-collection)**<br>
**[Florist Image Processing](#image-processing)**<br>
**[Convolutional Autoencoder](#convolutional-autoencoder)**<br>
**[k-Means Clustering](#k-means-clustering)**<br>
**[Evaluating the Model](#evaluating-the-model)**<br>
**[Building a Recommender](#building-a-recommender)**<br>
**[Recommender Application](#recommender-application)**<br>
**[Conclusion](#conclusion)**<br>
**[Next Steps](#next-steps)**<br>
**[Tech Stack](#tech-stack)**<br>



## Background
On [Wedfuly](https://wedfuly.com/), clients work online with wedding planners who help with the planning process, including choosing wedding vendors such as florists, photographers, and bakeries.

The goal of this project was to build a recommender that facilitates the process of wedding planners suggesting vendors to their clients. For the first part of this project, I focused on florists with the goal of training a model to meaningfully cluster floral arrangements (bridal bouquets specifically). As a second step, I built a similar recommender for photographers using pre-labeled images of Wedfuly photographers.

## Data Collection
Images of bridal bouquets collected online (train set: 2897, test set: 637)
Images from Wedfuly wedding photographers (~1200)

## Florist Image Processing 
1. Cropped bouquets from images
2. Centered bouquets and cropped images to square
3. Resized images to 100x100
4. Converted to RGB
5. Augmentation: LR, UD

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

## k-Means Clustering
Clustering with k-Means - 7 was optimal k
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/elbow_plot.png)


### Images from each cluster:


#### Cluster 0: "Moody and Dark" (38 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster1.png)

#### Cluster 1: "Earthy Modern" (55 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster6.png)

#### Cluster 2: "Simply Minimal" (59 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster3.png)

#### Cluster 3: "Moody and Wild" (63 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster5.png)

#### Cluster 4: "Traditional" (75 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster0.png)

#### Cluster 5: "Light and Modern" (44 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster2.png)


#### Cluster 6: "Colorful and Bold" (52 images)

![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/cluster4.png)


## Evaluating the Model - Florists
To evaluate the model, I googled "bridal bouquet" and chose 40 of the top images to mimic the process of clients adding images to their Pinterest board.

#### Cluster 0: "Moody and Dark" (38 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster1.png)

#### You might like these vendors...
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

#### Cluster 4: "Traditional" (75 images)

#### If you like this...
![](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/valid_cluster0.png)

#### You might like these vendors...
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

## Building a Recommender
k-Means cluster labels became a feature of an overall model that also included the following features:

#### Continuous Variables:
1. Total price 
2. Location of wedding - Boulder, Denver, Foothills, Summit County, Mountain Town Not Listed, Other

#### Categorical Variables (dummy coded):
3. Method of delivery - Delivery (full service), Delivery (no service)/Drop-off, or Pick-up
4. Services provided - Ceremony Decor, Reception Decor, and/or Handhelds
5. Size of wedding

#### Distance Metric
As clients enter specifics of their wedding as well as select images that match their style, *cosine similarity* will be calculated to find the vendor that has provided wedding services that are most similar to the clients needs.

## Recommender Application 
A [recommender application](http://54.242.37.247:8080/) built on Flask and Docker and hosted on Amazon Web Services allows clients and wedding planners to enter specifics of their wedding and select images to recommend vendors. 

![app](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/app_recommender1.png)
![app](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/app_recommender2.png)

## Conclusion
Even with limited training images, a k-Means algorithm on top of a CNN autoencoder has potential to  cluster images for recommendation purposes. Using k-Means cluster labels in addition to features related to the wedding provides a more robust recommender system that takes into account the specifics of the wedding as well as clients' style.

## Next Steps
1. As Wedfuly grows and more data is collected, the recommender will be further fine tuned and validated
2. While manually cropping bouquets was sufficient for this initial model, in the future, training a neural network to detect bouquets may be worth exploring
3. Using transfer learning such as XCeption may also be considered in the future to detect features in images

## Tech Stack
![tech_stack](https://github.com/michellesklee/wedfuly_recommender/blob/master/figures/tech_stack.png)


