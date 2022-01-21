# AutoEncoder

The goal of the assignment is to perform unsupervised learning on CIFAR-10
dataset. The initial task entails in creating a K-Means model from scratch
and the second part involves in creating an Auto-Encoder representation of the
dataset and on top of that perform K-Means clustering.

## Dataset
The dataset consists of 50,000 samples in the train set and 10,000 samples in
the test dataset. Each sample consist of 32x32 image which means the image
height is 32 pixels and image width is also 32 pixels. Each image also consist of
a label from the 10 classes described in the dataset.

## Python Implementation
I have used Jupyter Notebook IDE for implementation of the project and the
same has been shared.

### Data Preprocessing
The dataset is downloaded using Keras dataset api in python. All the images
in the dataset was of the dimension 32x32x3 meaning they had 3 channels
(RGB). In the first step of preprocessing I have converted the coloured images
into grayscale images making their dimension as 32x32x1. After this the images
very flattened to make their dimension 1x1024 for a single sample. Lastly all the
pixel values were divided by maximum pixel value which is 255 to standardize
the value across the dataset.

## K-Means
K-Means is a type of clustering technique. The goal of this technique is to
cluster ’n’ observation into ’k’ clusters by assigning each observation a cluster
value based on the nearest centroid point of that cluster. It was coded using
the below steps:

• Define the number of clusters you want your dataset to be divided into.
• After deciding on number of clusters, assign each cluster a centroid point
from the data.
• Once centroid point is decided for each cluster, measure the distance between
all the data points and every cluster’s centroid.
• Recalculate every cluster’s centroid by taking the mean of all the points
in that cluster.
• Keep repeating the above two steps until either the maximum number of
iterations are reached or the cluster’s centroid doesn’t change.

### Result
When I ran K-Means clustering on CIFAR-10 test data with 10 clusters, I
achieved a silhouette score of 0.057 and dunn index value of 0.089.

## Auto-Encoder
Auto-Encoder is a type of neural network which helps in extracting important
feature from a dataset which helps in encoding the dataset into lower dimension.
Once it has lowered the dimensionality of the dataset we reconstruct the dataset
back to its original form using the encoded features. It comprises of two element
which is the Encoder layer and Decoder layer respectively. Auto-Encoders has
multiple applications in today’s world like fraud detection, image reduction, etc.

![image](https://user-images.githubusercontent.com/47882482/150612511-4ba8dae4-600a-4f79-a910-daa34937aba2.png)

Figure 1: Auto-Encoder structure

The above image is taken from google.
I have used sparse representation of CIFAR-10 dataset to get the data from the
encoder layer which is the middle layer in the above image. Below is an image
of Auto-Encoder structure which I have used to implement it.

### Architecture
Initially I have taken a convolutional layer with kernel size of 5, num of filters
as 3, activation function as Relu and l1 regularizer for sparse representation.
After this layer I have taken a Max pooling on top of that layer with pool
size of 2 which will select the maximum value in that filter space. Again after
pooling I have used a convolutional layer with 1 num of filters and rest of the
feature keeping same as the previous convolutional layer. On top of that we
have another max pooling layer with the same hyperparameters as the previous
max pooling layer. This layer is also the encoded layer from which we will
extract the image data in lower dimensions. Now I have added a convolutional
layer with filter 1 and rest of the hyperparameters same as the previous layers.
I have added up sampling layer with 2 as the size. The same convolutional layer
and up sampling layer is repeated with just number of filters changing to 3. In
the end I have used another convolutional layer with 1 as the number of filters
which will put the image back to its original size.

![image](https://user-images.githubusercontent.com/47882482/150612542-d5d35eda-6c98-4ce0-baf9-ed28ce3b0452.png)

Figure 2: Auto-Encoder architecture

### Results

![image](https://user-images.githubusercontent.com/47882482/150612563-a21fb19a-cd54-4d80-9179-7db48174e728.png)

Figure 3: Comparison between images. The images in the top row are before
Auto-Encoder and images in the last row are there representation using Auto-
Encoder’s encoding layer
The Auto-Encoder model was trained for 5 epochs with batch size as 32. The
model was compiled with optimizer as ’Adam’ and loss as ’mse’ which is mean
squared error. The loss after 5 epoch for Auto-Encoder was 0.0114. Below is a
comparison of before and after Auto-Encoder encoding.
The embedding from encoder layer were provided to kmeans with 10 cluster
which was giving a silhouette score of 0.074 and dunn index value of 0.041.
