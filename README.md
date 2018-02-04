# Face_recognition

### How to run the program

    python eigenfaces.py

### Dataset

1. The dataset used here is the AT&T dataset of 400 images featuring 10 people. Each image is of size 92 * 112 pixels. 
2. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the 
subject number (between 1 and 40). In each of these directories, there are ten different images of that subject,which have names 
of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).

### Implementation

Each image is converted to a feature vector i.e, flattened to size 1*10304. But using Neural networks or SVM on a data with
a feature vector of that size will increase the computational a lot. So, dimension reduction techniques like PCA were used to reduce
the dimensions or bring latent factors from large data. 

We can also call them Eigen faces as a mean profile for all the images is constructed first and then we take the top k faces that 
can identify the uniqueness of all images.

Each image can be represented as a combination of these eigen faces with some error, but that is very minimal that we cannot observe
much differene between the two.

### Classifiers used

After applying PCA, Neural networks classifier is used to classify the images.
