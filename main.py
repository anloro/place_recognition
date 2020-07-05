import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans


def get_images(folder) -> dict:
    """
    Get images from a folder and put them into dictionaries.
    """
    images = {}
    for filename in os.listdir(folder):
        category = filename  # Dictionary name is each image name
        path = folder + "/" + filename
        img = cv.imread(path, 0)  # Read in grayscale
        images[category] = img

    return images


class orb_features:
    """ORB detector
    
    Attributes:
        param: Parameters of the ORB detector
        detector: Detector object
    """

    def __init__(self):
        self.param = dict(nfeatures=600, scaleFactor=1.2, nlevels=4,
                          edgeThreshold=31, firstLevel=0, WTA_K=2,
                          scoreType=cv.ORB_HARRIS_SCORE,
                          patchSize=31, fastThreshold=20)
        self.detector = cv.ORB_create(**self.param)

    def detectFromDict(self, images_dict) -> [list, dict]:
        """
        Get ORB features from a dictionary of images and save
        them in a new dictionary.
        """
        orb = self.detector
        descriptor_list = []  # a list of all feaatures
        descriptor_bycat = {}  # a dictionary of features by categories

        for imgname, img in images_dict.items():
            cat = imgname
            kp, des = orb.detectAndCompute(img, None)
            descriptor_list.extend(des)
            descriptor_bycat[cat] = des

        return descriptor_list, descriptor_bycat


def kmeans(descriptor_list) -> list:
    """
    A k-means clustering algorithm.
    
    Inputs:
        descriptor_list: Descriptors list(unordered nfeatures x 32 matrix).

    Returns: A matrix [nclusters, 32] that holds central points of the clusters.
    """
    k = 150
    cluster = KMeans(n_clusters=k, n_init=10)
    cluster.fit(descriptor_list)

    return cluster.cluster_centers_


def compute_histogram(featuresbycat, centers) -> dict:
    """
    Computes the histogram for every category

    Inputs:
        featuresbycat: Dictionary that contains features of each category
        centers: A matrix [nclusters, 32] that holds central points of the clusters.

    Returns: 
    """
    histogrambycat = {}  # dictionary of histograms
    nclusters = centers.shape[0]
    histogram = np.zeros([1, nclusters])
    for cat, features in featuresbycat.items():
        for n in range(features.shape[0]):
            word = matchWord(features[n, :], centers)
            histogram[0, word] += 1
        histogrambycat[cat] = histogram
        histogram = np.zeros([1, nclusters])

    return histogrambycat


def matchWord(feature, centers) -> int:
    """
    Finds the matching word searching for the minimum euclidean distance

    Inputs:
        feature: A single feature (descriptor)
        centers: A matrix [nclusters, 32] that holds central points of the clusters.
    Returns: 
        Index of the matched word
    """
    for n in range(centers.shape[0]):
        dist = np.linalg.norm(feature - centers[n, :])  # L2-norm
        if n == 0:
            mindist = dist
            minind = n
        elif dist < mindist:
            mindist = dist
            minind = n

    return minind


def matchCategory(trainh, testh) -> dict:
    """
    Finds the matching category for every image in test dataset using 1NN.

    Inputs:
        trainh: Dictionary of histograms by class, from training
        testh: Dictionary of histograms by image, from test
    Returns: 
        Dictionary of the test images classified
    """
    img_classified = {}
    n = 0
    for imgname, histtest in testh.items():
        for cat, hist in trainh.items():
            dist = np.linalg.norm(histtest - hist)  # L2-norm
            if n == 0:
                mindist = dist
                classification = cat
            elif dist < mindist:
                mindist = dist
                classification = cat
            n += 1
        img_classified[imgname] = classification
        n = 0

    return img_classified


def main():
    train_dset = get_images("ddataset/train")
    test_dset = get_images("ddataset/test")  # take test images

    # Gets all the features from the training images
    orb = orb_features()
    train_features, train_featuresbycat = orb.detectFromDict(train_dset)
    # Clusters the features using kmeans
    vw_centers = kmeans(train_features)
    # Compute histograms for every category in train dataset
    train_histogrambycat = compute_histogram(train_featuresbycat, vw_centers)

    # Get features and compute histrograms for every image in test dataset
    test_features, test_featuresbycat = orb.detectFromDict(test_dset)
    test_histogrambycat = compute_histogram(test_featuresbycat, vw_centers)

    # Classify the test images into the trainned categories
    classification_results = matchCategory(
        train_histogrambycat, test_histogrambycat)
    print("Classification results: ")
    print(classification_results)


if __name__ == "__main__":
    main()
