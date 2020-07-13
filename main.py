import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing


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
    """
    ORB detector
    
    Attributes:
        param: Parameters of the ORB detector.
        detector: Detector object.
    """

    def __init__(self):
        self.param = dict(nfeatures=600, scaleFactor=1.2, nlevels=4,
                          edgeThreshold=31, firstLevel=0, WTA_K=2,
                          scoreType=cv.ORB_HARRIS_SCORE,
                          patchSize=31, fastThreshold=20)
        self.detector = cv.ORB_create(**self.param)

    def detectFromDict(self, images_dict: dict) -> [list, dict]:
        """
        Get ORB features from a dictionary of images and save
        them in a new dictionary.
        """
        orb = self.detector
        descriptor_list = []  # a list of all features
        descriptor_bycat = {}  # a dictionary of features by categories

        for labels, img in images_dict.items():
            if isinstance(img, list):
                # if is a list then it is the dictionary with the classified images
                for n in range(len(img)):
                    kp, des = orb.detectAndCompute(img, None)
                    descriptor_list.extend(des)
                    descriptor_bycat[labels] = des
            else: 
                kp, des = orb.detectAndCompute(img, None)
                descriptor_list.extend(des)
                descriptor_bycat[labels] = des

        return descriptor_list, descriptor_bycat

    def normalizeAllFeatures(self, trainbycat: dict, testbycat: dict): 
        allfeatureslist = []
        traininfo = {}
        testinfo = {}
        trainlist = []
        testlist = []
        trainbycat_n = {}
        testbycat_n = {}

        # Get a list of all features
        for cat, features in trainbycat.items():
            traininfo[cat] = features.shape[0]
            allfeatureslist.extend(features)
        for cat, features in testbycat.items():
            testinfo[cat] = features.shape[0]
            allfeatureslist.extend(features)

        # Normalize it
        allfeatureslist = preprocessing.normalize(allfeatureslist, norm='l2')

        # Get the lists of normalized values for train and test
        ntrain = sum(traininfo.values()) 
        ntest = sum(testinfo.values())
        trainlist = allfeatureslist[0:ntrain-1]
        testlist = allfeatureslist[ntrain: ntest-1]

        # Organize normalized features by class for train and test
        for cat, nfeatures in traininfo.items():
            trainbycat_n[cat] = allfeatureslist[0:nfeatures]
            allfeatureslist = allfeatureslist[nfeatures+1:-1]
        for cat, nfeatures in testinfo.items():
            testbycat_n[cat] = allfeatureslist[0:nfeatures]
            allfeatureslist = allfeatureslist[nfeatures+1:-1]

        return trainlist, trainbycat_n, testlist, testbycat_n


class bow():
    def __init__(self):
        """ 
        Attributes:
            t: numerical threshold for classifying.
            vw_centers: list of the cluster centers for the visual words.
        """
        self.t = 1 
        self.vw_centers = []
        self.nclusters = 150
    
    def setThreshold(self, threshold):
        """
        Sets a new classifying threshold.
        """
        self.t = threshold

    def normalizeData(self, data):
        """ 
        Preprocesses the data by centering to the mean 
        and component wise scaling to unit variance.
        """ 
        if isinstance(data, dict):
            v = list(data.values())
            for l in range(len(v)):
                lists = v[l]
                lists = preprocessing.normalize(lists, norm='l2')
                # for f in range(lists.shape[0]):
                #     singlefeatures = lists[f]
                    # featuresfloat = [float(i) for i in singlefeatures]
                    # factorl2 = 1/np.linalg.norm(featuresfloat)
                    # lists[f] = [i * factorl2 for i in featuresfloat]
            count = 0
            for keys in data.keys():
                data[keys] = v[count]
        else: 
            data = preprocessing.normalize(data, norm='l2')

        return data
        
    def kmeans(self, descriptor_list: list) -> list:
        """
        A k-means clustering algorithm.
        
        Inputs:
            descriptor_list: Descriptors list
                (unordered nfeatures x 32 matrix).

        Returns: A matrix [nclusters, 32] that holds 
            central points of the clusters.
        """
        cluster = KMeans(n_clusters=self.nclusters, n_init=10)
        cluster.fit(descriptor_list)
        self.vw_centers = cluster.cluster_centers_ 

        return cluster.cluster_centers_

    def compute_histogram(self, featuresbycat: dict) -> dict:
        """
        Computes the histogram for every category.

        Inputs:
            featuresbycat: Dictionary that contains features 
                of each category.
            centers: A matrix [nclusters, 32] that holds 
                central points of the clusters.

        Returns: A dictionary with the histograms of
            each category.
        """
        histogrambycat = {}  # dictionary of histograms
        centers = self.vw_centers
        histogram = np.zeros([1, self.nclusters])
        for cat, features in featuresbycat.items():
            for n in range(features.shape[0]):
                word = self.matchWord(features[n, :])
                histogram[0, word] += 1
            histogrambycat[cat] = histogram
            histogram = np.zeros([1, self.nclusters])

        return histogrambycat

    def matchWord(self, feature) -> int:
        """
        Finds the matching word searching for the minimum 
        euclidean distance.

        Inputs:
            feature: A single feature (descriptor).
            centers: A matrix [nclusters, 32] that holds central 
                points of the clusters.
        Returns: 
            Index of the matched word.
        """
        centers = self.vw_centers
        for n in range(centers.shape[0]):
            dist = np.linalg.norm(feature - centers[n, :])  # L2-norm
            if n == 0:
                mindist = dist
                minind = n
            elif dist < mindist:
                mindist = dist
                minind = n

        return minind

    def matchCategory(self, trainh: dict, testh: dict) -> dict:
        """
        Finds the matching category for every image in test dataset using 1NN.

        Inputs:
            trainh: Dictionary of histograms by class, from training.
            testh: Dictionary of histograms by image, from test.
        Returns: 
            Dictionary of the test images classified.
        """
        img_classified = {}
        classification = np.nan
        for imgname, histtest in testh.items():
            for cat, hist in trainh.items():
                dist = np.linalg.norm(histtest - hist)  # L2-norm
                if dist < self.t:
                    classification = cat
            img_classified[imgname] = classification
            classification = np.nan

        return img_classified

    def findThreshold(self, trainh: dict, testh: dict):
        """
        Finds ans sets a new threshold based on the distances 
        of the test and training histogram sets.
        """
        dist_list = np.array([])
        for imgname, histtest in testh.items():
            for cat, hist in trainh.items():
                dist = np.linalg.norm(histtest - hist)  # L2-norm
                dist_list = np.append(dist_list, dist)
        distmin = np.min(dist_list)
        distmean = np.mean(dist_list)
        t = distmean - (distmean-distmin)/2
        self.setThreshold(t)



def main():
    # Obtain images by name and by class from dataset
    train_dset_bycat = get_images("ddataset/train")
    train_dset_byname = get_images("ddataset/train")
    test_dset_byname = get_images("ddataset/test")  # take test images

    # Gets all the features from the training images
    orb = orb_features()
    train_features, train_featuresbycat = orb.detectFromDict(train_dset_bycat)
    # Get all the features from the test images
    test_features, test_featuresbycat = orb.detectFromDict(test_dset_byname)
    # Normalize them
    train_features, train_featuresbycat, test_features, test_featuresbycat = \
    orb.normalizeAllFeatures(train_featuresbycat, test_featuresbycat)

    bow_obj = bow()
    # Clusters the features using kmeans
    bow_obj.kmeans(train_features)
    # Compute histograms for every category in train dataset
    train_histogrambycat = bow_obj.compute_histogram(train_featuresbycat)

    # Compute histrograms for every image in test dataset
    test_histogrambycat = bow_obj.compute_histogram(test_featuresbycat)

    # Find new threshold
    bow_obj.findThreshold(train_histogrambycat, test_histogrambycat)
    # Classify the test images into the trainned categories
    classification_results = bow_obj.matchCategory(
        train_histogrambycat, test_histogrambycat)
    print("Classification results: ")
    print(classification_results)


if __name__ == "__main__":
    main()
