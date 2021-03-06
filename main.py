import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
import h5py
from scipy.spatial.distance import squareform
import time
import re
import pickle
from scipy.spatial import distance

import matplotlib.pyplot as plt

# -------------- Data acquisition part


def getImgpaths(folder: str) -> dict:
    """
    Creates a dictionary with the location of all the images in the similarity
    matrix order.

    Parameters:
        folder: Location of the dataset.
    """

    locs = {}  # Dictionary with the locations
    unord_locs = {}
    # Gets all the images knowing all directories have the same structure.
    lvl_cam = {}  # Stores the amount of images in each cam folder
    n = 0
    for heir in sorted(os.walk(folder)):
        if heir[1] == []:  # if you are in the last folder of a tree
            lvl_cam[heir[0]] = len(heir[2])  # save the number of images
            for img in sorted(heir[2]):
                loc = heir[0] + '/' + img  # add to the path the img
                unord_locs[n] = loc  # and store the path
                n += 1

    # Order them in the similarity matrix order
    N = max(lvl_cam.values())
    nn = 0
    seq, cam = searchSeq(unord_locs)
    aa = '-'.join(list(unord_locs.values()))
    for s in seq:
        for n in range(2*N):
            n = str(n)
            i = '0'*(3-len(n)) + n + '.png'
            for c in cam:
                name = folder + s + c + '/' + i
                sear = re.search(name, aa)
                if sear is not None:  # check if is in the list
                    locs[nn] = name
                    nn += 1

    return locs


def searchSeq(my_dict):
    """
    Searches for the different levels inside the 
    folder tree.

    mydict: is the dictionary with the path of all the
        files in the folder tree.
    """
    lvl = []
    seq = []  # list of sequence folders
    cam = []  # list of cam folders
    pos = []
    values = my_dict.values()
    for name in values:
        for match in re.finditer('/', name):
            p = match.start()
            pos.append(p)  # find all positions of /
        for n in range(len(pos)-1):
            start = pos[n]
            end = pos[n+1]
            lvl.append(name[start:end])
        pos = []

    lvl = list(dict.fromkeys(lvl))
    for l in lvl:
        if re.search("Sequence", l):
            seq.append(l)
        if re.search("cam", l):
            cam.append(l)

    return seq, cam


def get_grps(folder):
    """
    Function 2: Input la similarity matrix y devuelve un diccionario que clasifica 
    en grupos las fotos que son parecidas según el input.
    """

    ini = folder.find('dataset/')
    name = folder[ini + 8: ini + 8 + 3]

    simil_path = folder + "/" + name + "_similarity.h5"
    with h5py.File(simil_path, "r") as f:
        # similarity labels in condensed form (shape=(1,n * (n-1) / 2))
        gt_labels = f["sim"][:].flatten()
        # similarity labels in matrix form (shape=(n, n))
        gt_labels = squareform(gt_labels)

    aa = []
    a = np.triu(gt_labels)
    b = a  # np.zeros(np.shape(a)) duplicate variable
    test_dict = {}  # create a test dictionary
    # test_dict = {'grp': 'matches'}  # create a test dictionary
    grp_count = 0
    aa = np.zeros(np.shape(a))  # initializes the mask creates a zero array

    for z in range(0, len(a)):  # number of rows increasing
        # gets the matches in each group, in row one gets matched column for each row
        temp = np.nonzero(b[z, :])
        # im setting all row values in b to false, temp is particular row
        b[temp, :] = False
        match_list = []

        if np.size(np.nonzero(temp)) > 0:  # it has a match
            #print(np.nonzero(temp))
            grp_count = grp_count+1

            for t in temp:
                # print(z,t)
                t = np.append(z, t)  # has all the values
                match_list.append(t)

            aa[grp_count][0:len(np.asarray(match_list)[0])] = np.asarray(
                match_list)  # converting dictionary into an array

    # you look into the array and get the index numbers of non duplicated values
    vals, ind = np.unique(aa, return_index=True)
    # another mask with all zeros again, same size as aa
    bb = np.zeros(np.shape(aa.flatten()))
    bb[ind] = 1  # bb of all unique values are iqual to 1
    bb = np.reshape(bb, np.shape(aa))  # shaping bb as the same shape as aa
    # multiplying the masks, all the unique values will be written
    cc = (bb*aa).astype(int)
    grp_count = 0  # the rows of the array cc are groups, the columns are the index values of all matched images in a group
    final_dict = {}
    # final_dict = {'grp': 'indices'}
    for n in range(0, len(a)-1):
        # gets all the index values which are matched for a particular row
        temp1 = np.nonzero(cc[n])
        if np.size(temp1) > 0:  # if there is a match
            # increase group count and add it to one group of the final dictionary
            if n == 1:
                temp1 = np.insert(temp1, 0, [0])
            final_dict['grp%d' % grp_count] = cc[n][temp1]
            grp_count = grp_count+1

    return final_dict


def setGroups(loc_dict: dict, grp_dict: dict) -> dict:
    """
    Creates a dictionary with the location of all the images by groups
    set by the similarity matrix.

    Parameters:
        loc_dict: Dictionary with the locations of the images in the
            similarity matrix order.
        grp_dict: Dictionary with the group information of the similarity
            matrix specifying the indexes of the images in each group.
    """
    loc_bygrp = {}

    for grp in grp_dict.keys():
        indexes = grp_dict[grp]
        loc_bygrp[grp] = []
        for ind in indexes:
            loc_bygrp[grp].append(loc_dict[ind])
        loc_bygrp[grp].sort()

    return loc_bygrp

# -------------- Data acquisition part


def get_images(folder) -> dict:
    """
    Get images from a folder and put them into dictionaries.
    """
    images = {}
    if folder == "ddataset/train":
        img_list = []
        dic = {"a": ["a01.png", "a02.png",
                     "a03.png", "a04.png", "a05.png"],
               "b": ["b02.png"],
               "c": ["c01.png"]}
        for k in dic.keys():
            for filename in dic[k]:
                path = folder + "/" + filename
                img = cv.imread(path, 0)  # Read in grayscale
                img_list.append(img)
            images[k] = img_list
            img_list = []
    else:
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
        self.param = dict(nfeatures=50, scaleFactor=1.2, nlevels=4,
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
        l = []  # a list for the features of each category
        descriptor_bycat = {}  # a dictionary of features by categories

        for labels, img_loc in images_dict.items():
            print(type(img_loc))
            if isinstance(img_loc, list):
                # if is a list then it is the dictionary with the classified images
                for n in range(len(img_loc)):
                    nimg_loc = img_loc[n]
                    nimg = cv.imread(nimg_loc, 0)
                    kp, des = orb.detectAndCompute(nimg, None)
                    # drawing keypoints
                    # tpm = np.copy(nimg)
                    # cv.drawKeypoints(nimg, kp, tpm, color=(0, 255, 0), Features2d.DRAW_RICH_KEYPOINTS)
                    # plt.imshow(tpm), plt.show()

                    descriptor_list.extend(des)
                    l.extend(des)
                descriptor_bycat[labels] = l
                l = []
            else:
                img = cv.imread(img_loc, 0)
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
            traininfo[cat] = len(features)
            allfeatureslist.extend(features)
        for cat, features in testbycat.items():
            testinfo[cat] = len(features)
            allfeatureslist.extend(features)

        # Normalize it
        allfeatureslist = preprocessing.normalize(allfeatureslist, norm='l2')

        # Get the lists of normalized values for train and test
        ntrain = sum(traininfo.values())
        ntest = sum(testinfo.values())
        trainlist = allfeatureslist[0:ntrain-1]
        testlist = allfeatureslist[ntrain:ntrain+ntest-1]

        # Organize normalized features by class for train and test
        for cat, nfeatures in traininfo.items():
            trainbycat_n[cat] = allfeatureslist[0:nfeatures]
            allfeatureslist = allfeatureslist[nfeatures:]
        for cat, nfeatures in testinfo.items():
            testbycat_n[cat] = allfeatureslist[0:nfeatures]
            allfeatureslist = allfeatureslist[nfeatures:]

        return trainlist, trainbycat_n, testlist, testbycat_n


class bow():
    def __init__(self):
        """ 
        Attributes:
            t: numerical threshold for classifying.
            vw_centers: list of the cluster centers for the visual words.
        """
        self.t = 0.6
        self.vw_centers = []
        self.nclusters = 150

    def setThreshold(self, threshold):
        """
        Sets a new classifying threshold.
        """
        self.t = threshold

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
            for n in range(len(features)):
                word = self.matchWord(features[n])
                histogram[0, word] += 1
            m = np.amax(histogram)
            histogram = histogram/m
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
        # distance.cdist(coords, coords, 'euclidean')
        l = len(centers)
        f = np.tile(feature, l)
        c = np.reshape(centers, (1, -1))
        d = np.linalg.norm(np.reshape(f-c, (l, -1)), axis=1)
        minind = np.argmin(d)
        # for n in range(centers.shape[0]):
        #     dist = np.linalg.norm(feature - centers[n, :])  # L2-norm
        #     if n == 0:
        #         mindist = dist
        #         minind = n
        #     elif dist < mindist:
        #         mindist = dist
        #         minind = n

        return minind

    def matchCategory(self, trainh: dict, testh: dict) -> dict:
        """
        Finds the matching category for every image in test dataset using 1NN
        Nearest Neightbour with k=1.

        Inputs:
            trainh: Dictionary of histograms by class, from training.
            testh: Dictionary of histograms by image, from test.
        Returns: 
            Dictionary of the test images classified.
        """
        img_classified = {}
        classification = np.nan

        flat_te = []
        for sublist in list(testh.values()):
            for item in sublist:
                flat_te.append(item)
        flat_tr = []
        for sublist in list(trainh.values()):
            for item in sublist:
                flat_tr.append(item)
        d = distance.cdist(flat_te, flat_tr, metric='euclidean')
        d = d/np.max(d)
        class_mat = d < self.t

        # classification = {}
        # for c in range(len(class_mat)):
        #     classification[c] = np.where(class_mat[c][:] > 0)

        print("--- Creating similarity Matrix: %s seconds ---" %
              (time.time() - start_time))
        similMat = self.createSimilarityMat(class_mat)

        # mask = np.empty(d.shape)
        # mask[:] = np.inf
        # mask[d < self.t] = 1
        # d_masked = np.multiply(d,mask)
        # class_mask = np.sum(~np.isinf(mask),axis=1)==0 # check if all elements in a row are inf
        # classification = np.argmin(d_masked, axis=1).astype(float)
        # classification[class_mask] = np.nan

        # keys = list(testh.keys())
        # img_classified = dict(zip(keys, classification.T))

        # similMat = self.createSimilarityMat(img_classified)

        return similMat

    def createSimilarityMat(self, classifications: dict):
        size = len(classifications)
        similarityMat = np.zeros([size, size])
        for c1 in range(size):
            aux = [classifications[c1]]*size
            match_list = np.multiply(aux, classifications)
            a = np.sum(match_list, axis=1) > 0
            a[c1] = 0
            similarityMat[c1][:] = a

        # for c1 in range(size):
        #     for c2 in range(size):
        #         n = np.isin(img_cats[c1], img_cats[c2])
        #         flag = np.sum(n)
        #         if flag>0 and c1 != c2:
        #             similarityMat[c1][c2] = 1

        return similarityMat

    def findThreshold(self, trainh: dict, testh: dict):
        """
        Finds ans sets a new threshold based on the distances 
        of the test and training histogram sets.
        """
        flat_te = []
        for sublist in list(testh.values()):
            for item in sublist:
                flat_te.append(item)
        flat_tr = []
        for sublist in list(trainh.values()):
            for item in sublist:
                flat_tr.append(item)
        d = distance.cdist(flat_te, flat_tr, metric='euclidean')
        distmin = np.min(d)
        distmean = np.mean(d)
        t = distmean - (distmean-distmin)/2

        # dist_list = np.array([])
        # for imgname, histtest in testh.items():
        #     for cat, hist in trainh.items():
        #         dist = np.linalg.norm(histtest - hist)  # L2-norm
        #         dist_list = np.append(dist_list, dist)
        # distmin = np.min(dist_list)
        # distmean = np.mean(dist_list)
        # t = distmean - (distmean-distmin)/2
        self.setThreshold(t)


def save_obj(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def main():
    start_time = time.time()
    train_folder = "dataset/W17"  # Location of the dataset.
    test_folder = "dataset/W17"

    pathtr = ("features/%s_tr.pkl" % (train_folder[-3:]))
    pathte = ("features/%s_te.pkl" % (test_folder[-3:]))

    if not os.path.exists('features'):
        os.makedirs('features')  # create a features dir

    # if os.path.exists(pathtr) and os.path.exists(pathte):
    #     pass
    # else:
    print("--- Obtaining images: %s seconds ---" % (time.time() - start_time))
    # Get the training dataset
    train_dict = getImgpaths(train_folder)
    grp_dict = get_grps(train_folder)
    loc_bygrp = setGroups(train_dict, grp_dict)
    # Get the test dataset
    test_dict = getImgpaths(train_folder)
    print("--- Images obtained: %s seconds ---" % (time.time() - start_time))

    # # Obtain images by name and by class from dataset
    # train_dset_bycat = get_images("ddataset/train")
    # # train_dset_byname = get_images("ddataset/train")
    # test_dset_byname = get_images("ddataset/test")  # take test images

    # Gets all the ORB features
    print("--- Obtaining features: %s seconds ---" %
          (time.time() - start_time))
    orb = orb_features()
    # if os.path.exists(pathtr) and os.path.exists(pathte):
    #     print("--- Loading.. ---")
    #     train_featuresbycat = load_obj(pathtr)
    #     test_featuresbycat = load_obj(pathte)
    # else:
    print("--- Calculating.. ---")
    # Gets all the features from the training images
    train_features, train_featuresbycat = orb.detectFromDict(loc_bygrp)
    save_obj(pathtr, train_featuresbycat)  # save it for later uses
    # Get all the features from the test images
    test_features, test_featuresbycat = orb.detectFromDict(test_dict)
    save_obj(pathte, test_featuresbycat)  # save it for later uses

    print("--- Features ontained: %s seconds ---" % (time.time() - start_time))
    # Normalize them
    train_features, train_featuresbycat, test_features, test_featuresbycat = \
        orb.normalizeAllFeatures(train_featuresbycat, test_featuresbycat)

    # Initialize BOW
    if os.path.exists("features/bowc.pkl"):
        bow_obj = load_obj("features/bowc.pkl")
    else:
        print("--- Beginning BOW: %s seconds ---" % (time.time() - start_time))
        bow_obj = bow()
        # Clusters the features using kmeans
        print("--- Computing centers: %s seconds ---" %
              (time.time() - start_time))
        bow_obj.kmeans(train_features)
        print("--- Centers computed: %s seconds ---" %
              (time.time() - start_time))

    # Compute histograms for every category in train dataset
    print("--- Computing histograms for train: %s seconds ---" %
          (time.time() - start_time))
    train_histogrambycat = bow_obj.compute_histogram(train_featuresbycat)

    # Plot the histograms and/or the frequency for each cluster center
    # top10 = []
    # sumh = np.zeros((1, 150))
    # for grp in train_histogrambycat.keys():
    #     top10.append(np.argpartition(train_histogrambycat[grp][0], -10)[-10:])
    #     sumh = sumh + train_histogrambycat[grp][0]

    # t = np.reshape(top10, (-1,))
    # n, bins, patches = plt.hist(list(range(0,150)), weights=sumh.transpose(), bins=150, color='#0504aa', alpha=0.7)
    # # n, bins, patches = plt.hist(t, bins=150, color='#0504aa', alpha=0.7)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Feature n')
    # plt.ylabel('Frequency')
    # # plt.title('Group 0')
    # plt.show()

    # Compute histrograms for every image in test dataset
    print("--- Computing histograms for test: %s seconds ---" %
          (time.time() - start_time))
    test_histogrambycat = bow_obj.compute_histogram(test_featuresbycat)

    # Find new threshold
    # bow_obj.findThreshold(train_histogrambycat, test_histogrambycat)
    # Classify the test images into the trainned categories
    print("--- Classifying test images: %s seconds ---" %
          (time.time() - start_time))
    classification_results = bow_obj.matchCategory(
        train_histogrambycat, test_histogrambycat)
    print("Classification results: ")
    print(classification_results)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

# To DO: Normalize or not? What number of features? What number of cluster centers?
#        Does it work with several entries for a class?
