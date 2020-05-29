import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.spatial import distance

def get_images(folder):
    """
    Get images from a folder and put them into dictionaries.
    """
    images = {}
    for filename in os.listdir(folder):
        category = filename # Dictionary name is each image name
        path = folder + "/" + filename
        img = cv.imread(path, 0) # Read in grayscale
        images[category] = img

    return images

train_dset = get_images("ddataset/train")
test = get_images("ddataset/test")  # take test images

def get_sift_features(images):
    """
    Get SIFT features from a dictionary of images and save
    them in a new dictionary.
    """
    sift_vectors = {}
    descriptor_list = []
    sift = cv.xfeatures2d.SIFT_create()
    features = []
    for cat, img in train_dset.items():
        # cv.imshow('image', images)
        # cv.waitKey(1000)
        kp, des = sift.detectAndCompute(img, None)
        descriptor_list.extend(des)
        features.append(des)
        sift_vectors[cat] = features

        # img2 = cv.drawKeypoints(img, kp, np.array([]))
        # cv.imshow('Kp', img2)
        # cv.waitKey(1000)
        
    return [descriptor_list, sift_vectors]


train_features = get_sift_features(train_dset)
# Takes the descriptor list which are all the descriptors unordered
train_descriptor_list = train_features[0]
# Takes the sift descriptors separated in categories for train
train_descriptors_bycat = train_features[1]
# Takes the sift descriptors separated in categories for test
test_descriptors_bycat = get_sift_features(test)[1]

def kmeans(k, descriptor_list):
    """
    A k-means clustering algorithm.
    
    Takes 2 parameter:
    k: Number of cluster(k).
    descriptor_list: Descriptors list(unordered 1d array).

    Returns: An array that holds central points.
    """
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_

    return visual_words


# Takes the central points which is visual words
visual_words = kmeans(150, train_descriptor_list)

def image_class(all_bovw, centers):
    """
    Computes histograms of images.

    Takes 2 parameters:
    all_bovw: Dictionary that holds the descriptors that are 
    separated class by class.
    descriptor_list: Array that holds the central points 
    (visual words) of the k means clustering.

    Returns: A dictionary that holds the histograms for 
    each image separated class by class.
    """
    dict_feature = {}
    for key, value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

def find_index(image, center):
    """
    Find the index of the closest central point 
    to each sift descriptor.
    
    Takes 2 parameters:
    image: A sift descriptor.
    center: The array of central points in k means.
    
    Returns: the index of the closest central point.
    """
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


# Creates histograms for train data
bovw_train = image_class(train_descriptors_bycat, visual_words)
# Creates histograms for test data
bovw_test = image_class(test_descriptors_bycat, visual_words)


def knn(images, tests):
    """
    1-NN algorithm. Prediction of the class of test images.
    
    Takes 2 parameters: 
    images: Feature vectors of train images
    tests: Feature vectors of test images

    Returns: Array that holds number of test images, 
    number of correctly predicted images and 
    records of class based images respectively
    """
    num_test = 0
    correct_predict = 0
    class_based = {}

    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0]  # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a"  # predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        #minimum = L1_dist(tst,train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        #dist = L1_dist(tst,train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key

            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]


# Call the knn function
results_bowl = knn(bovw_train, bovw_test)

def accuracy(results):
    """
    Calculates the average accuracy and class based accuracies.
    """
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key, value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))


# Calculates the accuracies and write the results to the console.
accuracy(results_bowl)

bump = 0
