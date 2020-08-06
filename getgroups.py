import os
import h5py
from scipy.spatial.distance import squareform
import numpy as np
import cv2 as cv


def getImgpaths(folder: str) -> dict:
    """
    Creates a dictionary with the location of all the images in the similarity
    matrix order.

    Parameters:
        folder: Location of the dataset.
    """
    
    locs = {} # Dictionary with the locations
    unord_locs = {}
    # Gets all the images knowing all directories have the same structure.
    lvl = {}
    n = 0
    for heir in sorted(os.walk(folder)):
        if heir[1] == []:
            lvl[heir[0]] = len(heir[2])
            for img in sorted(heir[2]):
                loc = heir[0] + '/' + img
                unord_locs[n] = loc
                n += 1
    # Order them in the similarity matrix order
    N = max(lvl.values())
    n = 0
    for nn in range(N):
        offset = 0
        for c in lvl.keys():
            if nn < lvl[c]:
                n_order = nn + offset
                locs[n] = unord_locs[n_order]
                n += 1
            offset += lvl[c]
    
    return locs


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


folder = "dataset/W17"  # Location of the dataset.

loc_dict = getImgpaths(folder)
grp_dict = get_grps(folder)
loc_bygrp = setGroups(loc_dict, grp_dict)