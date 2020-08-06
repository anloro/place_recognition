import numpy as np
import os
import cv2 as cv
import h5py
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
sequence_folder = "dataset/W17"  # Path del dataset

def ImageClass(sequence_folder):
    """
    First function, dictionary1: gets the sequence folder and returns a dictionary 
    with the picture name (pic_index) and another dictionary with picture itself (pic_dic).
    """
    n=0 #starting count to see the level, sequence, camera, files
    pic_dic={}
    pic_index={} # file name
    # pic_dic = {'index': 'image'}
    # pic_index = {'index': 'file'}  # file name
    pic_no=0 #it counts number of pictures 
    seq_no=0 #counts sequences

    for heir in sorted(os.walk(sequence_folder)): #heir is heirrarchy, os.walk is a function that read the files of the sequence folder, and it gets the name
        if n == 0: #w17 folder
            root=heir[0] #this returns the adress (the folders where the pictures are there),subfoldername and filename
            nlvl_1f= len(heir[1]) #numberlevel1folder, how many sequence folders are on the whole thing
            lvl_1f= heir[1] #names of all the folders (sequence1,2,3,4,5,6,...)
            lvl_1s=heir[1][0] #gets the name of the first folder (sequence 0)
        if n ==1: #we are in the sequence folder
            nlvl_2f= len(heir[1]) 
            lvl_2f= heir[1]
            total_f= nlvl_1f*nlvl_2f #gives total number of folders, if we have 17 sequence folders each one has 5 folders (total number of camera folders)
            lvl_2s=heir[1][0] #gives the name of the first camera folder
        if n >1: #read all the individual files (pictures and matrices)
            if len(heir[0])==len(root)+len(lvl_1s)+1 :
                seq_no=pic_no
            elif len(heir[0]) == len(root)+len(lvl_1s)+len(lvl_2s)+2 :
                f_no=(int(heir[0][len(heir[0])-1])) #im getting folder number from the string, so you get the last letters (camera01 would be 01)
                local_no=0
                for pic in heir[2]: #to solve mac problems
                    if pic[0] == '.' :
                        pass
                    else:
                        pic_no = pic_no+1
                        file_loc= os.path.join(heir[0],pic)
                        # img=cv.imread(file_loc)
                        # pic_dic['0%d'%((5*local_no)+f_no+seq_no)] =img
                        pic_index[(nlvl_2f*local_no)+f_no+seq_no]=['f0%d%s' % (f_no, pic)]
                        pic_dic[(nlvl_2f*local_no)+f_no+seq_no] = file_loc
                        local_no = local_no+1
        
        n =n+1

    return pic_index,pic_dic


simil_path = sequence_folder + "/W17_similarity.h5"
with h5py.File(simil_path, "r") as f: 
    gt_labels = f["sim"][:].flatten() #similarity labels in condensed form (shape=(1,n * (n-1) / 2))
    gt_labels = squareform(gt_labels) #similarity labels in matrix form (shape=(n, n))


def get_grps(gt_labels):
    """
    Function 2: Input la similarity matrix y devuelve un diccionario que clasifica 
    en grupos las fotos que son parecidas según el input.
    """
    aa = []
    a= np.triu(gt_labels)
    b=a#np.zeros(np.shape(a)) duplicate variable
    test_dict = {}  # create a test dictionary
    # test_dict = {'grp': 'matches'}  # create a test dictionary
    grp_count=0
    aa = np.zeros(np.shape(a)) #initializes the mask creates a zero array

    for z in range(0,len(a)): #number of rows increasing    
        temp=np.nonzero(b[z,:]) #gets the matches in each group, in row one gets matched column for each row
        b[temp,:] = False #im setting all row values in b to false, temp is particular row
        match_list = []
        
        if np.size(np.nonzero(temp)) >0: #it has a match
            #print(np.nonzero(temp))
            grp_count = grp_count+1

            for t in temp:
                # print(z,t)
                t=np.append(z,t) #has all the values
                match_list.append(t)

            aa[grp_count][0:len(np.asarray(match_list)[0])] = np.asarray(match_list) #converting dictionary into an array

    vals,ind= np.unique(aa,return_index=True) #you look into the array and get the index numbers of non duplicated values
    bb=np.zeros(np.shape(aa.flatten())) #another mask with all zeros again, same size as aa
    bb[ind] = 1 #bb of all unique values are iqual to 1
    bb= np.reshape(bb,np.shape(aa)) #shaping bb as the same shape as aa
    cc = (bb*aa).astype(int) #multiplying the masks, all the unique values will be written
    grp_count=0 #the rows of the array cc are groups, the columns are the index values of all matched images in a group
    final_dict ={}
    # final_dict = {'grp': 'indices'}
    for n in range(0,len(a)-1) :
        temp1=np.nonzero(cc[n]) #gets all the index values which are matched for a particular row
        if np.size(temp1) > 0 : #if there is a match
            final_dict['grp%d'%grp_count] = cc[n][temp1] #increase group count and add it to one group of the final dictionary
            grp_count = grp_count+1
    return final_dict

#Función 3: Relaciona las imagenes con los grupos

def matched_files(sequence_folder,gt_labels):
    pic_index,pic_dic = ImageClass(sequence_folder)
    test_dict = get_grps(gt_labels)
    matched_file=[]
    for n in range (0,len(test_dict)): #to number of groups in test dicti
        temp= test_dict['grp%d'%(n)] #getting the values of a particular group and put them in temp
        for n1 in range(0,np.size(temp)): #count the number of matched files
            #print(n1)
            matched_file=np.append(matched_file,pic_dic['0%d'%(n1+1)])#pic_dic["0%d"%pic_index[n1+1]]) 
        print(matched_file) #printing the matched file

for ii in range(len(gt_labels[0])):
    print(gt_labels[0][ii])
pic_index, pic_dic = ImageClass(sequence_folder)
# print(pic_index)
# print(pic_dic)
final_dict = get_grps(gt_labels)
pass
# print(final_dict)
# matched_files(sequence_folder, gt_labels)
for grp in final_dict.keys():
    indexes = final_dict[grp]
    for ind in indexes: 
        img_path = pic_dic[ind]
        img = cv.imread(img_path)
        cv.imshow('img in grp %s'%(grp), img)
        cv.waitKey()
        
    break

