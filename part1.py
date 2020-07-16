import numpy as np
import os
import cv2 as cv
import h5py
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
sequence_folder = "/Volumes/TOSHIBA EXT/Python/W17"



def ImageClass(sequence_folder):
    n=0
    pic_dic={'index':'image'}
    pic_index={'index':'file'}
    pic_no=0
    seq_no=0
    for heir in sorted(os.walk(sequence_folder)):
        print(heir)        #(heir[2][len(heir)-2])

    
        
        #f_no=n-1
        #print(f_no)

        if n == 0:
            root=heir[0]
            nlvl_1f= len(heir[1])
            lvl_1f= heir[1]
            lvl_1s=heir[1][0]
        if n ==1:
            nlvl_2f= len(heir[1]) 
            lvl_2f= heir[1]
            total_f= nlvl_1f*nlvl_2f
            lvl_2s=heir[1][0]
        if n >1: 
            print(len(root)+len(lvl_1s)+len(lvl_2s)+1)
            if len(heir[0])==len(root)+len(lvl_1s)+1 :
                print('seq')
                seq_no=pic_no
            elif len(heir[0]) == len(root)+len(lvl_1s)+len(lvl_2s)+2 :
                f_no=(int(heir[0][len(heir[0])-1]))
                local_no=0
                print(f_no)
                for pic in heir[2]:
                    if pic[0] == '.' :
                        pass
                    else :
                    #             #print(heir[2])
                                 pic_no = pic_no+1
                                 pic_index[(nlvl_2f*local_no)+f_no+seq_no]=['f0%d%s' % (f_no, pic)]

                                 local_no = local_no+1
                    
        
        n =n+1
    
    return pic_index


with h5py.File("/Volumes/TOSHIBA EXT/Python/W17/W17_similarity.h5", "r") as f:
    gt_labels = f["sim"][:].flatten() #similarity labels in condensed form (shape=(1,n * (n-1) / 2))
    # similarity labels in matrix form (shape=(n, n))
    gt_labels = squareform(gt_labels)




def get_grps(gt_labels):
        aa = []
        #images= ImageClass(sequence_folder)
        a= np.triu(gt_labels)
        b=a#np.zeros(np.shape(a))
        test_dict={'grp':'matches'}
        grp_count=0
        aa = np.zeros(np.shape(a))
        for z in range(0,len(a)):
            
            temp=np.nonzero(b[z,:])
            b[temp,:] = False
            #b.append(np.squeeze(temp) )
            #print(np.size(np.nonzero(b[temp,:])))
            match_list = []
            
            if np.size(np.nonzero(temp)) >0:
                #print(np.nonzero(temp))
                grp_count = grp_count+1
                for t in temp:
                    #print(t)
                    #t=images['%d'%t]
                    #z=images['%d'%z]
                    #print(z,t)
                    t=np.append(z,t)
                    #print(t)
                    match_list.append(t)

                #test_dict['grp%d'%grp_count]=match_list
                #print(len(np.asarray(match_list)[0]))
                aa[grp_count][0:len(np.asarray(match_list)[0])] = np.asarray(match_list)
                #print(aa[grp_count])

        vals,ind= np.unique(aa,return_index=True)
        bb=np.zeros(np.shape(aa.flatten()))
        bb[ind] = 1
        bb= np.reshape(bb,np.shape(aa))
        cc = bb*aa.astype(int)
        grp_count=0
        final_dict ={'grp':'indices'}
        for n in range(0,len(a)-1) :
            temp1=np.nonzero(cc[n])
            if np.size(temp1) > 0 :
                final_dict['grp%d'%grp_count] = cc[n][temp1]
                grp_count = grp_count+1
        return final_dict

