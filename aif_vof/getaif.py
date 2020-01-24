from lupy import *
from lufil import *
import os
import time
import sys
import math
import shutil
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt

# find files
if len(sys.argv) == 2:
    fname = sys.argv[1]
else:
    fname = input('Please enter the patient\'s name: ')
path = '../data/dcm'
save_path = '../data/mat_data'
files = fsearch(path = path,
                suffix='.nii',
                include=fname,
                sort_level=-1)
print(len(files),"files found:")
print('\n'.join(files))

for fin in files:
    if not os.path.exists('../data/mat_data/'+fname+'.png'):
        print('\n', fin)
        img = sitk.ReadImage(fin)
        arr = sitk.GetArrayFromImage(img)
        print('shape: ', arr.shape)
        print('spacing: ', img.GetSpacing())
        print('origin: ', img.GetOrigin())
        print('direction: ', img.GetDirection())
        print("original: max="+str(np.max(arr))+", min="+str(np.min(arr))+", dtype="+str(arr.dtype))

        arr_roi = arr
        # sampling rate
        n_slices = arr_roi.shape[1]
        switcher = {
            1: 2,
            2: 3,
        }
        step = switcher.get(n_slices, 6)

        z_step = 1
        if n_slices > 5:
            z_step = math.ceil(n_slices/5)

        shape1 = range(0, n_slices, z_step)
        shape2 = range(0, arr_roi.shape[2], step)
        shape3 = range(0, arr_roi.shape[3], step)
        total = len(shape1) * len(shape2) * len(shape3)
        c = 0
        q = 0

        ttp = []
        for z in shape1:
            for i in shape2:
                for j in shape3:
                    c += 1
                    loc = (z,i,j)
                    print('\n'+fin.split('/')[-3]+': '+str(n_slices)+' slices ')
                    print(str(c)+'/'+str(total)+' '+str(loc))

                    # 提取信号
                    sig = arr_roi[:,z,i,j]
                    l = len(sig)
                    #sig = interpolate(np.arange(0,len(sig)), sig)

                    pre = 5
                    base = np.mean(sig[0:pre])
                    if base < 0 or base > 120:
                        q += 1
                        print('blank', q)
                    else:
                        plt.subplot(311)
                        plt.plot(sig)
                        # MedianAerageFitering
                        sig = MyMedianAverage(sig, 5)
                        sig = MyMedianAverage(sig, 5)
                        sig = sig - np.mean(sig[0:pre])

                        #plt.subplot(222)
                        #plt.plot(sig)

                        idx_max = np.argmax(sig)
                        w = 5
                        if w < 2: w = 2
                        if l*0.20 <= idx_max < l*0.8:
                            idx1 = idx_max - w
                            idx2 = idx_max + w
                            bo = sig[idx1:idx2]
                            ot1 = sig[idx_max-2*w:idx_max]
                            ot2 = sig[idx_max:idx_max+2*w]
                            ot = np.hstack((sig[0:idx1],sig[idx2:-1]))
                            if np.sum(ot1) < np.sum(bo):
                               if np.sum(ot2) < np.sum(bo):
                                   if np.sum(bo) > np.sum(sig)*0.25:
                                       plt.subplot(312)
                                       plt.plot(sig)
                                       sig = np.append(sig, z)
                                       sig = np.append(sig, i)
                                       sig = np.append(sig, j)
                                       ttp.append(sig)

        l = len(ttp)
        print(l)
        ttp = sorted(ttp, key=lambda x:np.max(x[0:-3]), reverse=True)
        ttps = sorted(ttp[0:int(l*0.3)], key=lambda x:int(np.argmax(x[0:-3])))
        l = len(ttps)
        aifs = sorted(ttps[0:int(l*0.1)], key=lambda x:np.max(x[0:-3]), reverse=True)
        vofs = sorted(ttps[-(int(l*0.1)):-1], key=lambda x:np.max(x[0:-3]), reverse=True)
        ttps = np.asarray(ttps)
        print(ttps.shape)
        aifcor = (int(aifs[0][-3]), int(aifs[0][-2]), int(aifs[0][-1]))
        vofcor = (int(vofs[0][-3]), int(vofs[0][-2]), int(vofs[0][-1]))
        print(aifcor)
        print(vofcor)
        plt.subplot(313)
        #aif = arr[:,aifcor[0],aifcor[1],aifcor[2]]
        aif = np.mean(aifs[0:3], axis=0)[0:-3]
        #vof = arr[:,vofcor[0],vofcor[1],vofcor[2]]
        vof = np.mean(vofs[0:3], axis=0)[0:-3]
        if np.max(vof) < np.max(aif):
            vof = np.mean(ttp[0:3], axis=0)[0:-3]
        # aif = MyMedianAverage(aif, 5)
        # vof = MyMedianAverage(vof, 5)
        plt.plot(aif)
        plt.plot(vof)
        #plt.plot(arr[:,aifcor[0],aifcor[1],aifcor[2]])
        #plt.plot(arr[:,vofcor[0],vofcor[1],vofcor[2]])
        #plt.plot(np.mean(aifs[0:1], axis=0))
        #plt.plot(np.mean(vofs[0:1], axis=0))

        sio.savemat(save_path+'/'+fname+'.mat', mdict={'img_mat':arr, 'AIF':aif, 'VOF': vof, 'aif_z':aifcor[0], 'aif_x':aifcor[1], 'aif_y':aifcor[2],'vof_z':vofcor[0], 'vof_x':vofcor[1], 'vof_y':vofcor[2] })
        plt.savefig(save_path+'/'+fname+'.png')
        plt.cla()
        plt.clf()
