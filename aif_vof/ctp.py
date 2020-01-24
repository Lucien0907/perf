from lupy import *
from lufil import *
import os
import sys
import time
import math
import shutil
import multiprocessing
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import interpolate

# CTP parameters
PRE = 2
POST = 50
kappa = 0.73
loth = 0
hith = 100
rho = 1.04
fsize = 5
ftsize = 5

# SVD parameters
lamb = 0.3
m = 2

# set path
data_path = '../data/dcm'
output_path = '../data/output'

# Select patients
if len(sys.argv) == 2:
    patient_name = sys.argv[1]
else:
    patient_name = input('Please enter the patient\'s name: ')
files = fsearch(path = data_path, suffix='.nii', include=patient_name, sort_level=-1)
fname = files[0]

# Create output folder if not existed
output_path = os.path.join(output_path, patient_name)
if len(files) == 1 and not os.path.exists(output_path):
    os.makedirs(output_path)

# load data
arr_raw = sitk.GetArrayFromImage(sitk.ReadImage(fname))
print('shape:', arr_raw.shape, ', dtype:', arr_raw.dtype,"max=", np.max(arr_raw),", min=", np.min(arr_raw))

#----------------------------------------------------------------------------------#

# parameters for interpolation
[nt, nz, nx, ny] = arr_raw.shape
nt_new = 2*nt-1
times = np.linspace(0, nt-1, nt)
times_new = np.linspace(0, nt-1, nt_new)

# Interpolation funtion
def itp(args):
    [sig, times, times_new] = args
    f = interpolate.interp1d(times, sig, kind='linear', fill_value="extrapolate")
    return f(times_new)

# run interpolation with multiprocessing
start = time.time()
args = [(arr_raw[:,z,x,y],times,times_new) for z in range(nz) for x in range(nx) for y in range(ny) ]
p = multiprocessing.Pool(6)
arr = p.map_async(itp, args).get()
p.close()
p.join()
tcost = time.time() - start
print(int(tcost/60),'mins', int(tcost%60),'seconds')

# convert result to array
arr = np.array(arr, dtype=np.int16)
arr = arr.reshape(nz,nx,ny,nt_new)
arr = np.moveaxis(arr, -1, 0)
print('shape:',arr.shape, 'dtype:',arr.dtype, 'max=', np.max(arr), 'min=', np.min(arr))

# save interpolated array
plt.subplot(121)
plt.plot(arr_raw[:,4,5,6])
plt.subplot(122)
plt.plot(arr[:,4,5,6])
plt.savefig(os.path.join(output_path,patient_name)+'.png')
plt.show()

#--------------------- Automatic AIF/VOF selection -------------------------------#

nt = nt_new
pre = 5
plt.cla()
plt.clf()

# fucntion for auto-selecting aif/vof
def autoaif(sig):
    pre = 5
    w = 5
    base = np.mean(sig[0:pre])
    r = np.zeros(59, dtype=np.int16)
    if 0 < base < 120:
       #plt.subplot(311)
       #plt.plot(sig)
       sig = MyMedianAverage(sig, 5)
       sig = sig - base
       imax = np.argmax(sig)
       if nt*0.2 < imax < nt*0.8:
           idx1 = imax - w
           idx2 = imax + w
           bo = sig[idx1:idx2]
           ot1 = sig[imax-2*w:imax]
           ot2 = sig[imax:imax+2*w]
           ot = np.hstack((sig[0:idx1],sig[idx2:-1]))
           if np.sum(ot1) < np.sum(bo):
               if np.sum(ot2) < np.sum(bo):
                   if np.sum(bo) > np.sum(sig)*0.25:
                       #plt.subplot(312)
                       #plt.plot(sig)
                       r = sig
    return r

z_step = 1
if n_slices > 5:
    z_step = math.ceil(n_slices/5)

args = [arr[:,z,x,y] for z in range(0,nz,z_step) for x in range(0,nx,4) for y in range(0,ny,4) ]
start = time.time()
p = multiprocessing.Pool(6)
sigs = p.map_async(autoaif, args).get()
p.close()
p.join()
tcost = time.time() - start
print(int(tcost/60),'mins', int(tcost%60),'seconds')

for x in sigs:
    plt.plot(x)
plt.show()

# convert result to array
# arr = np.array(arr, dtype=np.int16)
# arr = arr.reshape(nz,nx,ny,nt_new)
# arr = np.moveaxis(arr, -1, 0)
# print('shape:',arr.shape, 'dtype:',arr.dtype, 'max=', np.max(arr), 'min=', np.min(arr))
