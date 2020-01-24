import os
import cv2
import numpy as np
import SimpleITK as sitk
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

############################# data processing ################################

def correct_bias(in_file, out_file=None, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    if out_file == None:
        out_file = in_file.rstrip('.nii')+"_bias_correct.nii"
    #使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)

def correct_bias_itk(in_file, out_file=None, image_type=sitk.sitkFloat64):
    if out_file == None:
        out_file = in_file.rstrip('.nii')+"_bias_corrected.nii"
    inputImage = sitk.ReadImage(in_file)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    output = corrector.Execute(inputImage, maskImage)
    sitk.WriteImage(output, out_file)
    return out_file

def normalization(x):
    mean = np.mean(x)
    std = np.std(x)
    out = (x-mean)/std
    print("Normalization done: mean="+str(mean)+", std="+str(std)+", input: "+str(x.dtype)+" output: "+str(out.dtype))
    return out

def norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    vmean = np.mean(x)
    delta = vmax - vmin
    out = (x-vmin)*255/delta
    print("Normalization done: max="+str(vmax)+", min="+str(vmin)+", mean="+str(vmean)+", dtype="+str(out.dtype))
    return out

def maxmin_norm(x):
    vmax = np.max(x)
    vmin = np.min(x)
    vmean = np.mean(x)
    delta = vmax - vmin
    out = (x-mean)/delta
    print("Normalization done: max="+str(vmax)+", min="+str(vmin)+", mean="+str(vmean)+", dtype="+str(out.dtype))
    return out

def pad(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image
    if output size is smaller"""
    delta_h = shape[0]-img.shape[0]
    delta_w = shape[1]-img.shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = np.vstack((np.zeros((up,img.shape[1]), dtype=np.float32), img))
        img = np.vstack((img, np.zeros((down,img.shape[1]), dtype=np.float32)))
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = np.hstack((np.zeros((img.shape[0],left), dtype=np.float32), img))
        img = np.hstack((img, np.zeros((img.shape[0], right), dtype=np.float32)))
    return img

def crop(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image
    if output size is smaller"""
    delta_h = img.shape[0]-shape[0]
    delta_w = img.shape[1]-shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = img[up:-down,:]
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = img[:,left:-right]
    return img

def pad_crop(img, shape):
    """aplly padding and cropping to resize the current image without rescaling"""
    img = pad(img, shape)
    img = crop(img, shape)
    return img

def resize_slices_cxy(slices, shape):
    resized = np.empty((slices.shape[0], shape[0], shape[1]), dtype=np.float32)
    for i in range(slices.shape[0]):
        resized[i] = pad_crop(slices[i], shape)
    return resized

def resize_slices_xyc(slices, shape):
    resized = np.empty((shape[0], shape[1], slices.shape[2]), dtype=np.float32)
    for i in range(slices.shape[2]):
        resized[:,:,i] = pad_crop(slices[:,:,i], shape)
    return resized

############################ read & wrtie ####################################

def read_nii(nii_path):
    itkimage = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(itkimage)
    spacing = itkimage.GetSpacing()
    origin = itkimage.GetOrigin()
    direction = itkimage.GetDirection()
    print("Read array from: " + nii_path + ", data type: " + str(np.dtype(img[0][0][0])))
    return img, spacing, origin, direction

def save_as_nii(path, img, spacing=None, origin=None, direction=None):
    itkimage = sitk.GetImageFromArray(img)
    if spacing != None:
        itkimage.SetSpacing(spacing)
    if origin != None:
        itkimage.SetOrigin(origin)
    if direction != None:
        itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, path)
    print("Images saved as: "+path+' , dtype='+str(img.dtype))
    return path

########################## loss function ####################################
def mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float"))**2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

