import os
import cv2
import PIL
import pydicom
import numpy as np
from pydicom.filereader import read_dicomdir
from fastai.vision import pil2tensor, Image


def extract_dcm_from_dir(dicomdir_path):
    """
    extract dcm from the dicomdir file. Visit all possible images inside
    the DICOM directory

    input:
    - dicomdir_path: path to dicomdir

    output:
    - a list of dcm from the dicom dir
    """

    input_path, _ = os.path.split(dicomdir_path)
    dicomdir = read_dicomdir(dicomdir_path)

    im_list = []

    # ugly :(
    for patient_record in dicomdir.patient_records:
        for study in patient_record.children:
            for series in study.children:
                for im in series.children:
                    dcm_file = pydicom.dcmread(
                        os.path.join(input_path, *im.ReferencedFileID))
                    im_list.append(dcm_file)
    return im_list


def check_dcm_inversion(dcm_in):
    """
    Check wether the input scan intensities are inverted or not
    based on DICOM information:
    - PresentationLUTShape == 'INVERTED'
    - PhotometricIntepretation = 'MONOCHROME'
    """
    inverted = False

    try:
        print(dcm_in.PresentationLUTShape, end=' ')
        if dcm_in.PresentationLUTShape == 'INVERSE':
            inverted = True
    except:
        pass

    try:
        print(dcm_in.PhotometricInterpretation, end=' ')
        if dcm_in.PhotometricInterpretation == 'MONOCHROME1':
            inverted = True
    except:
        pass

    return inverted


def normalize_scan(im):
    """
    Normalize scan
    """
    return ((im / im.max()) * 255).astype('uint8')


def load_image(image_path):
    """
    Load the input images. `image_path` can be an image, a DICOMDIR
    or a dcm file

    input:
    - image_path: path to image or dcmdir

    output:
    - tensor_images: list of Pytorch tensors ready for inference

    """

    # check if the image is already an image, a DICOMDIR or a DCM file
    im_list = []
    dcm_list = []
    scan_name = []
    tensor_images = []
    
    #('.png', '.jpg', '.jpeg', '.tiff', '.bmp')#
    valid_images = [".jpg","jpeg",".tiff",".png",".bmp"]
    # if it is a folder (get all images from folder)
    if (os.path.isdir(image_path)):
        #print('isdir')
        for f in os.listdir(image_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() in valid_images:
                im = cv2.imread(os.path.join(image_path,f))
                im = normalize_scan(im)
                im_list.append(im)
                scan_name.append(f)
                #debug print('loaded '+os.path.join(image_path,f))
    elif os.path.splitext(image_path)[1].lower() in valid_images:
        # input scan is an image
        im = cv2.imread(image_path)
        im = normalize_scan(im)
        im_list.append(im)
        scan_name.append(os.path.split(image_path)[1])

    elif 'DICOMDIR' in image_path:
        dcm_list += extract_dcm_from_dir(image_path)
    else:
        # use pydicom to load dcm files
        try:
            dcm_list.append(pydicom.dcmread(image_path))
        except:
            return tensor_images.append(False)

    # process dcm scans, check if inverted and process instensities
    i=0
    for dcm in dcm_list:

        try:
            im = dcm.pixel_array
            im = normalize_scan(im)
        
            # check intensities
            if check_dcm_inversion(dcm):
                im = ~im
        except:
            im = False

        im_list.append(im)
        # adding something like series number
        scan_name.append(os.path.split(image_path)[1]+str(i)) 
        print(os.path.split(image_path)[1]+str(i))
        i = i+1

    # trasnform the image to uint8 rgb

    # convert images to tensors
    for im in im_list:
        if im is not False:
            x = PIL.Image.fromarray(im).convert('RGB')
            tensor_images.append(Image(pil2tensor(x, np.float32).div_(255)))
        else:
            tensor_images.append(False)
    return tensor_images, scan_name

def crop_image(image, bbox):
    """
    Crop image using bbox

    input:
    - image
    - bounding box (minx, miny, maxx, maxy) 

    output:
    - image cropped. 
    
    """
    # left, top right bottom
    return image.crop((bbox[0],bbox[1],bbox[2],bbox[3]))



def save_timage(t_image, image_path):
    """
    Write the tensor_images into the image_path . `image_path` should be an image file path (with extension),
     tensor_image image content in tensor format.  

    input:
    - t_image: tensor image
    - image_path: path to image 

    output:
    
    """
    image = transforms.ToPILImage()(t_image).convert("RGB")
    save_image(image, image_path)


def save_image(image, image_path):

    img_format = os.path.splitext(image_path)[1].upper()
    if (not img_format): 
        image_path = image_path +'.png'
    try:
        image.save(image_path) 
    except:
        print("Error saving "+image_path)
    
