# ### Inference of DBT lesion detection 
# 
# Input 
# - csv with the path to the images
# 
# Output
# - csv with lesion bounding boxes
# #
# Robert Marti 2021
# robert.marti@udg.edu
#   TODO: 
# --------------------------------------------------


import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from duke_dbt_data import dcmread_image, read_boxes, draw_box,evaluate
import omidb
import numpy as np
import cv2
import csv
import torch, torchvision
import tensorflow
print(torch.__version__, torch.cuda.is_available())

assert torch.__version__.startswith("1.7")
import detectron2
from detectron2.utils.logger import setup_logger
from matplotlib import pyplot as plt
import numpy as np
import json, cv2, random


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import argparse

import os
import sys
import itertools

# detr stuff
sys.path.insert(1, os.path.join(sys.path[0], '/home/robert/src/detr/detr'))


from d2.detr import DetrDatasetMapper, add_detr_config


setup_logger()

##############################
# get the breast bounding box
#
##############################
def get_normal_BBox (image):
    #threshold image 
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape,dtype=np.uint8)
    img2[output == max_label] = 255

    contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    aux_im = img2
    x,y,w,h = cv2.boundingRect(cnt)
    out_bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)
    
    return out_bbox, img2 # returns bounding box and mask image. 

        
##############################
# predict slice 
#
##############################
def predict_slice(view, client, episode,image_rgb,side): 
   
    # filename =client+"_"+episode+"_"+view+"_"+str(slice_index)+"_rgb.png"
    # print(filename)
    
    # print(image_rgb.shape)
    
    if (image_rgb.ndim>2): # contains rgb image
        comp = image_rgb.shape[0]
    else: comp = 1
    
    for nc in range(comp):       
        if comp ==1: image = image_rgb # image of only component.
        else: image =  image_rgb[nc]
            
        dims = image.shape
        image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        if (side=='r' or side=='R'): # flip image and ROI coordinates.
            image_2d_scaled =cv2.flip(image_2d_scaled, 1)
            
        dims = image_2d_scaled.shape
        # print("dimension",dims)  
        
        if (nc == 0) : # same bbox for all slices
            bbox,mask = get_normal_BBox(image_2d_scaled)
        image_crop = image_2d_scaled[bbox.y1:bbox.y2,bbox.x1:bbox.x2]
        # print(image_crop.shape)
        if (nc==0) and (comp>1): 
                out_rgb = np.zeros([image_crop.shape[0],image_crop.shape[1],comp])
        if (comp==1): out_rgb = image_crop
        else: out_rgb[:,:,nc] = image_crop
    
    outputs = predictor(out_rgb)
    out = outputs["instances"].to("cpu")
    if (len(out.scores)>0):
        print(out.pred_boxes)
        print(out.scores)    
    
        for score, pred_box in zip(out.scores, out.pred_boxes):
#             print(pred_box, score)
            # adapt bbox_ROI to original  image (add bbox.x1 and bbox.y1)
            pred_box[0] += bbox.x1
            pred_box[2] += bbox.x1
            pred_box[1] += bbox.y1
            pred_box[3] += bbox.y1
            if (side=='r' or side=='R'): 
                # flip bounding box back in case of R image         
                aux = float(pred_box[2])
                pred_box[2] = dims[1]-pred_box[0] 
                pred_box[0] = dims[1]-aux                 
#     print(out.pred_boxes)
#     print(out.scores)      
    return out.pred_boxes, out.scores
    
def NMS_volume (pred_boxes_vol,pred_scores_vol):
    length = len(pred_boxes_vol)

    final_boxes_vol = []
    final_scores_vol = []
    final_slices_vol = []
    depth = int(length/depth_d) # otherwise other lesions are still doble detected.
    print('Depth ',depth)
    # now get the maximum along the each slice depth
    for i in range(0,length):
        if (len(pred_boxes_vol[i])>0):
            s0 = max(i-depth,0)
            s1 = min(i+depth,length)
            # print("i s",i, s0,s1)
            # checking slice i against all other slices (doing a cat)           
            all_boxes = detectron2.structures.Boxes.cat(pred_boxes_vol[s0:s1])
            all_scores =  torch.squeeze(torch.cat([b for b in pred_scores_vol[s0:s1]], dim=0))
            # there is more than one annotation.
            if (all_scores.dim()>0):
                if (all_scores.shape[0] >0): # this is to solve bug found that some tensors were [0,1]
                    # check if there is overlap
                    iou_matrix = torch.squeeze(detectron2.structures.pairwise_iou(pred_boxes_vol[i], all_boxes))
    #                 print("iou: ",iou_matrix)
                    j = 0
                    is_all_zero = int(iou_matrix.sum())==0
                    for box, score in zip(pred_boxes_vol[i],pred_scores_vol[i]):
                        if (iou_matrix.dim()>1):
                            idx_s = ((iou_matrix[j,:] > min_iou).nonzero())
                            r_iou = iou_matrix[j,idx_s] 
                        else: 
                            idx_s = ((iou_matrix > min_iou).nonzero())
                            r_iou = iou_matrix[idx_s]                                                
                        # condition 1: my score is larger or equal than all other overlapping boxes (idx_s)
                        # print(idx_s, all_scores, score, iou_matrix)
                        if (score >= torch.max(all_scores[idx_s])):
                            if (score >= min_score):
    #                             print("adding slice ",i+start_pred, score, box)
                                final_boxes_vol.append(box)
                                final_scores_vol.append(score)
                                final_slices_vol.append(i+start_pred)
                        j +=1
            else: # there is only one annotatio only check single threshold.
                for box, score in zip(pred_boxes_vol[i],pred_scores_vol[i]):
                    if (score >= min_score_s):
#                         print("adding Single slice ",i+start_pred, score, box)
                        final_boxes_vol.append(box)
                        final_scores_vol.append(score)
                        final_slices_vol.append(i+start_pred)
    print('Final Boxes')
    print(final_boxes_vol)
    print(final_scores_vol)
    print(final_slices_vol)
    # input("Press Enter to continue...")
    
    return final_boxes_vol ,final_scores_vol, final_slices_vol



def write_csv (final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, total_slices):
     
   # write CSV with coordinates 
#     depth = int(total_slices/8) Changed to 0. 
# depth for the evaluation of the challenge is not important, only used to get the central slice (and we have that)
   depth = 0
   with open(output_csv, 'a+', newline='') as file:
        writer = csv.writer(file)
        listScores = sorted(zip(final_boxes_vol, final_scores_vol,final_slices_vol), key = lambda t: t[1],reverse=True)
        n_boxes = 0
        for box,score,slice_num in listScores:
            if (n_boxes < max_n_boxes):
                print(client, episode, view, int(box[0]),int(box[2]-box[0]), int(box[1]), int(box[3]-box[1]),
                        max(0,slice_num-depth),  min(total_slices-1, slice_num+depth)-max(0,slice_num-depth),float(score))
                writer.writerow([client, episode, view, int(box[0]),int(box[2]-box[0]), int(box[1]), int(box[3]-box[1]),
                        max(0,slice_num-depth),  min(total_slices-1, slice_num+depth)-max(0,slice_num-depth),float(score)])
                n_boxes +=1
       
# PatientID: string - patient identifier
# StudyUID: string - study identifier
# View: string - view name, one of: RLL, LCC, RMLO, LMLO
# X: integer - X coordinate (on the horizontal axis) of the left edge of the predicted bounding box in 0-based indexing (for the left-most column of the image x=0)
# Width: integer - predicted bounding box width (along the horizontal axis)
# Y: integer - Y coordinate (on the vertical axis) of the top edge of the predicted bounding box in 0-based indexing (for the top-most column of the image y=0)
# Height: integer - predicted bounding box height (along the vertical axis)
# Z: integer - the first bounding box slice number in 0-based indexing (for the first slice of the image z=0)
# Depth: integer - predicted bounding box slice span (size along the depth axis)
# Score: float - predicted bounding box confidence score indicating the confidence level that the detection represents an actual lesion. This score can have an arbitrary scale, but has to be unified across all the cases within a single submission (e.g. 0.0 – 1.0)
# Example:
# PatientID,StudyUID,View,X,Width,Y,Height,Z,Depth,Score ID1,UID1,RLL,X(int),Width(int),Y(int),Height(int),Z(int),Depth(int),Score(float) ID2,UID2,LCC,X(int),Width(int),Y(int),Height(int),Z(int),Depth(int),Score(float) ID3,UID3,RMLO,X(int),Width(int),Y(int),Height(int),Z(int),Depth(int),Score(float) ID4,UID4,LMLO,X(int),Width(int),Y(int),Height(int),Z(int),Depth(int),Score(float)                      





### main program

if __name__ == "__main__":

    # Parse input options
    parser = argparse.ArgumentParser(
        description="predict given ")
    parser.add_argument('--input_csv',
                        action='store',
                        required=True,
                        help='input cvs file-paths (mandatory)')
    parser.add_argument('--base_path',
                        action='store',
                        required=True,
                        help='base path for images (mandatory)')
    parser.add_argument('--model',
                        action='store',
                        required=True,
                        help='model to be used')                        
    parser.add_argument('--config',
                        action='store',
                        required=True,
                        help='config model file')                        
    parser.add_argument('--output_csv',
                        action = 'store',
                        required=True,
                        help='output csv (mandatory) without extension')
    parser.add_argument('--labels_csv',
                        action = 'store',
                        required=False,
                        help='labels csv (optional) to predict only biopsy lesions')
    
    parser.add_argument('--gray',
                        action = 'store',
                        required=False,
                        help='use gray level (not different channels)')

    
    opt = parser.parse_args()
    
    inference_file = opt.input_csv
    model_file = opt.model 
    config_file = opt.config
    output_file = opt.output_csv
    base_path = opt.base_path
    filter_labels = False
    use_gray = False

    print ("Opts: ", inference_file, model_file, config_file, output_file, base_path)

    # if labels are provided, only benign and malign cases are predicted (runs faster)
    if opt.labels_csv is not None:
        label_file = opt.labels_csv
        print (label_file)
        filter_labels = True
    
    if opt.gray is not None:
        use_gray = True
        print("using GRAY")
    else: print ("using RGB")

    # parameters for N
    start_pred = 4 # 5start_pred:end to look for prediction not implemented
    end_pred = 4 # 5 end_pred to look for prediction not implemented (all slices are used)
    min_score = 0.01 # minimum score to keep prediction
    min_iou = 0.5 #0.5 #0.75 # minimum iou to look for overlappin boxes.
    max_n_boxes = 15 # how many detections per volume. 
    min_score_s = 0.0 # minimum score for single predictions 
    depth_d = 4 #1 # division for depth 1= 100%
    debug = True
    use_detr = 'detr' in model_file
    if (use_detr):
        print(" using DETR")
    else: print("using FASTER RCNN")

    # model_r2 = True
    # local_train = False
    # test_challenge = False


    output_csv = output_file + str(depth_d)+'slices'+str(start_pred)+'s_sc'+str(min_score)+'_iou'+ str(min_iou)+'.csv'
        
    debug_dir = './debug'
        
    df = pd.read_csv(inference_file)
    if (filter_labels): 
        df_labels = pd.read_csv(label_file)
        df_labels = df_labels.loc[(df_labels["Benign"] == 1) | (df_labels["Cancer"] == 1)]
        df_labels = df_labels.reset_index().set_index(["StudyUID", "View"]).sort_index()
        # print(df_labels)

        df = df.reset_index().set_index(["StudyUID", "View"]).sort_index()
        df = df[df.index.isin(df_labels.index)]
        df = df.reset_index()
        print(df)

    # load trained network 
    cfg = get_cfg()

    if (use_detr):
        add_detr_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 10
        cfg.MODEL.DETR.NUM_CLASSES = 2 
    else: 
        cfg.merge_from_file(config_file)
        #Use the final weights generated after successful training for inference  
        cfg.MODEL.WEIGHTS = model_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score #0.8  # set the testing threshold for this model
    #Pass the validation dataset
    cfg.DATASETS.TEST = ("DBT_train", )
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("DBT_train")

    if (debug): print(df)

    with open(output_csv, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PatientID','StudyUID','View','X','Width','Y','Height','Z','Depth','Score'])

    skip_it = False

    for index,view_series in df.iterrows():
        print(index, view_series)
        if (debug): print(view_series["PatientID"],view_series["View"])
        view = view_series["View"]
        # if you have image data saved in classic path folder structure, read the file path from "classic_path"
        # image_path = os.path.join("/data", view_series["classic_path"])
        
        image_path = os.path.join(base_path, view_series["descriptive_path"])
        print(image_path)
        
        
        client = view_series["PatientID"]
        episode = view_series["StudyUID"]
        
    #     # debug
    #     image_path = '/home/robert/data/DBT/Breast-Cancer-Screening-DBT/DBT-P00259/01-01-2000-DBT-S00746-MAMMO screening digital bilateral-91492/7897.000000-60823/1-1.dcm'
    #     client = 'DBT-P01282'
    #     episode = 'DBT-S01508'
    # #     view = 'lcc2'
        # if (client == 'DBT-P03978'): skip_it = False
        # else:
        #     print("skipping", client) 
        #     skip_it = True

        side = view[0]
        pred_boxes_vol = []
        pred_scores_vol =[]

        if (not skip_it):
            image = dcmread_image(fp=image_path, view=view)   
            dims = image.shape
            
            
            for slice_index in range(dims[0]):
                if (slice_index >= start_pred and slice_index < dims[0]-end_pred ) :
        #             print ("Slice ",slice_index)
                    # copy grey-scale image
                    if (use_gray): sl = 0
                    else: sl = 2
                    
                    image_rgb = []
                    image_rgb.append(image[slice_index-sl])
                    image_rgb.append(image[slice_index])
                    image_rgb.append(image[slice_index+sl])
                    image_rgb = np.array(image_rgb)
                    # image_2d = image[slice_index,:,:]
                    pred_boxes, pred_scores = predict_slice(view,client, episode,image_rgb,side)
                    # # if (use_detr): # remove lower scores
                    # #     keep = pred_scores > min_score
                    # #     pred_boxes = pred_boxes[keep]
                    #     print(keep, pred_boxes)                        
                    pred_boxes_vol.insert(slice_index, pred_boxes)                   
                    pred_scores_vol.insert(slice_index,pred_scores)
        #     print(pred_boxes_vol)
            final_boxes_vol, final_scores_vol, final_slices_vol = NMS_volume (pred_boxes_vol,pred_scores_vol)
            # sort them and keep the N best scores
            
            listScores = sorted(zip(final_boxes_vol, final_scores_vol,final_slices_vol), key = lambda t: t[1],reverse=True)
            n_boxes = 0
            for box,score,slice_i in listScores:
                if (n_boxes < max_n_boxes):
                    image_2d = image[slice_i,:,:]
                    if (debug):
                        print (box, score,slice_i)
                        image_crop = image_2d[int(box[1]): int(box[3]), int(box[0]):int(box[2])]
                        cv2.imwrite(debug_dir+"/rois/"+client+"_"+episode+"_"+view+"_"+str(slice_i)+"_"+str(score)+".png", image_crop)
                        cv2.rectangle(image_2d,(int(box[0]),int(box[1])),(int(box[2]), int(box[3])),(255,255,255),5)
                        cv2.imwrite(debug_dir+"/"+client+"_"+episode+"_"+view+"_"+str(slice_i)+"_"+str(score)+".png", image_2d)
                    n_boxes +=1
            write_csv (final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, dims[0])
            #input("Press Enter to continue...")
