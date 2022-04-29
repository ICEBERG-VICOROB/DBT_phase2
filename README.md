# VICOROB DBT Challenge participation Phase 2

Participation on the DBT Challenge https://spie-aapm-nci-dair.westus2.cloudapp.azure.com/competitions/6 from the VICOROB research group (University of Girona, Spain). 

Contact: Robert Marti, robert.marti@udg.edu


Using additional functions from duke-dbt-data 

https://github.com/MaciejMazurowski/duke-dbt-data


## Files
- pre_process_rm.ipynb
Notebook for pre-processing of DBT images to generate 2D slices and bounding boxes to be trained in detectron

- image_utils.py
Utils related to normalisation and saving images.

- DBT_detectron_omidb.ipynb
Notebook for implementation of detectron training with the OMIDB and DBT datasets

- fp_reduction_train.ipynb
Notebook for training the FP reduction step. 

- inference_DBT_NMS2.py
Inference of new images (validation provided in the web page)

- inference_DBT_NMS_fpred.py
Inference of new images using FP reduction step. 

- run_inf_fold2_model_fold1_30k_gray.sh 
Example for calling inference.

- output: config files and link to the model file
