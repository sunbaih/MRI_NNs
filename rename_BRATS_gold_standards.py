#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:25:23 2023

@author: baihesun
"""

import os
import shutil 

folderpath = "/Users/baihesun/cancer_data/TCGA-GBM_all_niftis/"
new_folder_path = "/Users/baihesun/cancer_data/BRATS_TCGA_GBM_all_niftis/"

os.chdir(folderpath)

for filename in os.listdir():
    patient_id = filename.split("_")[0]
    new_name = patient_id + ".nii.gz"
    
    old_file_path = os.path.join(folderpath, filename)
    new_file_path = os.path.join(new_folder_path, new_name)
    shutil.copyfile(old_file_path, new_file_path)
    