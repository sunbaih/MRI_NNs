import os
import torch 
import SimpleITK as sitk
import nibabel as nib 
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from skimage import measure
from stl import mesh
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def read_nifti_data(folder_path, output_shape = [10, 10, 10]):
    """
    Parameters
    ----------
    folder_path : path to folder containing NIFTI data 

    Returns
    -------
    numpy array of NIfTI data, list of corresponding patient IDs 

    """
    nifti_arrays = []
    patient_ids = []
    
    os.chdir(folder_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('nii') or filename.endswith('.nii.gz'): 
            file_path = os.path.join(folder_path, filename)
            nifti_image = sitk.ReadImage(file_path)
  
            if nifti_image.GetSize() == output_shape:
                nifti_image = nifti_image 
            else: 
                nifti_image = reshape_nifti(nifti_image, output_shape)
        
            nifti_array = sitk.GetArrayFromImage(nifti_image)
            nifti_arrays.append(nifti_array)
            patient_id = filename.split(".")[0]
            patient_ids.append(patient_id)

    return nifti_arrays, patient_ids


def reshape_nifti(nifti_img, output_shape = [155, 240, 240]):
    """
    Input 2D images are not consistent in terms of length, width, and height of stacked images. 
    Resize images for consistency, save to folder. 

    """
    resampled_img = sitk.Resample(nifti_img, output_shape, sitk.Transform(), sitk.sitkLinear, nifti_img.GetOrigin(), nifti_img.GetSpacing(), nifti_img.GetDirection(), 0.0, nifti_img.GetPixelID())
    
    return resampled_img 

            
            
class TumorData(Dataset):
    def __init__(self, data_path_2d, data_path_3d):
        
        self.patient_ids, self.data_2d, self.data_3d, _ = self.loaddata(data_path_2d, data_path_3d)

        self.length = len(self.patient_ids)
        self._num_samples = self.length
            
    def loaddata(self, data_path_2d, data_path_3d):
        """
        Parameters
        ----------
        data_path_2d : path to folder containing 2D MRI images 
        data_path_3d : path to folder containing 3D MRI images 

        Returns
        -------
        patient_ids : list of patient IDs with both 2D and 3D data
        data_2d : list of arrays containing 2D tumor image information corresponding to patient IDs
        data_3d : list of arrays containing 2D tumor image information corresponding to patient IDs
        test_ids : second list of patient IDs used to check correspondence between data_2d and data_3d 
        """
        
        patient_ids = []
        data_2d = []
        data_3d = []
        test_ids = []
        
        nifti_arrays_2d, patient_ids_2d = read_nifti_data(data_path_2d)
        nifti_arrays_3d, patient_ids_3d = read_nifti_data(data_path_3d)  
        
        for i in range(len(patient_ids_2d)):
            
            patient_id = patient_ids_2d[i]
            
            try: 
                index = patient_ids_3d.index(patient_id)
     
                patient_ids.append(patient_id)
                data_2d.append(nifti_arrays_2d[i])
                data_3d.append(nifti_arrays_3d[index])
                
                test_ids.append(patient_ids_3d[index])
                
            except: 
                pass
            
        print(len(data_2d))
        data_2d = torch.tensor(data_2d)
        data_3d = torch.tensor(data_3d)
    
        return patient_ids, data_2d, data_3d, test_ids
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]
        data_2d = self.data_2d[idx]
        data_3d = self.data_3d[idx]
            
        return patient_id, data_2d, data_3d 
            
        

def get_dataloader(data_path_X = "/Users/baihesun/cancer_data/TCIA_manual_segmentations", 
                   data_path_Y = "/Users/baihesun/cancer_data/BRATS_TCGA_GBM_all_niftis", 
                   batch_size = 1, test_size=0.2):

    dataset = TumorData(data_path_X, data_path_Y)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=100)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

"""
    
if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(batch_size = 1)
    """

            
            
            
            



