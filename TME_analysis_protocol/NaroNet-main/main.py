from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
from NaroNet.architecture_search.architecture_search import architecture_search
from NaroNet.NaroNet import run_NaroNet
from NaroNet.NaroNet_dataset import get_BioInsights
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


##Monika
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
##


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')    

    # Preprocess Images
    preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

    # Patch Contrastive Learning
    patch_contrastive_learning(path,params)    

    # Architecture Search
    # params = architecture_search(path,params,possible_params)

    # run_NaroNet(path,params)
    
    # BioInsights
    get_BioInsights(path,params)

if __name__ == "__main__":
    
    path = '/home/hippo/Monika/NaroNet_Nencki/DATASET_DATA_DIR/'            
    main(path)
 