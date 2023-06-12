# NaroNet: discovery of tumor microenvironment elements from highly multiplexed images.
***Summary:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to classify subjects into predefined types. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding subject-level labels. See our [*paper*](https://arxiv.org/abs/2103.05385) for further description of NaroNet.  

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview_big.png' />

© [Daniel Jiménez Sánchez - CIMA University of Navarra](https://cima.cun.es/en/research/research-programs/solid-tumors-program/research-group-preclinical-models-preclinical-tools-analysis) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes. 


## Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

To install NaroNet we recommend creating a new [*anaconda*](https://www.anaconda.com/distribution/) environment with TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer). For GPU support, install the versions of CUDA that are compatible with TensorFlow's and Pytorch's versions.

Once inside the created environment, install pytorch-geometric where ${CUDA} and ${TORCH} should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111, cu113) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0). Run the following commands in your console replacing ${CUDA} and ${TORCH}:
```sh
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

Install NaroNet downloading this repository or through pip:
```sh
pip install NaroNet
```

## Preparing datasets
Create the target folder (e.g., 'DATASET_DATA_DIR') with your image and subject-level information using the following folder structure:

```bash
DATASET_DATA_DIR/
    └──Raw_Data/
        ├── Images/
                ├── image_1.tiff
                ├── image_2.tiff
                └── ...
        └── Experiment_Information/
                ├── Channels.txt                
                ├── Image_Labels.xlsx
		└── Patient_to_Image.xlsx (Optional)
		
```
In the 'Raw_Data/Images' folder we expect multiplex image data consisting of multi-page '.tiff' files with one channel/marker per page.
In the 'Raw_Data/Experiment_Information' two files are expected:
* Channels.txt contains per row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Channels.txt) or example below:
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image names and their corresponding image-level labels. In column 'Image_Names' image names are specified. The next columns (e.g., 'Control vs. Treatment', 'Survival', etc.) specify image-level information, where 'None' means that the image is excluded from the experiment. In case more than one image is available per subject and you want to make it sure that images from the same subject don't go to different train/val/test splits, it is possible to add one column named "Subject_Names" specifying, for each image, the subject to whom it corresponds. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Image_Labels.xlsx) or example below:

| Image_Names | Control vs. Treatment | Survival | 
| :-- | :-:| :-: |
| image_1.tiff | Control  | Poor |
| image_2.tiff | None | High |
| image_3.tiff | Treatment | High |
| ... | ... | ... |


## Citation
Please cite this paper in case our method or parts of it were helpful in your work.
```diff
@article{jimenez2021naronet,
  title={NaroNet: Discovery of tumor microenvironment elements from highly multiplexed images},
  author={Jiménez-Sánchez, Daniel and Ariz, Mikel and Chang, Hang and Matias-Guiu, Xavier and de Andrea, Carlos E and Ortiz-de-Solórzano, Carlos},
  journal={arXiv preprint arXiv:2103.05385},
  year={2021}
}
```


