In this part of methods we implemented following pipeline on our single-stained tissue microarray:
NaroNet - discovery of tumor microenvironment elements; 
authors paper: [*paper*](https://arxiv.org/abs/2103.05385). 

© [Daniel Jiménez Sánchez - CIMA University of Navarra](https://cima.cun.es/en/research/research-programs/solid-tumors-program/research-group-preclinical-models-preclinical-tools-analysis) - This code is made available under the GNU GPLv3 License and is available for non-commercial academic purposes. 


## Requirements and installation
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 4 on GPU server, and Nvidia P100, K80 GPUs on Google Cloud)

To install NaroNet it is recommended creating a new [*anaconda*](https://www.anaconda.com/distribution/) environment with TensorFlow (either TensorFlow 1 or 2) and Pytorch (v.1.4.0 or newer). For GPU support, install the versions of CUDA that are compatible with TensorFlow's and Pytorch's versions.

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
* Channels.txt contains per row the name of each marker/channel present in the image.
* Image_Labels.xlsx contains the image names and their corresponding image-level labels.
## Citation
The paper of the NaroNet creators:
```diff
@article{jimenez2021naronet,
  title={NaroNet: Discovery of tumor microenvironment elements from highly multiplexed images},
  author={Jiménez-Sánchez, Daniel and Ariz, Mikel and Chang, Hang and Matias-Guiu, Xavier and de Andrea, Carlos E and Ortiz-de-Solórzano, Carlos},
  journal={arXiv preprint arXiv:2103.05385},
  year={2021}
}
```


