# Pixel-wise Anomaly Detection for Complex Driving Scenes
This repository is the paper implementation for Pixel-wise Anomaly Detection for Complex Driving Scenes (`HYPER-LINK PAPER ONCE IN ARXIV`). 

![Alt text](display_images/methodology.png?raw=true "Methodology")

### Installation

In order to set-up the project, please follow these steps:
1) Run  `git clone https://github.com/giandbt/driving_uncertainty.git`. 
2) Download pre-trained models using `wget https://dissimilarity.s3.eu-central-1.amazonaws.com/models.tar`. 
De-compress file and save inside the repository (`tar -xvf ./models.tar`)
3) We need to install Apex (https://github.com/NVIDIA/apex) running the following:
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir ./
    ```
4) Install all the neccesary python modules with `pip install -r requirements_demo.txt`

### Training 
The anomaly pipeline uses pre-trained models for segmentation and image re-synthesis. 
You can find this pre-trained models using `wget https://dissimilarity.s3.eu-central-1.amazonaws.com/models.tar`. 
Additionally, you can refer to the original repositories. 

In order to trained the dissimilarity network, we have to do the following:
1) `cd image_dissimilarity`
2) Modify the necessary parameters in the configuration file `image_dissimilarity/configs/train/default_configuration.yaml`. 
More importanly, modify the folder paths for each dataset to your local path. In order to get the required data for training, please 
refere to the Dataset section. 
3) Run `train.py --config configs/train/default_configuration.yaml`

### Evaluation
The repository already includes some sample images to test the pipeline, which are found under `./sample_images/`. 
In order to run inference in these images, run the following command: `python demo.py`

In case custom images want to be tested change the `--demo_folder` flag. Information about all the other flags can be 
found running `demo.py -h`.

![Alt text](display_images/three_anomaly_scenarios.png?raw=true "Methodology")

### Datasets 
The repository uses the Cityscapes Dataset [4] as the basis of the training data for the dissimilarity model. 
To download the dataset please register and follow the instructions here: https://www.cityscapes-dataset.com/downloads/.

Then, we need to pre-process the images in order to get the predicted entropy, distance, perceptual difference, synthesis, and semantic maps. 
The neccesary files to do all the operations can be found under `data_preparation` folder. In future releases, we will make one script to do all 
the pre-process automatically. For the time being, you can download the processed datasets used in the paper here: http://robotics.ethz.ch/~asl-datasets/Dissimilarity/ .

### Framework Light Version 
The original paper discussed the implementation of a lighter version in order to demostrate the generalization ability of the network to different
synthesis and segmentation networks (even with lower performance).

In the repository, we include the code and pre-trained model used for this lighter version. However, compatiblity with `demo.py` is still not supported. 

### Google Colab Demo Notebook
A demo of the anomaly detection pipeline can be found here: https://colab.research.google.com/drive/1HQheunEWYHvOJhQQiWbQ9oHXCNi9Frfl?usp=sharing#scrollTo=gC-ViJmm23eM

### ONNX Conversion 

In order to convert all three models into `.onnx`, it is neccesary to update the `symbolic_opset11.py` file from the
original `torch` module installation. The reason for this is that `torch==1.4.0` does not have compatibility for `im2col`
which is neccesary for the synthesis model. 

Simply copy the `symbolic_opset11.py` from this repository and replace the one from the torch module inside your project environment. 
The file is located `/Path/To/Enviroment/lib/python3.7/site-packages/torch/onnx`

### Notes 

- The image segmentaion folder is heavily based on [1], specifically commit `b4fc685`. Additionally, 
the image synthesis folder is based on [2]. specifically commit `0486b08`. For light weight version of the segmentation
model, we used the code from [3] (commit `682e0e6`) as our segmentation model, and [4] as our synthesis model (commit `5a2c872`)
- The branch `fishyscapes_package` includes the code as a package specifically made for Fishyscapes submission.
In ther to get the class for the detector simply `from test_fishy_torch import AnomalyDetector`.


## References
[1] Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis.
Xihui Liu, Guojun Yin, Jing Shao, Xiaogang Wang and Hongsheng Li.

[2] Improving Semantic Segmentation via Video Propagation and Label Relaxation
Yi Zhu1, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro.

[3] https://github.com/lxtGH/Fast_Seg

[4] High-resolution image synthesis and semantic manipulation with conditional gans.
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. 

[5] The cityscapes dataset for semantic urban scene understanding. 
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele
