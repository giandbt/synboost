# Semantic Segmentation Uncertainty Estimation
This repository tackles uncertainty estimation for semantic segmentation using visual dissimilarity. The project is a 
continuation of the work presented by [1] and it relates to my master thesis in the Autonomous Systems Lab (ASL) 
at ETH Zurich. 

### Installation

In order to set-up the project, please follow these steps:
1) Run  `git clone https://github.com/giandbt/driving_uncertainty.git`. 
2) Download pre-trained model for CC-FPSE [2] 
(https://drive.google.com/uc?id=1m4JMtKLDfcXCW1HXHKz-fP6y3_SAaUqX&export=download) and save it under `/models/image-synthesis/`
3) Download pre-trained model for DeepLabV3+ with WiderResNet38 [3] (https://drive.google.com/file/d/1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl/view). 
and save it `./models/image-segmentation/`. 
4) We need to install Apex manually. Instructions can be found here: https://github.com/NVIDIA/apex
5) Install all the neccesary python modules with `pip install -r requirements.txt`

### Datasets 
Todo 

### Training 
Todo

### Evaluation
The repository already includes some sample images to test the pipeline, which are found under `./sample_images/`. 
In order to run inference in these images, run the following command: `python demo_folder.py --use_vae`

In case custom images want to be tested change the `--demo-folder` flag. Information about all the other flags can be 
found running `demo_folder.py -h`.

### ONNX Conversion 

In order to convert all three models into `.onnx`, it is neccesary to update the `symbolic_opset11.py` file from the
original `torch` module installation. The reason for this is that `torch==1.4.0` does not have compatibility for `im2col`
which is neccesary for the synthesis model. 

Simply copy the `symbolic_opset11.py` from this repository and replace the one from the torch module inside your project environment. 
The file is located `/Path/To/Enviroment/lib/python3.7/site-packages/torch/onnx`

### Notes 

- The image segmentaion folder is heavily based on [3], specifically commit `b4fc685`. Additionally, 
the image synthesis folder is based on [2]. specifically commit `0486b08`. For light weight version of the segmentation
model, we used the code from ``, and also Pix2PixHD commit ``


## References
[1] This is not what I imagined: Error Detection for Semantic Segmentation
through Visual Dissimilarity. David Haldimann, Hermann Blum, Roland Siegwart, Cesar Cadena.

[2] Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis.
Xihui Liu, Guojun Yin, Jing Shao, Xiaogang Wang and Hongsheng Li.

[3] Improving Semantic Segmentation via Video Propagation and Label Relaxation
Yi Zhu1, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro.
