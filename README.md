# cv_final_hw
cv final project (colorization) of 2021-2022 winter term

***
### Introduction
The project includes two CNN-based end-to-end colorization algorithms, listed in reference. 
Model from ECCV16 was run on Win10, while the other on Ubuntu and PBS.

### Dataset
* **Pretrain**: Both models were pretrained using [ImageNet](https://www.image-net.org/).
* **Valid**: Both models were tested on 1,000 pictures selected from [Place365](http://places2.csail.mit.edu/).
### Environment
* conda 4.11
* python 3.9
* torch 1.11
* lua 5.4
### Folder
* imgs/: input images
* img_out/: output results
* model/: pretrained siggraph16 model
* colorizers/: ECCV16 model

### How to run
`ECCV16: python main.py` <br>
`siggraph16: ./model/download_model_imagenet.sh and 
             qsub run_siggraph16.sh` 
### Reference
1. R. Zhang, P. Isola, and A. A. Efros, Colorful image colorization," European Conference on Computer
Vision, 2016. <br>
2. S. Iizuka, E. Simo-Serra, and H. Ishikawa, Let there be Color!: Joint End-to-end Learning of Global
and Local Image Priors for Automatic Image Colorization with Simultaneous Classification," ACM
Transactions on Graphics (Proc. of SIGGRAPH 2016), vol. 35, no. 4, 2016.