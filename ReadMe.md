### Introduction
![Copyright](https://img.shields.io/badge/Copyright-CVTEAM-red)

This code provides an initial version for the implementation of the CVPR2021 paper "Graph-based High-Order Relation Discovery for Fine-grained Recognition" [Paper Link](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Graph-Based_High-Order_Relation_Discovery_for_Fine-Grained_Recognition_CVPR_2021_paper.pdf). The projects are still under construction.

For our projects, please refer to http://cvteam.net/projects/2021/Gard/Gard.html



### Using API

Please refer to  /doc/build/html/index.html, including function definition.

or finding at: http://cvteam.net/projects/2021/Gard/html/



### How to run

#### For Training:

1. Download the benchmark dataset and unzip them in your customized path.

   CUB-200-2011 http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

   FGVC-Aircraft https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

   Stanford-Cars http://ai.stanford.edu/~jkrause/cars/car_dataset.html

   NABirds https://dl.allaboutbirds.org/nabirds

2. Build the Train/validation partition by yourself or download the files from [here](http://cvteam.net/projects/2021/Gard/dataset_split.zip). 

3. Modify the configuration files in /config/config.py and /config/default.py

4. Dataset config

   4-1 Set the dataset path in /config/config.py  if using CUB dataset

   4-2 Set the dataset path in /datasets/UnifiedLoader.py if using other datasets

   4-3 Add functions and make your own datasets in  /datasets/UnifiedLoader.py

5. Modify Line 70~76, uncomment used dataset  and comment out other datasets

6. Run train.py for training.

#### For Quick Testing:

1. repeat or confirm the operations in **training** 1~5

2. Modify the class_num in /config/config.py

3. Download Testing weights. 

   For example, CUB-200-2011 [Model Link](https://drive.google.com/file/d/14_sssTjbF-0okVLKeqndgXsOBMR-sJtY/view?usp=sharing)

4. Modify the Test weights dir in /config/config.py

5. Run test.py 



### Prerequisites

PyTorch, tqdm, torchvsion, Pillow, cv2

Running on Two GPUs to achieve the reported performance.

If on other GPU settings, the hyper params should be modified to achieve similar results.



### Known issues

The performance would be fluctuated by 0.1~0.5% in different GPUs and PyTorch platforms. Pytorch versions higher than 1.3.1 are tested. Some results are much higher than the reported results (e.g. results on CUB dataset can reach 90.4% vs reported 89.6% when changing different platforms).



### To do

1. The project is still ongoing, finding suitable platforms and GPU devices for complete stable results.

2. The project is re-constructed for better understanding, we release this version for a quick preview of our paper.

   

### Citations:

Please remember to cite us if u find this useful.
```
@inproceedings{zhao2021graph,
  title={Graph-based High-Order Relation Discovery for Fine-grained Recognition},
  author={Zhao, Yifan and Yan, Ke and Huang, Feiyue and Jia, Li},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
}
```



