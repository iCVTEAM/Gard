#### Introduction

This code provides an initial version for the implementation of the CVPR2021 paper "Graph-based High-Order Relation Discovery for Fine-grained Recognition". The projects are still under construction.



For our projects, please refer to http://cvteam.net/projects/2021/Gard/Gard.html



#### Using API

Please refer to  /doc/build/html/index.html, including function definition.

or finding at: http://cvteam.net/projects/2021/Gard/html/



#### How to run

##### For Training:

1. Download the benchmark dataset and unzip them in your customized path.

   CUB-200-2011 http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

   FGVC-Aircraft https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

   Stanford-Cars http://ai.stanford.edu/~jkrause/cars/car_dataset.html

   NABirds https://dl.allaboutbirds.org/nabirds

2. Build the Train/validation partition by yourself or download the files from [here](http://cvteam.net/projects/2021/Gard/dataset_split.zip). 

3. Modify the configuration files in /config/config.py and /config/default.py

4. Dataset config

   4-1 Set the dataset path in /config/config.py  if using CUB dataset

   4-2 Set  the dataset path in /datasets/UnifiedLoader.py if using other datasets

   4-3 Add functions and make your own datasets in  /datasets/UnifiedLoader.py

5. Modify Line 70~76, uncomment used dataset  and comment out other datasets

6. Run train.py for training.

##### For Testing:

1. repeat or confirm the operations in **training** 1~5

2. Modify the class_num in /config/config.py

3. Download Testing weights. (Only test demo, will be updated soon)

   For example, CUB-200-2011: https://drive.google.com/file/d/14_sssTjbF-0okVLKeqndgXsOBMR-sJtY/view?usp=sharing

4. Modify the Test weights dir in /config/config.py

5. Run test.py 



#### Prerequisites

PyTorch, tqdm, torchvsion, Pillow, cv2

Running on Two GPUs to achieve higher performance.

If on other GPU settings, the hyper params should be modified to achieve similar results.



#### Known issues & Bugs

The performance would be fluctuated by 0.1~0.5% in different GPUs and PyTorch platforms. Pytorch versions higher than 1.3.1 are tested. Some results are much higher than the reported results (e.g. some incidental results in CUB can reach 90.4%), but others are lower.



#### To do

1. The project is still ongoing, finding suitable platforms and GPU devices for complete stable results.

2. The project is re-constructed for better understanding, we release this version for a quick preview of our paper.

   

#### Citations:

Please remember to cite us if u find this useful.

@inproceedings{zhao2021graph,
  title={Graph-based High-Order Relation Discovery for Fine-grained Recognition},
  author={Zhao, Yifan and Yan, Ke and Huang, Feiyue and Jia, Li},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
}



#### License

The code of the paper is freely available for non-commercial purposes. Permission is granted to use the code given that you agree:

1. That the code comes "AS IS", without express or implied warranty. The authors of the code do not accept any responsibility for errors or omissions.

2. That you include necessary references to the paper [1] in any work that makes use of the code. 

3. That you may not use the code or any derivative work for commercial purposes as, for example, licensing or selling the code, or using the code with a purpose to procure a commercial gain.

4. That you do not distribute this code or modified versions. 

5. That all rights not expressly granted to you are reserved by the authors of the code.



