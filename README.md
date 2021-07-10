# BacteriaImageProcess
This is for our project: to segment bacteria images and identify what bacteria species are in this image. In essence, this is a deep learning application on bacteria data. We implement Convolutional RBM in matlab to fulfill our tasks. 

The cdbn-github is code for unsupervised feature learning using Convolutional RBM, and we use GMM/BMM to do initialization which can
shorten the training process. And after feature extraction, we exploit liblinear toolbox to do supervised classification.

The first part of our project is in cdbn folder, in which I uploaed all the codes, including codes for CRBM written by me. GMM is written to initialize the first layer, and BMM is used to do initialization for 2nd layer. (Note, the initialization is very important, for more details, please refer to Sohn etal.'s paper: "Efficient learning of sparse, distributed, convolutional feature representations for object recognition". Aslo, to train Convolutional Deep Belief Network, you have to do layerwise pretraining, which means, train the first layer first, after training, frozen the first layer's parameters, and then train the 2nd layer....)

In the second part, we try to identify what bacteria speices are in this image, so we manually label the foreground patches, and label them as 17 classes, as some bordering patches are involving species and background, we also take them into considerations, thus, there are 18 classes. We employ CNN to learn a classification model, the result outperforms the SIFT+SVM and HoG+SVM, however, we still analyze the failure cases, they mainly come from two parts: 1. the two species looks too similar, even human are hard to distinguis them; 2. the bacteria interaction area plays a major role, which makes the two species hard to distinguish. As the first failure, we cannot do anything as far, while we design a new experiment for the new one, we manuualy label the interaction areas as another class, then we do a new classification with neural networks.

Tools for 2nd part is from caffe, and the prototxt files are provided, also, the models are provided.

If you are interested in this project, you can refer to http://dl.acm.org/citation.cfm?id=2808751 for more details. And if you want to our code or model, please cite our paper:

```
@inproceedings{nie2015deep,
  title={A deep framework for bacterial image segmentation and classification},
  author={Nie, Dong and Shank, Elizabeth A and Jojic, Vladimir},
  booktitle={Proceedings of the 6th ACM Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={306--314},
  year={2015},
  organization={ACM}
}
```

Thanks.
