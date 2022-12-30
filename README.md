# SKINET (Segmentation of the KIdney through a Neural nETwork) Project - MEST-C

The SKINET Project is meant to provide a segmentation of the different structures in kidney histological tissue (biopsy or nephrectomy).
This is an updated version of the original [Skinet Tool](https://github.com/SkinetTeam/Skinet) that provides indicators to compute MEST-C score.
It allows the segmentation of sclerotic and non sclerotic glomeruli, healthy or atrophic tubules, intra-glomerular lesions (mesangial hypercellularity, endocapillary hypercellularity, segmental glomerulosclerosis, crescents) and assessment of interstitial fibrosis.

This repository contains all the project's code. 
You can use our online inference tool to test our tool on your biopsies (tutorial in the "docs" folder). 
You can also create a local version of this project by cloning this repo and installing a suitable environment (tutorial in the "docs" folder).
 
The project's code is based on [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) and [Navidyou's repository](https://github.com/navidyou/Mask-RCNN-implementation-for-cell-nucleus-detection-executable-on-google-colab-). The model used in this updated version is based on Mask R-CNN Inception ResNet V2 from [tensorflow/models' Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

This project is a collaboration between a Nephrology team from [Dijon Burgundy Teaching Hospital](https://www.chu-dijon.fr/), [LEAD Laboratory](http://leadserv.u-bourgogne.fr/en/), and a student from [ESIREM](https://esirem.u-bourgogne.fr/), all located in Dijon, Burgundy, France.

## Inference tool
Last : [![Open Inference Tool In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkinetTeam/Skinet-MEST-C/blob/main/Skinet_Inference_Tool.ipynb)

v1.0 : [![Open Inference Tool In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkinetTeam/Skinet-MEST-C/blob/v1.0/Skinet_Inference_Tool.ipynb)

