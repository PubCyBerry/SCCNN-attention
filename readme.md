# ADHD-Classification-with-rs-FMRI

## ToC

- [ADHD-Classification-with-rs-FMRI](#adhd-classification-with-rs-fmri)
  - [ToC](#toc)
  - [Paper](#paper)
  - [Data](#data)
  - [Exp. Setup](#exp-setup)

---

## Paper

1. [3D CNN Based Automatic Diagnosis of Attention
   Deficit Hyperactivity Disorder Using Functional
   and Structural MRI](https://ieeexplore.ieee.org/abstract/document/8067637)
2. [Separated Channel Attention Convolutional Neural
   Network (SC-CNN-Attention) to Identify ADHD in
   Multi-Site Rs-fMRI Dataset](https://www.mdpi.com/1099-4300/22/8/893)

---

## Data

1. [RfMRI](http://rfmri.org/maps) :  `wget -r --ftp-user=ftpdownload --ftp-password=FTPDownload ftp://lab.rfmri.org/sharing/RfMRIMaps/ADHD200`

---

## Exp. Setup

Docker

- pyTorch 1.10.0
- python 3.7.11

  - `docker pull pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`
