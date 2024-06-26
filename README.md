# Revisiting Foreground and Background Separation in Weakly-supervised Temporal Action Localization: A Clustering-based Approach

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-foreground-and-background-1/weakly-supervised-action-localization-on-2)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on-2?p=revisiting-foreground-and-background-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-foreground-and-background-1/weakly-supervised-action-localization-on-1)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on-1?p=revisiting-foreground-and-background-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-foreground-and-background-1/weakly-supervised-action-localization-on)](https://paperswithcode.com/sota/weakly-supervised-action-localization-on?p=revisiting-foreground-and-background-1)


The code is assembled to [OpenWTAL](https://github.com/Qinying-Liu/OpenWTAL), which implements multiple WTAL methods in a unified codebase.

<div align="center">
<img src="figures/framework.png" width="100%">
</div>

> **Revisiting Foreground and Background Separation in Weakly-supervised Temporal Action Localization: A Clustering-based Approach** <br>
> [Qinying Liu](https://scholar.google.com/citations?user=m8Wioy0AAAAJ), [Zilei Wang](https://scholar.google.com/citations?user=tMO7jm4AAAAJ), [Shenghai Rong](https://scholar.google.com/citations?user=A0_yGSoAAAAJ), [Junjie Li](https://scholar.google.com/citations?user=3fXVH5oAAAAJ), [Yixin Zhang](https://scholar.google.com/citations?user=F24AuKYAAAAJ)<br>
> **ICCV2023** <br>

[[Paper](https://arxiv.org/pdf/2312.14138.pdf)]


## Data Preparation
1. Download the features of THUMOS14 from [dataset.zip](https://rec.ustc.edu.cn/share/e1472d30-5f38-11ee-a8ae-cff932c459ec). 
2. Place the features inside the `./data` folder.

## Train and Evaluate
1. Train the CASE model by run 
   ```
   python main_case.py --exp_name CASE
   ```
2. The pre-trained model will be saved in the `./outputs` folder. You can evaluate the model by running the command below.
   ```
   python main_case.py --exp_name CASE --inference_only
   ```
   We provide our pre-trained checkpoints in [checkpoints.zip](https://rec.ustc.edu.cn/share/ba014cd0-5f3a-11ee-9e84-ebc071ec19ed)

<div align="center">
<img src="figures/sota.png" width="80%">
</div>

<div align="center">
<img src="figures/exp.png" width="80%">
</div>

 ## References

* [https://github.com/Pilhyeon/BaSNet-pytorch](https://github.com/Pilhyeon/BaSNet-pytorch)
* [https://github.com/layer6ai-labs/ASL](https://github.com/layer6ai-labs/ASL)
