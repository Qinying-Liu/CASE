# Revisiting Foreground and Background Separation in Weakly-supervised Temporal Action Localization: A Clustering-based Approach

The code is assembled to [UniWTAL](https://github.com/Qinying-Liu/UniWTAL), which implements multiple WTAL methods in a unified codebase.

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


 ## References

* [https://github.com/Pilhyeon/BaSNet-pytorch](https://github.com/Pilhyeon/BaSNet-pytorch)
* [https://github.com/layer6ai-labs/ASL](https://github.com/layer6ai-labs/ASL)
