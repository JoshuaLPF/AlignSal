# AlignSal -- TGRS 2025
## Efficient Fourier Filtering Network with Contrastive Learning for AAV-based Unaligned Bi-modal Salient Object Detection [[paper]](https://arxiv.org/abs/2411.03728)
- **April 2, 2025**  
  The paper is accepted by TGRS:tada::tada:
- **November 4, 2024**  
  The paper is undergoing peer review. The code will be released upon acceptance of the paper.
- ![Framework](https://github.com/JoshuaLPF/AlignSal/blob/main/Figure/framework.png)
- In this project, we designed **AlignSal**, which achieves both real-time performance and high accuracy for UAV-based unaligned Bi-modal Salient Object Detection (BSOD).  
- Please cite our paper if you find it useful for your research.
```
@ARTICLE{10975009,
  author={Lyu, Pengfei and Yeung, Pak-Hei and Yu, Xiaosheng and Cheng, Xiufei and Wu, Chengdong and Rajapakse, Jagath C.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Efficient Fourier Filtering Network With Contrastive Learning for AAV-Based Unaligned Bimodal Salient Object Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-12},
  publisher={IEEE}
}
```

## Requirements

List of prerequisites or required libraries for the project to run:

- Pytorch 2.0.0
- Cuda 11.8
- Python 3.8 or higher
- tensorboardX
- opencv-python
- timm==0.6.13
- thop
- numpy

## Datasets in paper
- UAV RGB-T 2400: [link](https://github.com/VDT-2048/UAV-RGB-T-2400);
- UNVT821, UNVT1000, UNVT5000: [link](https://github.com/lz118/Deep-Correlation-Network).

## Results
The results of our AlignSal and other SOTA models
### UAV RGB-T 2400:
- AlignSal: [link](https://pan.baidu.com/s/1M2xWybKfdOV3GLhnxFQlQg?pwd=rxyj);
- Comparison model: [link](https://pan.baidu.com/s/165OwbmbMzwb5gPvwzBSpOQ?pwd=28f5).
### UNVT821, UNVT1000, UNVT5000
- AlignSal: [link](https://pan.baidu.com/s/1hhboN8oskn4JPgXPgZ6kaA?pwd=8fvr);
- Comparison model: [link](https://pan.baidu.com/s/1oHcMoWgNS_0Ep43fegFUNA?pwd=nuns).

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors, and projects (*e.g.*, [MROS](https://github.com/VDT-2048/UAV-RGB-T-2400), [ContrastAlign](https://github.com/modaxiansheng/ContrastAlign/), [DCNet](https://github.com/lz118/Deep-Correlation-Network), and [SwinNet](https://github.com/liuzywen/SwinNet)).

## Contact Us
If you have any questions, please contact us (lvpengfei1995@163.com).
