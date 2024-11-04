# AlignSal
## Efficient Fourier Filtering Network with Contrastive Learning for UAV-based Unaligned Bi-modal Salient Object Detection
- (Code is coming soon.)
- In this project, we designed **AlignSal**, which achieves both real-time performance and high accuracy for UAV-based unaligned Bi-modal Salient Object Detection (BSOD). To this end, we first designed SCAL to align bi-modal features in a contrastive manner and enhance inter-modal information exchange. **Notably, SCAL not only works on our proposed AlignSal but also improves the performance of various existing aligned BSOD models on the unaligned BSOD data.** Second, we proposed the SAF to align slightly offset features in multiple dimensions and facilitate bi-modal fusion by efficiently acquiring global relevance. Extensive experiments demonstrated that AlignSal achieves significantly faster inference speed and superior performance compared to existing models on the UAV RGB-T 2400 dataset and three weakly aligned datasets. These advantages make AlignSal suitable for deployment on UAVs in practice, enabling its use in industrial and real-life applications that require both timeliness and accuracy.

## Requirements

List of prerequisites or required libraries for the project to run:

- Pytorch 2.0.0
- Cuda 11.8
- Python 3.8 or higher
- tensorboardX
- opencv-python
- timm == 0.5.4
- thop
- numpy

## Datasets
- UAV RGB-T 2400: [link](https://github.com/VDT-2048/UAV-RGB-T-2400); 
- UNVT821, UNVT1000, UNVT5000: [link](https://github.com/lz118/Deep-Correlation-Network)

## Results
The results of our AlignSal and other SOTA models
### UAV RGB-T 2400:
-AlignSal: [link](https://pan.baidu.com/s/1M2xWybKfdOV3GLhnxFQlQg?pwd=rxyj), (rxyj);  
-Comparison model: [link](https://pan.baidu.com/s/165OwbmbMzwb5gPvwzBSpOQ?pwd=28f5), (28f5).
### UNVT821, UNVT1000, UNVT5000
-AlignSal: [link](https://pan.baidu.com/s/1hhboN8oskn4JPgXPgZ6kaA?pwd=8fvr), fetch code(8fvr);
-Comparison model: [link](https://pan.baidu.com/s/1oHcMoWgNS_0Ep43fegFUNA?pwd=nuns), fetch code(nuns).

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors!

## Contact Us
If you have any questions, please contact us (lvpengfei1995@163.com).
