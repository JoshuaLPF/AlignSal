# AlignSal
## Efficient Fourier Filtering Network with Contrastive Learning for UAV-based Unaligned Bi-modal Salient Object Detection
In this project, we designed **AlignSal**, which achieves both real-time performance and high accuracy for UAV-based unaligned Bi-modal Salient Object Detection (BSOD). To this end, we first designed SCAL to align bi-modal features in a contrastive manner and enhance inter-modal information exchange. **Notably, SCAL not only works on our proposed AlignSal, but also improves the performance of various existing aligned BSOD models on the unaligned BSOD data.** Second, we proposed the SAF to align slightly offset features in multiple dimensions and facilitate bi-modal fusion by efficiently acquiring global relevance. Extensive experiments demonstrated that AlignSal achieves significantly faster inference speed and superior performance compared to existing models on the UAV RGB-T 2400 dataset and three weakly aligned datasets. These advantages make AlignSal suitable for deployment on UAVs in practice, enabling its use in industrial and real-life applications that require both timeliness and accuracy.

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
- VT821, VT1000, VT5000: Baidu cloud disk [link](https://pan.baidu.com/s/1Vv6mYz4RL2VnwWwZWKLHyA), fetch code (djas); 
- VI-RGBT1500: [link](https://github.com/huanglm-me/VI-RGBT1500)
## Pertrain Models

|          |Params(M)| FLOPs(G)| Input size |  Backbone   |      |
|----------|---------|---------|------------|-------------|------|
| DFENet-m | 149.2   |  139.7  |  384x384   | CDFFormer-m | [link](https://pan.baidu.com/s/1j_u9YGr-9zwNOJHJ9WCuqQ), fetch code(kbpy) |
| DFENet-b | 264.6   |  238.0  |  384x384   | CDFFormer-b | [link](https://pan.baidu.com/s/1S23SqxzzsNj-39nkaxb7xA), fetch code(eysg) |
| DFENet-h | 149.8   |  248.3  |  512x512   | CDFFormer-m | [link](https://pan.baidu.com/s/1kaJ4ukqcfhdoq1wvq9EDeQ), fetch code(ph20) |

## Results
The results of our DFENets (-m, -h, and -b) can be found at [link](https://pan.baidu.com/s/19aWbiGBD6AqWrP0e_PwYWw), fetch code(1ryz).

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors who put in the effort.

## Contact Us
If you have any questions, please contact us ( lvpengfei1995@163.com ).
