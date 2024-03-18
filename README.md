# A multi-task learning method for extraction of newly constructed areas based on bi-temporal hyperspectral images

The Pytorch implementation of the paper "A multi-task learning method for extraction of newly constructed areas based on bi-temporal hyperspectral images".

## Example of usage

Step 1: Run train_SAHR_Net.py to train the SAHR_Net feature extractor.

Step 2: Run train_Multitask.py to train the semantic segmentation module and DMAD change detection module in the multi-task framework.

Step 3: Run predict.py to predict semantic segmentation and change detection results.

Step 4: Run generate_NCA.py to generate NCA according to semantic segmentation and change detection results.

## Paper

L. Tu, X. Huang, J. Li, J. Yang, and J. Gong, “A multi-task learning method for extraction of newly constructed areas based on bi-temporal hyperspectral images,” ISPRS J. Photogramm. Remote Sens., vol. 208, pp. 308–323, 2024. [[Link]](https://www.sciencedirect.com/science/article/pii/S092427162400025X)

