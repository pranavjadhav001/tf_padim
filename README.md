# tf_padim
unofficial implementation of PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization<br />
paper link : https://arxiv.org/pdf/2011.08785v1.pdf

## Requirements
Tensorflow - 2.5.0<br />
scikit-image - 0.17.2<br />
image-classifiers - 1.0.0b1<br />
opencv - 4.5.3.56<br />

## Usage
python train.py<br /> --base_path mvtec_anomaly_detection_folder<br /> --folder_path bottle<br /> --model resnet18<br /> --dim 100<br />
--image_size 256 256 3<br /> --center_size 256 256 3<br />
## References
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master<br />
https://github.com/remmarp/PaDiM-TF
