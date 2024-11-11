# <h1 align="center">  <b>VHF_ParaNet|VISION TRANSFORMERS FEATURE HARMONIZATION NETWORK FOR  HIGH-RESOLUTION LAND COVER MAPPING </b><br></h1>

<h2 align="left">Authors <br></h2>

[![Author](https://img.shields.io/badge/Boaz-MWUBAHIMANA-orange.svg)](https://github.com/BoazGithub) 
[![Author](https://img.shields.io/badge/YAN-Jianguo-orange.svg)](http://www.lmars.whu.edu.cn/enjianguo-yan/) 
[![Author](https://img.shields.io/badge/Swalpa-KumarRoy-orange.svg)](https://ieeexplore.ieee.org/author/37086689617) 
[![Author](https://img.shields.io/badge/Maurice-Mugabowindekwe-orange.svg)](https://researchprofiles.ku.dk/en/persons/maurice-mugabowindekwe) 
[![Author](https://img.shields.io/badge/Xiao-Huang-orange.svg)](https://envs.emory.edu/people/bios/Huang-Xiao%20.html) 
[![Author](https://img.shields.io/badge/Elias-Nyandwi-orange.svg)](https://cst.ur.ac.rw/?Dr-Elias-Nyandwi-723) 
[![Author](https://img.shields.io/badge/Eric-Habineza-orange.svg)](https://www.linkedin.com/in/eric-habineza-79559519b/?originalSubdomain=rw) 
[![Author](https://img.shields.io/badge/Fidele-Mwizerwa-orange.svg)](https://cst.ur.ac.rw/?Mrs-Fidele-MWIZERWA) 
[![Author](https://img.shields.io/badge/Joseph-Tuyishimire-orange.svg)](https://cst.ur.ac.rw/?Mr-Joseph-Tuyishimire) 
[![Author](https://img.shields.io/badge/Dingruibo-Miao-orange.svg)](https://ieeexplore.ieee.org/author/37089315877) 

## Updates
| :zap:         | November, 2024: VHF_ParaNet has been submitted for publication at the International Journal of Applied Earth Observation and Geoinformation. |
|---------------|:------------------------|

This study addresses the challenge of autonomous high-resolution land cover mapping, which is vital for understanding ecological dynamics and the Earth's surface. The research utilizes the GLC10 dataset, a freely available global low-resolution land cover product, to inform fine-scale mapping, while high-resolution imagery from Google Earth serves as the primary training data for the model. To overcome the limitations of traditional CNNs in capturing global context, we introduce Vision Feature Harmonisation Learning (VHF-Para Learning), a novel framework that integrates Vision Transformers with CNNs for parallel feature learning and improved edge refinement. VHF_ParaNet Model and sKwanda_V2_(Bugesera_Rwamagana{Rwanda}) and Oklahoma{USA} dataset can be downloaded [[here](https://drive.google.com/drive/folders/1h9T6w84P8b2xyD81at4JMn30VekAS53E?usp=drive_link)], [[here](https://drive.google.com/file/d/1X_Fz7LQIeix3rV3K29FBfKiU1WMdROe-/view?usp=drive_link)]  respectively.
 
## Graphical abstract: 

![Illustration2_of_resolution mismatch_page-0001](https://github.com/user-attachments/assets/1830acbc-aa7f-4e49-b650-038919f72964)

Graphical abstract illustrating fine-scale mapping using noise labels. It highlights the issue of spatial resolution mismatch between high-resolution (HR) remote sensing images and pseudo-low-resolution (PLR) ground truth (GT) labels. An algorithm is needed to reduce inconsistencies and lead to uncertainty in predictions. The final results support real-world applications such as forests, agriculture, urbanization, and natural hazard monitoring for early warnings.

# Requirements 


[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/) 
[![Pytorch 1.7.1](https://img.shields.io/badge/Pytorch-1.7.1-blue.svg)](https://pytorch.org/get-started/previous-versions/)
[![torchvision 0.8.2](https://img.shields.io/badge/torchvision-0.8.2-blue.svg)](https://pypi.org/project/torchvision/0.8.2/)
[![Opencv 4.5.5](https://img.shields.io/badge/Opencv-4.5.5-blue.svg)](https://opencv.org/opencv-4-5-5/)
[![CUDA Toolkit 10.1](https://img.shields.io/badge/CUDA-10.1-blue.svg)](https://developer.nvidia.com/cuda-10.1-download-archive-base)
[![Wandb 0.13.10](https://img.shields.io/badge/Wandb-0.13.10-blue.svg)](https://pypi.org/project/wandb/)
## Description

###  Study Area
The Vision transformer feature harmonization network was trained over Nyagatare, and Rwamagana{top-left-corner at red A}, and migration was made over the National Agriculture imagery program{top-left-corner at red B} and CheasaBay land cover products

![3MVH_ParaNetl_2024](https://github.com/user-attachments/assets/6dbddb6a-6101-4944-84c6-b59101ff4345)

###  Study Area Data descriptions:
| Image Ref. |      Site     | Image Acquisition Date  |   GT Date   |
| ---------- | ------------- | ----------- | ------------ | 
|   Img (1)  |   The City of Kigali  |  04-03-2023 |  27-06-2023 |
|   Img (2)  |      Bugesera        |  22-07-2023  |  25-08-2023  |  
|   Img (3)  |   Oklahoma State    |  26-08-2022  |  03-80-2022  |  



# VHF_ParaNet Architecture

The VHF-Net architecture consists of three core components: a CNN and ViT branch, a deep noise-label assistant training (NLAT) module, and a multi-layer agent (MLA) module. These components use attention mechanisms and residual learning to focus on key regions while removing irrelevant background noise, enabling the model to capture distinctive features for land cover classification. The CNN-based deep attention module further extracts high-level features from the input images to enhance classification accuracy. 
### Algorithm flows:
1. ![image](https://github.com/user-attachments/assets/c9925924-8750-41c3-86b5-8fe279f437c5)

### VFH_Para workflows
2. ![CNN_ViT_branchs VFH_ISDE_Rda_original7_page-0001 (1)](https://github.com/user-attachments/assets/b51404f5-b143-4116-89be-f2fdc1efdd90)



# Results:

##### Tracking training progress:
in both valuation and validation in a variety of metrics ranging from how our loss function is declining, Accuracy, F1-Score, Recall, and their validation
![image](https://github.com/user-attachments/assets/316990c2-101f-4159-9397-22671d9b0863)

#### Harmonization stages and improvement of feature pattern recognition 

![Features_flowsaF3_page-0001](https://github.com/user-attachments/assets/2db43d0a-a2ed-40a8-a763-e1782b69c191)


# Quantitative Results

![image](https://github.com/user-attachments/assets/9d7fb582-3134-43c6-9c44-0daaf8b3d362)


# Qualitative Results

![nyagate_multimodel_performance_metrics_histograms_with_lines_and_unique_colormap_page-0001](https://github.com/user-attachments/assets/9c1c3e94-46a5-4a65-b35a-229156094ea0)

## SoA

1. ![image](https://github.com/user-attachments/assets/35db271b-8d92-480d-9f36-24ea08ad41ec)



2. ![VHF_Para_Dataset_correlation_heatmap5_page-0001](https://github.com/user-attachments/assets/5d3040cb-7e4c-4cde-8b5d-b7b244091aa6)


### 🔭 Baseline:

📖 📖 📖 
- :open_book:	:open_book:	 :open_book: DTCDSCN [[here](https://www.sciencedirect.com/science/article/abs/pii/S0924271622002180)]
- :open_book:	:open_book:	 :open_book: UNet [[here](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIV-4-W3-2020/215/2020/)]
- :open_book:	:open_book:	 :open_book: ResNet50-IMP [[here](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
- :open_book:	:open_book:	 :open_book: ResNet50-RSP [[here](https://ieeexplore.ieee.org/abstract/document/9782149)]
- :open_book:	:open_book:	 :open_book: ViTAEv2 [[here](https://arxiv.org/pdf/2202.10108.pdf)]
📖 📖 📖


💬 Dataset Preparation


👉 Data Structure
### Dataset Overview
The sKwanda_V1_d dataset includes 256 × 256 pixel image patches collected from various regions. It is organized into three subsets: train, val, and test, each containing images and their corresponding ground truth labels. This dataset supports tasks such as supervised land cover classification and semantic segmentation.
### sKwanda_V1_d Dataset

The *sKwanda_V1_d* dataset is organized into training, validation, and testing sets, each containing images and ground truth (GT) labels for land cover mapping. The structure is as follows:

### Dataset Structure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


### Define the dataset structure
🚚 Datasets


- [x] [sKwanda)V1_dataset_Bugesera][Google Drive Link]([https://drive.google.com/file/d/1W-gnUU-AaYbJ8KMdfnbrI7ySHkiKjOvo/view?usp=drive_link](https://drive.google.com/file/d/1X_Fz7LQIeix3rV3K29FBfKiU1WMdROe-/view?usp=drive_link)


###  Contact Information:


If you have any questions or would like to collaborate, please reach out to me at aiboaz1896@gmail.com or feel free to make issues.

### License: 


The code and datasets are released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

### Acknowledgment:


Appreciate the work from the following repositories:


Planetary Science group at the State key laboratory of information engineering in surveying, mapping and remote sensing of the Wuhan University 

1. L2HNet

2. Related resources:


3. L2HNet dataset


4. Sentine2-Hub


5. ESRI 
