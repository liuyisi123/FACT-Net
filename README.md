### **Edge-Intelligent Cross-Platform Architecture for Continuous Non-Invasive Arterial Blood Pressure Reconstruction in Distributed Healthcare IoT Networks**

---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13+](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20With%20Applications-green)](https://www.elsevier.com/locate/eswa)

Official PyTorch implementation of **"Edge-Intelligent Cross-Platform Architecture for Knowledge-Intensive Arterial Blood Pressure Inference in Distributed Healthcare IoT Networks"** published in *Expert Systems With Applications*.



## **🧠 Multimodal Physiological Signal Acquisition System**  
The **multimodal physiological signal acquisition system** developed in this study integrates cutting-edge hardware components, ensuring precise, reliable, and efficient data acquisition for ABP reconstruction tasks in diverse healthcare settings.

### **🔧 System Components**  
1. **Hardware Modules**:  
   - **(a)** 🛠 **Sensor Unit**: Collects physiological signals (e.g., ECG, PPG).  
   - **(b)** 📊 **Signal Processing Circuitry**: Processes raw signals into usable data.  
   - **(c)** 🖥 **Microcontroller Unit (MCU)**: Coordinates the system’s operations.  
   - **(d)** 💾 **Data Processing Module**: Handles complex computations for signal analysis.  
   - **(e)** 🔋 **Charging Unit**: Powers the system with high-efficiency charging.  
2. **System Design**:  
   - **(f)** 🖼 **Simulation Diagram**: Illustrates the functional design of the system.  
   - **(g)** 🔲 **Front View**: Showcases the compact and ergonomic design.  
   - **(h)** 🔲 **Rear View**: Highlights modularity for easy upgrades.  
   - **(i)** 🌟 **PPG Sensor**: Provides high-accuracy signal acquisition.  
   - **(j)** 🔋 **3D-Printed Casing**: Integrated with a lithium battery for enhanced portability and durability.  


![CPMP-IoT Framework](https://github.com/liuyisi123/FACT-Net/blob/main/CPMP-IoT.png)  



---

## ✨ Key Features

- ✅ **Two-Stage Architecture**: Combines local feature extraction (CNN) and global dependency modeling (Transformer)
- ✅ **Multi-Modal Fusion**: Synchronized ECG and PPG signal processing with intelligent feature integration
- ✅ **Edge Optimization**: 90% parameter reduction via structured pruning while maintaining clinical accuracy
- ✅ **Clinical Standard Compliance**: Meets AAMI/BHS/IEEE Grade A standards
- ✅ **Cross-Database Validation**: Robust performance across VitalDB, MIMIC-III, UCI, and private datasets
- ✅ **Real-Time Processing**: Sub-3ms latency for continuous monitoring applications
- ✅ **Scalable Deployment**: From individual edge devices to multi-patient hospital systems

---
### FACT-Net Architecture Details
```
Stage I: Parallel Cross-Hybrid Modeling (PCHM)
├── CNN Backbone
│ ├── Stem Layer (Conv1d + BN + h-swish)
│ ├── Layer 1-4 (Multi-scale DW-Conv)
│ └── Hybrid Attention Mechanism
├── Mix-Transformer Backbone
│ ├── Hierarchical Structure (4 stages)
│ ├── W-MSA / SW-MSA alternating blocks
│ └── Hybrid Self-Attention + Convolution
└── Output: Feature Maps + BP Category (4 classes)

Stage II: Serial Hybrid CNN-Transformer (SHCT)
├── Feature Extractor
│ └── Dilated Convolutions (rates: 1, 2)
├── Transformer Encoder-Decoder
│ ├── Encoder: 6 layers, 8 heads
│ └── Decoder: 6 layers, 8 heads
└── Feature Reconstructor
└── Linear layers → ABP Waveform
```
---

## 📦 Installation

### Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.11 or higher
- **CUDA**: 11.7+ (for GPU acceleration)
- **Hardware**: 
  - Training: GPU with ≥16GB VRAM (A100/V100 recommended)
  - Inference: CPU or GPU with ≥4GB VRAM

### Step 1: Clone Repository

```
git clone https://github.com/liuyisi123/FACTNET.git
cd FACT-Net-ABP
```
### Step 2: Create Conda Environment

#### 1) Create new environment
```
conda create -n factnet python=3.11 -y
conda activate factnet
```
#### 2) Install PyTorch with CUDA support
```
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c 
```
##### Step 1: Install Dependencies
```
pip install -r requirements.txt
```
##### Step 2: Verify Installation
``` 
import torch
print(f'PyTorch: {torch.__version__}')
``` 
## 📊 Data Preparation
### Step 1: Obtain PhysioNet credentials at https://physionet.org/
### Step 2: Download database
``` 
VitalDB: https://vitaldb.net/; 
MIMIC-III: https://physionet.org/content/mimic3wdb/1.0/;
UCI:https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
``` 
### Step 3: Preprocess
``` 
python scripts/preprocess_mimic3.py \
    --input_dir ./mimic3wdb/1.0 \
    --output_dir ./data/mimic3 \
    --num_subjects 1000
``` 

## 🚀 Quick Start
### 1.Training
``` 
python train.py \
    --config configs/factnet_full.yaml \
    --dataset vitaldb \
    --data_dir ./data/vitaldb/splits \
    --output_dir ./outputs/factnet_full \
    --gpu 0 \
    --seed 42
``` 
### 📈 2.Evaluation
#### Evaluate on Test Set
``` 
python evaluate.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --dataset vitaldb \
    --data_dir ./data/vitaldb/splits \
    --split test \
    --output_dir ./results/vitaldb_test \
    --gpu 0
``` 

#### Evaluate on MIMIC-III
``` 
python evaluate.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --dataset mimic3 \
    --data_dir ./data/mimic3/splits \
    --output_dir ./results/mimic3_test \
    --gpu 0
``` 
#### Evaluate on UCI
``` 
python evaluate.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --dataset uci \
    --data_dir ./data/uci/splits \
    --output_dir ./results/uci_test \
    --gpu 0
``` 
#### Evaluate on Private Dataset
``` 
python evaluate.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --dataset private \
    --data_dir ./data/private/splits \
    --output_dir ./results/private_test \
    --gpu 0
``` 
### 3.Reconstruction plots
``` 
python visualize/plot_reconstruction.py \
    --results_dir ./results/vitaldb_test \
    --output_dir ./figures/reconstruction \
    --num_samples 20
``` 
### 4.Bland-Altman plots
```
python visualize/bland_altman.py \
    --results_dir ./results/vitaldb_test \
    --output_dir ./figures/bland_altman
```
### 5.t-SNE feature visualization
```
python visualize/tsne_features.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --dataset vitaldb \
    --data_dir ./data/vitaldb/splits \
    --output_dir ./figures/tsne \
    --num_samples 5000 \
    --gpu 0
```
## 🔬 Model Compression

### 1.30% Sparsity (Recommended for balanced performance)
```
python prune_model.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --sparsity 0.3 \
    --method structured \
    --output_dir ./outputs/factnet_pruned_30 \
    --gpu 0
```
### 2.70% Sparsity (Ultra-lightweight for resource-constrained devices)
```
python prune_model.py \
    --checkpoint ./outputs/factnet_full/best_model.pth \
    --sparsity 0.7 \
    --method structured \
    --output_dir ./outputs/factnet_pruned_70 \
    --gpu 0
```    
### 3.Fine-Tune Pruned Model
``` 
python train.py \
    --config configs/factnet_finetune.yaml \
    --checkpoint ./outputs/factnet_pruned_30/pruned_model.pth \
    --dataset vitaldb \
    --data_dir ./data/vitaldb/splits \
    --output_dir ./outputs/factnet_pruned_30_finetuned \
    --num_epochs 100 \
    --gpu 0
``` 




## **📑 Appendices**  

### **📐 Appendix I: Circuit Schematic**  
The **circuit schematic** provides a comprehensive illustration of the hardware design, detailing the interconnections between key components in the physiological signal acquisition system.  
[**Download Circuit Schematic PDF**](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-II-Circuit%20Schematic.pdf)  

### **🖥 Appendix II: PCB Design**  
The **PCB design** outlines the printed circuit board layout, ensuring optimal integration and functionality of the system components.  
[**Download PCB Design PDF**](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-III-PCB.pdf)

---

## **💾 Download Buttons**  
You can easily download the **circuit schematic** and **PCB design** from the following links:

[![Download Circuit Schematic](https://img.shields.io/badge/Download%20Circuit%20Schematic-blue?style=for-the-badge&logo=github)](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-II-Circuit%20Schematic.pdf)  
[![Download PCB Design](https://img.shields.io/badge/Download%20PCB%20Design-blue?style=for-the-badge&logo=github)](https://github.com/liuyisi123/FACT-Net/blob/main/Appendix-III-PCB.pdf)

## 📖 Citation
``` 
@article{liu2025factnet,
  title={Edge-Intelligent Cross-Platform Architecture for Knowledge-Intensive Arterial Blood Pressure Inference in Distributed Healthcare IoT Networks},
  author={Liu, Jian and Hu, Shuaicong and Wang, Yanan and Wang, Daomiao and Yang, Cuiwei},
  journal={Expert Systems With Applications},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2025.xxxxx}
}
``` 
