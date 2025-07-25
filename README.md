# Equivariant Graph Transformer
3D Geometric Deep Learning for Reaction Prediction with Equivariant Graph Transformer

# Step-by-Step Instructions:
## 1.	Environment Step:
* Clone repository and install dependencies

git clone https://github.com/DennnisWang/EquivariantGraphTransformer.git

cd EquivariantGraphTransformer

pip install -r requirements.txt  
## 2.	Data Preparation:
python download_rwa_data.py

python preprocess.py
## 3.	Run Experiments:
First, training step:

nohup python train.py [parameters]

or sh ./scripts/train_g2s.sh

Then, run validate:
Sh ./scrpits/validate.sh
## 4.	Hardware/Software Requirements
OS: Tested on Ubuntu 20.04

Python: 3.8+

GPU: CUDA 11.8 

Dependencies: Exact versions pinned in requirements.txt

# Citation
```
@misc{
      title={3D Geometric Deep Learning for Reaction Prediction with Equivariant Graph Transformer}, 
      author={Zhouxiang Wang, Haicheng Yi, Zhuhong You and Qiangguo Jin},
      year={2024},
}
```
