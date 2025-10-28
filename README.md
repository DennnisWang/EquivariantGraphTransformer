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
      title={Three-dimensional geometric deep learning for reaction prediction with equivariant graph transformer}, 
      author={Zhouxiang Wang, Haicheng Yi, Zhuhong You and Qiangguo Jin},
      journal={Engineering Applications of Artificial Intelligence},
      publisher={Elsevier BV},
      doi={https://doi.org/10.1016/j.engappai.2025.112850},
      year={2026},
}
```

