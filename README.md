# AERO-GNN

This is the official repository of "Towards Deep Attention in Graph Neural Networks: Problems and Remedies," published in ICML 2023.
https://arxiv.org/pdf/2306.02376.pdf

Codes to reproduce node classification results in Tables 3 & 8 are provided. 


## Basics
**AERO-GNN** model code is in model.py. 
Tuned hyperparameters for all models are written in shell files in ./run/model_name.

main.py loads datasets and initializes hyperparameters.
train_dense.py and train_sparse.py load, train, and evaluate GNNs for node classification.
model.py has all the models used for experiments.
layer.py has implementations of some models' graph convolution layers.


## Run Code
```bash
python ./AERO-GNN/main.py --model aero --dataset chameleon --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 2
```

## Citation
```latex
@inproceedings{lee2023towards,
  title={Towards Deep Attention in Graph Neural Networks: Problems and Remedies},
  author={Lee, Soo Yong and Bu, Fanchen and Yoo, Jaemin and Shin, Kijung},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
