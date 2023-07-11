# AERO-GNN

This is the official repository of "Towards Deep Attention in Graph Neural Networks: Problems and Remedies," published in ICML 2023 (https://arxiv.org/pdf/2306.02376.pdf).

Codes to reproduce node classification results in Tables 3 & 8 are provided. 

### Table 3
![image](https://github.com/syleeheal/AERO-GNN/assets/66083092/6b2850c6-20bd-471f-84b1-25414eecee64)


## Basics
**AERO-GNN** model code is in **model.py**. 

**Tuned hyperparameters** for all models are written in shell files in ./run/model_name.

**main.py** loads datasets and initializes hyperparameters.

**train_dense.py** and **train_sparse.py** load, train, and evaluate GNNs for node classification.

**model.py** has all the models used for experiments.

**layer.py** has implementations of some models' graph convolution layers.


## Run Code
The code will run 100 trials of node classification on the designated dataset. The results for every trial will be printed every trial. The codes are saved in shell files in ./run folder.
### Example: Running Cora 
```bash
python ./AERO-GNN/main.py --model aero --dataset chameleon --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 2
```


## Requirements
```latex
dgl==1.1.1
dgl_cu113==0.9.1.post1
numpy==1.21.2
torch==1.11.0+cu113
torch_geometric==2.1.0
torch_scatter==2.0.9
torch_sparse==0.6.13
tqdm==4.62.3
```

## Bibtex
```latex
@inproceedings{lee2023towards,
  title={Towards Deep Attention in Graph Neural Networks: Problems and Remedies},
  author={Lee, Soo Yong and Bu, Fanchen and Yoo, Jaemin and Shin, Kijung},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## Contacts
For any question, please email us ({syleetolow, boqvezen97, kijungs}@kaist.ac.kr, {jaeminyoo}@cmu.edu)! 
