# AERO-GNN

This is the official repository of "[Towards Deep Attention in Graph Neural Networks: Problems and Remedies](https://arxiv.org/abs/2306.02376)," published in ICML 2023. \
Codes to reproduce node classification results in Tables 3 & 8 are provided. \
\
Also, Poster.pdf is the poster file presented @ICML23, with some additional visualizations from the original paper. \
The link to my video presentation is [here](https://www.youtube.com/watch?v=LbTuet6K-so&ab_channel=KAISTDataMiningLab). 

### Table 3
![image](https://github.com/syleeheal/AERO-GNN/assets/66083092/6b2850c6-20bd-471f-84b1-25414eecee64)


## Basics
**AERO-GNN** model code is in _**./AERO-GNN/model.py**_.  \
**Tuned hyperparameters** for all models are written in shell files in _**./run**_ folder. \
**Datasets** are in _**./graph-data**_ folder, which should be automatically downloaded when running the code.


## Codes in AERO-GNN Folder
_**main.py**_ loads datasets, initializes hyperparameters, and runs the entire codes. \
_**train_dense.py**_ and _**train_sparse.py**_ load, train, and evaluate designated GNNs for node classification. \
_**model.py**_ has all the models used for experiments. \
_**layer.py**_ has implementations of some models' graph convolution layers. 


## Run Code
The code will run 100 trials of node classification on the designated dataset. The results for every trial will be printed every trial. The codes are saved in shell files in ./run folder.
### Example: Running Chameleon 
```bash
python ./AERO-GNN/main.py --model aero --dataset chameleon --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 2
```


## Datasets
Running _**main.py**_ will automatically download the designated datasets from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). \
The codes to load the filtered **Chameleon** and **Squirrel** datasets, proposed by [Platonov et al. (2023, ICLR)](https://arxiv.org/pdf/2302.11640.pdf), are in _**filtered_dataset.py**_. \
The loading and preprocessing code for each dataset is in _**utils.py**_. 


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
