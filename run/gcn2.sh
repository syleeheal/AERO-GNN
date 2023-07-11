python ./AERO-GNN/main.py --model gcn2 --dataset cora --num-layers 64 --alpha 0.1 --lambd 0.5 --hid-dim 64 --dr 5e-4 --dr-prop 0.01 --dropout 0.6  
python ./AERO-GNN/main.py --model gcn2 --dataset citeseer --num-layers 32 --alpha 0.1 --lambd 0.6 --hid-dim 256 --dr 5e-4 --dr-prop 0.01 --dropout 0.7  
python ./AERO-GNN/main.py --model gcn2 --dataset pubmed --num-layers 16 --alpha 0.1 --lambd 0.4 --hid-dim 256 --dr 5e-4 --dr-prop 5e-4 --dropout 0.5 
python ./AERO-GNN/main.py --model gcn2 --dataset wiki --num-layers 4 --alpha 0.1 --lambd 0.5 --hid-dim 64 --dr 5e-5 --dr-prop 0.001 --dropout 0.5 
python ./AERO-GNN/main.py --model gcn2 --dataset photo --num-layers 32 --alpha 0.2 --lambd 0.5 --hid-dim 64 --dr 5e-06 --dr-prop 0.005 --dropout 0.7 --split sparse
python ./AERO-GNN/main.py --model gcn2 --dataset computers --num-layers 4 --alpha 0.3 --lambd 0.5 --hid-dim 64 --dr 1e-05 --dr-prop 0.0005 --dropout 0.7 --split sparse

python ./AERO-GNN/main.py --model gcn2 --dataset chameleon --lr 0.01 --num-layers 8 --alpha 0.2 --lambd 1.5 --hid-dim 64 --dr 5e-4 --dr-prop 5e-4 --dropout 0.5 
python ./AERO-GNN/main.py --model gcn2 --dataset squirrel --lr 0.005 --num-layers 4 --alpha 0.1 --lambd 1.0 --hid-dim 64 --dr 0.0005 --dr-prop 0.0001 --dropout 0.6 
python ./AERO-GNN/main.py --model gcn2 --dataset actor --lr 0.005 --num-layers 4 --alpha 0.3 --lambd 1.5 --hid-dim 64 --dr 0.001 --dr-prop 0.001 --dropout 0.7 
python ./AERO-GNN/main.py --model gcn2 --dataset texas --lr 0.005 --num-layers 4 --alpha 0.5 --lambd 1.5 --hid-dim 64 --dr 0.0001 --dr-prop 0.0005 --dropout 0.8 
python ./AERO-GNN/main.py --model gcn2 --dataset wisconsin --lr 0.005 --num-layers 4 --alpha 0.5 --lambd 1.0 --hid-dim 64 --dr 5e-05 --dr-prop 0.0005 --dropout 0.8 
python ./AERO-GNN/main.py --model gcn2 --dataset cornell --lr 0.005 --num-layers 4 --alpha 0.5 --lambd 1.5 --hid-dim 64 --dr 5e-06 --dr-prop 0.005 --dropout 0.5

python ./AERO-GNN/main.py --model gcn2 --dataset chameleon-filtered --lr 0.005 --num-layers 4 --alpha 0.1 --lambd 1.5 --hid-dim 64 --dr 0.0005 --dr-prop 0.01 --dropout 0.8 --lambd-l2 0.0
python ./AERO-GNN/main.py --model gcn2 --dataset squirrel-filtered --lr 0.005 --num-layers 4 --alpha 0.1 --lambd 1.5 --hid-dim 64 --dr 0.001 --dr-prop 0.001 --dropout 0.8 --lambd-l2 0.0005
