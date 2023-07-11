python ./AERO-GNN/main.py --model gat --dataset cora --dr 0.0005 --dropout 0.8 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset citeseer --dr 0.0005 --dropout 0.7 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset pubmed --dr 0.001 --dropout 0.7 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset wiki --dr 0.0005 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2  
python ./AERO-GNN/main.py --model gat --dataset photo --dr 0.0001 --dropout 0.6 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse  
python ./AERO-GNN/main.py --model gat --dataset computers --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse 

python ./AERO-GNN/main.py --model gat --dataset chameleon --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gat --dataset squirrel --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset actor --dr 0.001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset texas --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset wisconsin --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gat --dataset cornell --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2

python ./AERO-GNN/main.py --model gat --dataset chameleon-filtered --dr 0.001 --dropout 0.5 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gat --dataset squirrel-filtered --dr 0.0005 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 