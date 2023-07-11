python ./AERO-GNN/main.py --model gatv2 --dataset cora --dr 0.0005 --dropout 0.8 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset citeseer --dr 0.0001 --dropout 0.6 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2
python ./AERO-GNN/main.py --model gatv2 --dataset pubmed --dr 0.0001 --dropout 0.7 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset wiki --dr 0.0005 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2  
python ./AERO-GNN/main.py --model gatv2 --dataset photo --dr 0.0001 --dropout 0.6 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse 
python ./AERO-GNN/main.py --model gatv2 --dataset computers --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse

python ./AERO-GNN/main.py --model gatv2 --dataset chameleon --dr 0.0001 --dropout 0.6 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset squirrel --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset actor --dr 0.0001 --dropout 0.5 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset texas --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset wisconsin --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset cornell --dr 0.0005 --dropout 0.5 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2

python ./AERO-GNN/main.py --model gatv2 --dataset chameleon-filtered --dr 0.001 --dropout 0.6 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gatv2 --dataset squirrel-filtered --dr 0.0005 --dropout 0.6 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 

