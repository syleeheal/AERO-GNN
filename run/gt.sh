python ./AERO-GNN/main.py --model gt --dataset cora --dr 0.001 --dropout 0.8 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset citeseer --dr 0.0005 --dropout 0.6 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gt --dataset pubmed --dr 0.0005 --dropout 0.5 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gt --dataset wiki --dr 0.0005 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 2 
python ./AERO-GNN/main.py --model gt --dataset photo --dr 0.0001 --dropout 0.8 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse 
python ./AERO-GNN/main.py --model gt --dataset computers --dr 0.0001 --dropout 0.7 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 --split sparse 

python ./AERO-GNN/main.py --model gt --dataset chameleon --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset squirrel --dr 0.0001 --dropout 0.5 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset actor --dr 0.01 --dropout 0.5 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset twitch --dr 0.0005 --dropout 0.8 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3
python ./AERO-GNN/main.py --model gt --dataset texas --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset wisconsin --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset cornell --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 --hid-dim 8 --num-heads 8 --num-layers 2

python ./AERO-GNN/main.py --model gt --dataset chameleon-filtered --dr 0.005 --dropout 0.7 --lambd-l2 0.0  --hid-dim 8 --num-heads 8 --num-layers 3 
python ./AERO-GNN/main.py --model gt --dataset squirrel-filtered --dr 0.01 --dropout 0.8 --lambd-l2 0.0005  --hid-dim 8 --num-heads 8 --num-layers 2 