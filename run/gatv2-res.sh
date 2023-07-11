python ./AERO-GNN/main.py --model gat-res --dataset cora --hid-dim 8 --num-heads 8 --dr 0.0005 --dropout 0.8 --num-layers 2 --alpha 0.4 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset citeseer --hid-dim 8 --num-heads 8 --dr 0.0001 --dropout 0.5 --num-layers 2 --alpha 0.3 --lambd-l2 0.0005
python ./AERO-GNN/main.py --model gat-res --dataset pubmed --hid-dim 8 --num-heads 8 --dr 0.001 --dropout 0.7 --num-layers 2 --alpha 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset wiki  --hid-dim 8 --num-heads 8 --num-layers 2 --alpha 0.5 --dr 0.0005 --dropout 0.6 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset photo --hid-dim 8 --num-heads 8 --num-layers 4 --alpha 0.5 --dr 0.0001 --dropout 0.7 --lambd-l2 0.0  --split sparse 
python ./AERO-GNN/main.py --model gat-res --dataset computers --hid-dim 8 --num-heads 8 --num-layers 4 --alpha 0.4 --dr 0.0001 --dropout 0.6 --lambd-l2 0.0  --split sparse 

python ./AERO-GNN/main.py --model gat-res --dataset chameleon  --hid-dim 8 --num-heads 8  --num-layers 32 --alpha 0.2 --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset squirrel --hid-dim 8 --num-heads 8 --num-layers 16 --alpha 0.5 --dr 0.0001 --dropout 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset actor  --hid-dim 8 --num-heads 8  --num-layers 16 --alpha 0.2 --dr 0.0001 --dropout 0.7 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset cornell --hid-dim 8 --num-heads 8 --dr 0.0005 --dropout 0.7 --num-layers 2 --alpha 0.3 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model gat-res --dataset wisconsin --hid-dim 8 --num-heads 8 --dr 0.0005 --dropout 0.7 --num-layers 2 --alpha 0.5 --lambd-l2 0.0
python ./AERO-GNN/main.py --model gat-res --dataset texas --hid-dim 8 --num-heads 8 --dr 0.001 --dropout 0.7 --num-layers 8 --alpha 0.5 --lambd-l2 0.0 

python ./AERO-GNN/main.py --model gat-res --dataset chameleon-filtered  --hid-dim 8 --num-heads 8  --num-layers 2 --alpha 0.2 --dr 0.001 --dropout 0.7 --lambd-l2 0.0005
python ./AERO-GNN/main.py --model gat-res --dataset squirrel-filtered --hid-dim 8 --num-heads 8 --num-layers 4 --alpha 0.2 --dr 0.0005 --dropout 0.6 --lambd-l2 0.0005