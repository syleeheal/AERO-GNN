python ./AERO-GNN/main.py --model fagcn --dataset cora --hid-dim 16 --iterations 4 --alpha 0.2 --dr 1e-3 --dropout 0.6  --lr 0.01 
python ./AERO-GNN/main.py --model fagcn --dataset citeseer --hid-dim 16 --iterations 4 --alpha 0.3 --dr 1e-3 --dropout 0.6  --lr 0.01  
python ./AERO-GNN/main.py --model fagcn --dataset pubmed --hid-dim 16 --iterations 8 --alpha 0.3 --dr 1e-3 --dropout 0.6 --lr 0.01  
python ./AERO-GNN/main.py --model fagcn --dataset wiki --hid-dim 16 --iterations 3 --alpha 0.1 --dr 5e-05 --dropout 0.4 --lr 0.005 --lambd-l2 0.0005 
python ./AERO-GNN/main.py --model fagcn --dataset photo --hid-dim 16 --iterations 8  --alpha 1.0 --dr 0.0001 --dropout 0.5 --split sparse 
python ./AERO-GNN/main.py --model fagcn --dataset computers --hid-dim 16 --iterations 7  --alpha 0.9 --dr 0.0001 --dropout 0.4 --split sparse

python ./AERO-GNN/main.py --model fagcn --dataset chameleon --hid-dim 32 --iterations 2 --alpha 0.4 --dr 5e-5 --dropout 0.5 --lr 0.01  
python ./AERO-GNN/main.py --model fagcn --dataset squirrel --hid-dim 32 --iterations 2 --alpha 0.3 --dr 5e-5 --dropout 0.5  --lr 0.01 
python ./AERO-GNN/main.py --model fagcn --dataset actor --hid-dim 32 --iterations 6 --alpha 0.9 --dr 0.005 --dropout 0.5 --lambd-l2 0.0005 --lr 0.005
python ./AERO-GNN/main.py --model fagcn --dataset texas --hid-dim 32 --iterations 7 --alpha 0.2 --dr 0.0001 --dropout 0.5 --lambd-l2 0.0 --lr 0.005
python ./AERO-GNN/main.py --model fagcn --dataset wisconsin --hid-dim 32 --iterations 7 --alpha 0.2 --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 --lr 0.005
python ./AERO-GNN/main.py --model fagcn --dataset cornell --hid-dim 32 --iterations 2 --alpha 1.0 --dr 0.0001 --dropout 0.6 --lr 0.005

python ./AERO-GNN/main.py --model fagcn --dataset chameleon-filtered --hid-dim 32 --lr 0.005 --iterations 5 --alpha 0.1 --dr 0.01 --dropout 0.4  
python ./AERO-GNN/main.py --model fagcn --dataset squirrel-filtered --hid-dim 32 --lr 0.005 --iterations 3 --alpha 0.2 --dr 0.01 --dropout 0.6
