python ./AERO-GNN/main.py --model gprgnn --dataset cora --dr 0.01 --dropout 0.8  --lambd-l2 0.0005 --alpha 0.3 --iterations 8 
python ./AERO-GNN/main.py --model gprgnn --dataset citeseer  --dr 0.005 --dropout 0.7  --lambd-l2 0.0005 --alpha 0.5  --iterations 8
python ./AERO-GNN/main.py --model gprgnn --dataset pubmed  --dr 0.01 --dropout 0.6  --lambd-l2 0.0005 --alpha 0.5  --iterations 16
python ./AERO-GNN/main.py --model gprgnn --dataset wiki  --dr 0.0005 --dropout 0.6  --lambd-l2 0.0005 --alpha 0.9  --iterations 32
python ./AERO-GNN/main.py --model gprgnn --dataset photo  --dr 0.0001 --dropout 0.7  --alpha 0.9 --split sparse  --iterations 4
python ./AERO-GNN/main.py --model gprgnn --dataset computers  --dr 0.0001 --dropout 0.6 --alpha 0.1  --split sparse  --iterations 4

python ./AERO-GNN/main.py --model gprgnn --dataset chameleon --dr 0.0001 --dropout 0.6  --lambd-l2 0.0 --alpha 0.1  --iterations 16
python ./AERO-GNN/main.py --model gprgnn --dataset squirrel --dr 0.0001 --dropout 0.7  --lambd-l2 0.0 --alpha 0.3 --iterations 16 
python ./AERO-GNN/main.py --model gprgnn --dataset actor --dr 0.0001 --dropout 0.7  --lambd-l2 0.0005 --alpha 0.5 --iterations 4
python ./AERO-GNN/main.py --model gprgnn --dataset texas --dr 0.0001 --dropout 0.8  --lambd-l2 0.0005 --alpha 0.9 --iterations 4
python ./AERO-GNN/main.py --model gprgnn --dataset wisconsin --dr 0.01 --dropout 0.8 --alpha 0.1 --iterations 8
python ./AERO-GNN/main.py --model gprgnn --dataset cornell --dr 0.01 --dropout 0.5 --alpha 0.1 --iterations 4 --lambd-l2 0.0005

python ./AERO-GNN/main.py --model gprgnn --dataset chameleon-filtered --dr 0.005 --dropout 0.7  --lambd-l2 0.0005 --alpha 0.9  --iterations 10
python ./AERO-GNN/main.py --model gprgnn --dataset squirrel-filtered --dr 0.01 --dropout 0.7  --lambd-l2 0.0005 --alpha 0.3 --iterations 10