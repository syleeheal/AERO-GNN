python ./AERO-GNN/main.py --model  gcn --dataset cora --dr 0.001 --dropout 0.8 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model  gcn --dataset citeseer --dr 0.0005 --dropout 0.5 --lambd-l2 0.0005 
python ./AERO-GNN/main.py --model  gcn --dataset pubmed --dr 0.001 --dropout 0.7 --lambd-l2 0.0
python ./AERO-GNN/main.py --model  gcn --dataset wiki --dr 0.0001 --dropout 0.6 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model  gcn --dataset photo --dr 0.0001 --dropout 0.5 --split sparse 
python ./AERO-GNN/main.py --model  gcn --dataset computers --dr 0.0001 --dropout 0.5 --split sparse

python ./AERO-GNN/main.py --model  gcn --dataset chameleon --dr 0.0001 --dropout 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model  gcn --dataset squirrel --dr 0.0001 --dropout 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model  gcn --dataset actor --dr 0.001 --dropout 0.7 --lambd-l2 0.0005 
python ./AERO-GNN/main.py --model  gcn --dataset texas --dr 0.0005 --dropout 0.5 --lambd-l2 0.0005 
python ./AERO-GNN/main.py --model  gcn --dataset wisconsin --dr 0.0005 --dropout 0.6 
python ./AERO-GNN/main.py --model  gcn --dataset cornell --dr 0.001 --dropout 0.5

python ./AERO-GNN/main.py --model  gcn --dataset chameleon-filtered --dr 0.005 --dropout 0.6 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model  gcn --dataset squirrel-filtered --dr 0.0005 --dropout 0.5 --lambd-l2 0.0005 
