python ./AERO-GNN/main.py --model dagnn --dataset cora --iterations 10 --dr 0.005 --dropout 0.8 
python ./AERO-GNN/main.py --model dagnn --dataset citeseer --iterations 10  --dr 0.02 --dropout 0.5 
python ./AERO-GNN/main.py --model dagnn --dataset pubmed --iterations 20  --dr 0.005 --dropout 0.8 
python ./AERO-GNN/main.py --model dagnn --dataset wiki --iterations 5 --dr 0.0005 --dropout 0.5 --lambd-l2 0.0005 
python ./AERO-GNN/main.py --model dagnn --dataset photo --iterations 10 --dr 5e-05 --dropout 0.5 --split sparse 
python ./AERO-GNN/main.py --model dagnn --dataset computers --iterations 5 --dr 5e-05 --dropout 0.5  --split sparse 

python ./AERO-GNN/main.py --model dagnn --dataset chameleon --iterations 5 --dr 5e-05 --dropout 0.7 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model dagnn --dataset squirrel --iterations 20 --dr 0.005 --dropout 0.6 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model dagnn --dataset actor  --iterations 5 --dr 0.005 --dropout 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model dagnn --dataset texas  --iterations 5 --dr 0.0 --dropout 0.5 --lambd-l2 0.0005
python ./AERO-GNN/main.py --model dagnn --dataset wisconsin  --iterations 5 --dr 0.005 --dropout 0.6 
python ./AERO-GNN/main.py --model dagnn --dataset cornell  --iterations 5 --dr 0.005 --dropout 0.5 

python ./AERO-GNN/main.py --model dagnn --dataset chameleon-filtered --iterations 5 --dr 0.02 --dropout 0.5 --lambd-l2 0.0 
python ./AERO-GNN/main.py --model dagnn --dataset squirrel-filtered --iterations 5 --dr 0.02 --dropout 0.8 --lambd-l2 0.0005 
