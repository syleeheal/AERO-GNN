python ./AERO-GNN/main.py --model appnp --dataset cora --dr 0.0001 --dropout 0.7  --lambd-l2 0.0005 --alpha 0.1 
python ./AERO-GNN/main.py --model appnp --dataset citeseer  --dr 0.001 --dropout 0.5  --lambd-l2 0.0005 --alpha 0.1  
python ./AERO-GNN/main.py --model appnp --dataset pubmed  --dr 0.001 --dropout 0.5  --lambd-l2 0.0 --alpha 0.1  
python ./AERO-GNN/main.py --model appnp --dataset wiki  --dr 0.0001 --dropout 0.6  --lambd-l2 0.0 --alpha 0.1  
python ./AERO-GNN/main.py --model appnp --dataset photo  --dr 0.0001 --dropout 0.6  --alpha 0.3  --split sparse 
python ./AERO-GNN/main.py --model appnp --dataset computers  --dr 0.0001 --dropout 0.5  --alpha 0.3 --split sparse 

python ./AERO-GNN/main.py --model appnp --dataset chameleon --dr 0.0005 --dropout 0.7  --lambd-l2 0.0 --alpha 0.1 
python ./AERO-GNN/main.py --model appnp --dataset squirrel --dr 0.0001 --dropout 0.6  --lambd-l2 0.0 --alpha 0.1  
python ./AERO-GNN/main.py --model appnp --dataset actor --dr 0.0001 --dropout 0.8  --lambd-l2 0.0 --alpha 0.9  
python ./AERO-GNN/main.py --model appnp --dataset texas --dr 0.0005 --dropout 0.5  --lambd-l2 0.0 --alpha 0.9 
python ./AERO-GNN/main.py --model appnp --dataset wisconsin --dr 0.0001 --dropout 0.5  --alpha 0.9
python ./AERO-GNN/main.py --model appnp --dataset cornell --dr 0.0005 --dropout 0.5  --alpha 0.9 

python ./AERO-GNN/main.py --model appnp --dataset chameleon-filtered --dr 0.0005 --dropout 0.8 --lambd-l2 0.0005 --alpha 0.9 
python ./AERO-GNN/main.py --model appnp --dataset squirrel-filtered --dr 0.001 --dropout 0.8 --lambd-l2 0.0005 --alpha 0.9
