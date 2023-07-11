python ./AERO-GNN/main.py --model mixhop --dataset wiki  --dr 0.0001 --dropout 0.8  --lambd-l2 0.0005 --iterations 5  
python ./AERO-GNN/main.py --model mixhop --dataset photo --dr 0.0001 --dropout 0.8  --iterations 5 --split sparse 
python ./AERO-GNN/main.py --model mixhop --dataset computers --dr 0.0001 --dropout 0.8  --iterations 5 --split sparse

python ./AERO-GNN/main.py --model mixhop --dataset chameleon --dr 0.0001 --dropout 0.8  --lambd-l2 0.0 --iterations 5 
python ./AERO-GNN/main.py --model mixhop --dataset squirrel --dr 0.0005 --dropout 0.8  --lambd-l2 0.0 --iterations 3 
python ./AERO-GNN/main.py --model mixhop --dataset actor --dr 0.01 --dropout 0.6  --lambd-l2 0.0005 --iterations 5
python ./AERO-GNN/main.py --model mixhop --dataset texas --dr 0.001 --dropout 0.6  --lambd-l2 0.0 --iterations 5  
python ./AERO-GNN/main.py --model mixhop --dataset wisconsin --dr 0.001 --dropout 0.6  --iterations 5 
python ./AERO-GNN/main.py --model mixhop --dataset cornell --dr 0.0001 --dropout 0.7  --iterations 3  

python ./AERO-GNN/main.py --model mixhop --dataset chameleon-filtered --dr 0.01 --dropout 0.7  --lambd-l2 0.0 --iterations 5 
python ./AERO-GNN/main.py --model mixhop --dataset squirrel-filtered --dr 0.01 --dropout 0.8  --lambd-l2 0.0005 --iterations 3 
