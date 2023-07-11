python ./AERO-GNN/main.py --model aero --dataset cora --iterations 32 --dr 0.01 --dr-prop 0.01 --dropout 0.8 --add-dropout 1 --lambd 0.5 
python ./AERO-GNN/main.py --model aero --dataset citeseer --iterations 32 --dr 0.04 --dr-prop 0.01 --dropout 0.7 --add-dropout 1 --lambd 0.25 
python ./AERO-GNN/main.py --model aero --dataset pubmed --iterations 32 --dr 0.02 --dr-prop 0.01 --dropout 0.8 --add-dropout 0 --lambd 0.25 
python ./AERO-GNN/main.py --model aero --dataset wiki --iterations 16 --dr 0.01 --dr-prop 0.005 --dropout 0.6 --add-dropout 1 --lambd 1.0
python ./AERO-GNN/main.py --model aero --dataset photo --iterations 32 --dr 0.0001 --dr-prop 0.0005 --dropout 0.6 --add-dropout 1 --lambd 1.0 --split sparse 
python ./AERO-GNN/main.py --model aero --dataset computers --iterations 32 --dr 0.0005 --dr-prop 0.0001 --dropout 0.8 --add-dropout 0 --lambd 1.0  --split sparse

python ./AERO-GNN/main.py --model aero --dataset chameleon --iterations 32 --dr 0.0001 --dr-prop 0.0001 --dropout 0.7 --add-dropout 0 --lambd 1.0 --num-layers 2
python ./AERO-GNN/main.py --model aero --dataset squirrel --iterations 16 --dr 0.0001 --dr-prop 0.0001 --dropout 0.6 --add-dropout 0 --lambd 1.0 --num-layers 2
python ./AERO-GNN/main.py --model aero --dataset actor --iterations 4 --dr 0.02 --dr-prop 0.01 --dropout 0.5 --add-dropout 1  --lambd 0.25 --num-layers 2 
python ./AERO-GNN/main.py --model aero --dataset texas --iterations 8 --dr 0.005 --dr-prop 0.005 --dropout 0.5 --add-dropout 1  --lambd 0.25 --num-layers 2 
python ./AERO-GNN/main.py --model aero --dataset wisconsin  --iterations 4 --dr 0.001 --dr-prop 0.0005 --dropout 0.5 --add-dropout 0 --lambd 1 --num-layers 2 
python ./AERO-GNN/main.py --model aero --dataset cornell --iterations 4 --dr 0.001 --dr-prop 0.01 --dropout 0.5 --add-dropout 0 --lambd 0.25 --num-layers 2

python ./AERO-GNN/main.py --model aero --dataset chameleon-filtered --iterations 4 --dr 0.001 --dr-prop 0.01 --dropout 0.8 --add-dropout 1 --lambd 0.5 --num-layers 2 
python ./AERO-GNN/main.py --model aero --dataset squirrel-filtered --iterations 32 --dr 0.001 --dr-prop 0.0001 --dropout 0.8 --add-dropout 1 --lambd 1.0 --num-layers 2 
