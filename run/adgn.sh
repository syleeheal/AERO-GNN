python ./AERO-GNN/main.py --model adgn --dataset cora --iterations 4 --alpha 0.1 --lambd 1.0 --dr 0.0005 --dr1 0.0005 --dropout 0.5
python ./AERO-GNN/main.py --model adgn --dataset citeseer --iterations 16 --alpha 0.01 --lambd 0.1 --dr 0.001 --dr-prop 0.0005 --dropout 0.3
python ./AERO-GNN/main.py --model adgn --dataset pubmed --iterations 8 --alpha 0.1 --lambd 0.001 --dr 0.001 --dr-prop 0.005 --dropout 0.5 
python ./AERO-GNN/main.py --model adgn --dataset wiki --iterations 16 --alpha 0.1 --lambd 0.01 --dr 0.0005 --dr-prop 0.0005 --dropout 0.7 
python ./AERO-GNN/main.py --model adgn --dataset photo --iterations 8 --alpha 0.1 --lambd 0.001 --dr 5e-06 --dr-prop 0.0005 --dropout 0.7 --split sparse
python ./AERO-GNN/main.py --model adgn --dataset computers --iterations 8 --alpha 0.1 --lambd 1 --dr 1e-06 --dr-prop 0.0001 --dropout 0.5 --split sparse

python ./AERO-GNN/main.py --model adgn --dataset chameleon --iterations 16 --alpha 1.0 --lambd 0.0001 --dr 1e-06 --dr-prop 0.0005 --dropout 0.7
python ./AERO-GNN/main.py --model adgn --dataset squirrel --iterations 32 --alpha 1.0 --lambd 0.01 --dr 0.0005 --dr-prop 0.0001 --dropout 0.7
python ./AERO-GNN/main.py --model adgn --dataset actor --iterations 32 --alpha 1.0 --lambd 0.0001 --dr 5e-06 --dr-prop 0.0005 --dropout 0.5
python ./AERO-GNN/main.py --model adgn --dataset texas --iterations 2 --alpha 0.1 --lambd 0.01 --dr 5e-06 --dr-prop 0.0001 --dropout 0.3 &
python ./AERO-GNN/main.py --model adgn --dataset wisconsin --iterations 2 --alpha 1.0 --lambd 0.0001 --dr 0.0005 --dr-prop 0.0005 --dropout 0.3
python ./AERO-GNN/main.py --model adgn --dataset cornell --iterations 32 --alpha 0.001 --lambd 1.0 --dr 1e-05 --dr-prop 0.0001 --dropout 0.3


python ./AERO-GNN/main.py --model adgn --dataset chameleon-filtered --iterations 16 --alpha 1.0 --lambd 0.0001 --dr 0.0001 --dr-prop 0.005 --dropout 0.7
python ./AERO-GNN/main.py --model adgn --dataset squirrel-filtered --iterations 32 --alpha 1.0 --lambd 0.1 --dr 0.0001 --dr-prop 0.0005 --dropout 0.5 
