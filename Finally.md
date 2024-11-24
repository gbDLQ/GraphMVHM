```bash
python main.py -dataset cora -layer gcn -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2
python main.py -dataset cora -layer gat -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2
python main.py -dataset cora -layer sage -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2

python main.py -dataset CiteSeer -layer gcn -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2
python main.py -dataset CiteSeer -layer gat -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2
python main.py -dataset CiteSeer -layer sage -imbalance_ratio 100 -small 50 -cl_1 0.1 -cl_2 0.3 -theta 0.4 -train_ratio 0.6 -test_ratio 0.2

python main.py -dataset Computers -layer gcn -imbalance_ratio 20 -train_ratio 0.1 -test_ratio 0.8 -cl_1 0.7 -cl_2 0.25 -theta 0.9 -small 50
python main.py -dataset Computers -layer gat -imbalance_ratio 20 -train_ratio 0.1 -test_ratio 0.8 -cl_1 0.7 -cl_2 0.25 -theta 0.9 -small 50
python main.py -dataset Computers -layer sage -imbalance_ratio 20 -train_ratio 0.1 -test_ratio 0.8 -cl_1 0.7 -cl_2 0.25 -theta 0.9 -small 50
```