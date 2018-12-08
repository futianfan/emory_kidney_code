python3 src/preprocess.py 

python3 src/logistic_regression.py
### Logistic regression AUC:0.8221 -> 0.805 (0.001597)

python3 src/train.py 
### Full-Connect NN AUC:0.834,   1.4 seconds/epoch. 10 epoch to converge

python3 src/decision_tree_sklearn.py 
### decision tree:  AUC:0.810 -> 0.795;  generalize poorly (0.825 train accuracy)
dot -Tpng ./data/tree.dot -o ./data/tree.png


python3 src/decision_tree_NN.py 
## NN + decision tree 0.834  1.94 seconds/epoch  73 epoch to converge


python3 src/plot_bar_accuracy.py

python3 src/plot_bar_runtime.py

python3 src/plot_curve.py


