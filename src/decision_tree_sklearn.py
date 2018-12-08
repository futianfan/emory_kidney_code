from sklearn import tree
import numpy as np 
from sklearn.metrics import roc_auc_score
from os import system
from graphviz import Source
from time import time

clf = tree.DecisionTreeClassifier(max_depth = 6)
dataFolder = '/Users/futianfan/Downloads/Gatech_Courses/emory_kidney/data'
train_data = np.load('{}/train.npy'.format(dataFolder))
test_data = np.load('{}/test.npy'.format(dataFolder))
feature_name = open('{}/feature_name_for_all_data'.format(dataFolder), 'r').readline().rstrip().split()
dotfile = '{}/tree.dot'.format(dataFolder)
feature_map_file = open('{}/feature_map'.format(dataFolder), 'r').readlines()
feature_map = {line.split()[0]:line.split()[1] for line in feature_map_file}


train_label = train_data[:,0]
trainData = train_data[:,1:]
test_label = test_data[:,0]
testData = test_data[:,1:]
class_name = np.unique(['success', 'failure'])

############## training  
t1 = time()
clf = clf.fit(trainData, train_label)
def evaluate(clf, Data, Label):
	prediction = clf.predict_proba(Data)
	prediction = prediction[:,1]
	prediction = list(prediction)
	label = list(Label)
	return str(roc_auc_score(label, prediction))[:5]

print('train accuracy: {}'.format(evaluate(clf, trainData, train_label)))
print('test accuracy: {}'.format(evaluate(clf, testData, test_label)))
print('cost {} seconds'.format(time() - t1))

tree.export_graphviz(clf, out_file = dotfile, feature_names = feature_name[1:], impurity = False, \
	class_names = class_name, \
	proportion = False)

with open(dotfile, 'r') as fin:
	lines = fin.readlines()


leaf_node_assign = clf.apply(trainData)
leaf_node_assign = list(leaf_node_assign)
leaf_node_set = set(leaf_node_assign)
leaf_node_lst = list(leaf_node_set)
leaf_node_lst.sort()
leaf_node_assign = [list(filter(lambda i:leaf_node_assign[i] == leaf_node_indx , list(range(len(leaf_node_assign))))) for leaf_node_indx in leaf_node_lst]

compute_probability = lambda lst: np.sum(train_label[lst]) * 1.0 / len(lst)
leaf_node_prob = list(map(compute_probability, leaf_node_assign))
idxx = 0




tree = clf.tree_
print('num of node in decision tree is {}'.format(len(tree.threshold)))
with open(dotfile, 'w') as fout:
	for i,line in enumerate(lines):
		if i <= 1 or i == len(lines) - 1:
			fout.write(line)
		else:
			if "->" in line:
				fatherNode = int(line.split()[0])
				sonNode = int(line.split()[2])
				if sonNode - fatherNode == 1:
					line = str(fatherNode) + ' -> ' + str(sonNode) \
						+ ' [headlabel="T"] ;\n' 
				else:
					line = str(fatherNode) + ' -> ' + str(sonNode) \
						+ ' [headlabel="F"] ;\n'
			else:
				stt = line.index("samples = ")
				#stt = stt - 2 if line[stt-2:stt] == '\\n' else stt
				endn = line.index("value = ")
				endn = line[endn:].index(']') + endn			
				endn = endn + 2 if line[endn+1:endn+3] == '\\n' else endn
				line = line[:stt] + line[endn+1:] 
				##print(line)
				if "__is__" in line:  ### categorical
					line = line.split()
					line = line[0] + ' ' + line[1] + '"] ;\n'
					line = line.replace('__', ' ')
				elif 'label="class' in line:
					if 'success' in line:
						assert 'class = success' in line 
						line = line.replace('class = success', 'failure')
					elif 'failure' in line: 
						assert 'class = failure' in line 
						line = line.replace('class = failure', 'success')
					stt = line.index('"')
					stt = stt + line[stt+1:].index('"') + 1
					line = line[:stt] + '\\n' + 'prob=' + str(leaf_node_prob[idxx])[:5] + line[stt:]
					idxx += 1
				else:  ### noncategorical 
					stt = line.index('label="') + 7 
					endn = line[stt:].index(' ') + stt 
					k = line[stt:endn]
					assert k in feature_map
					stt = line.index('<=') + 3
					endn = line.index('\\n')
					threshold = float(line[stt:endn])
					if threshold == 0.5:
						raw_threshold = float(feature_map[k].split(':')[1])
					elif threshold == -0.5:
						raw_threshold = float(feature_map[k].split(':')[0])
					elif threshold == 0.0:
						raw_threshold = float(feature_map[k].split(':')[0])
					else:
						assert False 
					line = line[:stt] + ' ' + str(raw_threshold) + ' ' + line[endn:]
				
				if 'label="class' not in line and 'class = ' in line:
					stt = line.index('\\nclass =')
					endn = stt + line[stt:].index('"')
					line = line[:stt] + line[endn:]
			fout.write(line) 
system("dot -Tpng ./data/tree.dot -o ./data/tree.png")



