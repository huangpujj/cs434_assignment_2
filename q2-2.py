import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
# get data from files
result = np.loadtxt(open("./data/knn_train.csv","rb"),delimiter=",",skiprows=0)
x, y =  result[:, 1:-1], result[:, 0]
weight = np.zeros((x.shape[1],1))
result_test = np.loadtxt(open("./data/knn_test.csv","rb"),delimiter=",",skiprows=0)
x_test, y_test =  result_test[:, 1:-1], result_test[:, 0]

x = x.tolist()
y =y.tolist()
x_test = x_test.tolist()
y_test = y_test.tolist()

class Node:
    def __init__(self, entropy):
        self.type = 1 # type 1: node, type 2: leaf node
        self.entropy = entropy #熵
        self.infoGain = 0  #信息增益
        self.left = None
        self.right = None
        self.value = 0
        self.split_id = -1
        self.split_value = 0

def Calculate_Entropy(x,y):
    entropy = 0
    size = len(x)
    count = len([row for row in y if row == 1])
    if(count != size and count != 0):
        p = count / float(size)
        entropy = - p * math.log(p, 2) - (1 - p) * math.log((1 - p), 2)
    return entropy

def split(node,x,y,depth,n):

    if(depth == 0):
        LeafNode(node, x, y)
        return
    if(len(x) < 5):
        LeafNode(node, x, y)
        return
    if Calculate_Entropy(x,y) == 0:
        LeafNode(node, x, y)

    left_x,left_y, right_x,right_y = split_best(node, x, y)
    split(node.left, left_x,left_y, depth - 1, 5)
    split(node.right, right_x,right_y, depth - 1, 5)

def LeafNode(node, x, y):
    node.type = 2
    node.left_x = None
    node.left_y = None
    node.right_y = None
    node.right_x = None

    count = y.count(1)

    invCount = len(x) - count
    if count > invCount:
        node.value = 1
    else:
        node.value = -1

def split_best(node,x,y):
    best = {
        'leftEntropy': 0,
        'rightEntropy': 0,
        'leftData_x': [],
        'leftData_y': [],
        'rightData_x': [],
        'rightData_y': [],
    }

    for i, row in enumerate(x):
        for j, val in enumerate(row):

            # split and get entropy, information gain
            left_x, left_y,right_x,right_y = test_split(x,y, j, val)

            leftEntropy = Calculate_Entropy(left_x, left_y)
            rightEntropy = Calculate_Entropy(right_x, right_y)
            leftProb = len(left_x) / float(len(x))
            infoGain = node.entropy - leftProb * leftEntropy - (1 - leftProb) * rightEntropy

            # if information gain is better, record it
            if (infoGain > node.infoGain):
                #print("here")
                best['leftEntropy'] = leftEntropy
                best['rightEntropy'] = rightEntropy
                best['leftData_x'] = left_x
                best['leftData_y'] = left_y
                best['rightData_x'] = right_x
                best['rightData_y'] = right_y
                node.split_id = j
                node.split_value = val
                node.infoGain = infoGain
    #print(best['leftEntropy'])
    #print(best['rightEntropy'])
    # Create nodes for left and right
    node.left = Node(best['leftEntropy'])
    node.right = Node(best['rightEntropy'])
    return best['leftData_x'],best['leftData_y'], best['rightData_x'],best['rightData_y']

def test_split(x,y, idx, val):

    left_x = []
    left_y = []
    right_y = []
    right_x = []

    for i, row in enumerate(x):
        if row[idx] < val:
            left_x.append(row)
            left_y.append(y[i])
        else:
            right_x.append(row)
            right_y.append(y[i])

    return left_x, left_y, right_x,right_y

def calculate_Error(x,y, root):
    correct = 0
    for i, row in enumerate(x):
        node = root
        while node.type != 2:
            idx = node.split_id
            val = node.split_value
            if row[idx] < val:
                node = node.left
            else:
                node = node.right
        predictY = node.value

        if predictY == y[i]:
            correct += 1

    errorRate = 1 - correct / float(len(x))
    return errorRate

def main(train_x,train_y,test_x,test_y):
    d = range(1,7)
    trainErrorRate = []
    testErrorRate = []
    for i in d:
        root = Node(Calculate_Entropy(x,y))
        split(root,x,y,i,5)
        trainErrorRate.append(calculate_Error(x,y, root))
        testErrorRate.append(calculate_Error(x_test,y_test, root))

    print("\n-------> Training Error Rate: ")
    print(trainErrorRate)
    print("\n-------> Testing Error Rate: ")
    print(testErrorRate)

    plt.plot(d, trainErrorRate, marker='o', label="training error rate")
    plt.plot(d, testErrorRate, marker='o', label="testing error rate")
    plt.xlabel("# of d")
    plt.ylabel("Error rate")
    plt.legend()
    plt.savefig("q2_2.png")

if __name__ == "__main__":
	main(x,y,x_test,y_test)
