# import testGspan
import GroupGraphic as GG
import numpy as np
import testGspan as tG
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as py

n = 100
f = 20

X = np.zeros([100,20,10])
prec = np.zeros([20,20,10])


for i in range(10):
    X1,prec1 = GG.createToyData_spd(ranNum = i+100)
    X[:,:,i] = X1
    prec[:,:,i] = prec1

    s = "groudtruth"+str(i)+".jpg"
    py.imshow(prec1)
    py.savefig(groudtruth)

def iteration(group):
    List = []
    for i in range(10):

        # createToyData
        X1 = X[:,:,i]
        prec1 = prec[:,:,i]

        rho = 0.01
        alpha = 0.01
        # GroupGraphicLasso to get all vertex and edges list

        List = GG.gSpanList(X1,i,List,group,rho,alpha)

    # save list to database.txt
    GG.saveDatabase(List)

    # gspan
    orig_stdout = sys.stdout
    f = file('result.txt', 'w')
    sys.stdout = f

    tG.Gspan(0.5)

    sys.stdout = orig_stdout
    f.close()

    # gspan to group
    graph_c = GG.transforGroupInfo(r"result.txt")
    group,truegroups = GG.transformat(graph_c)

    # embedding map
    subgraph = GG.embedding(group)

    orig_stdout = sys.stdout
    f = open('group.txt', 'a')
    sys.stdout = f
    print "******************************"
    print truegroups
    sys.stdout = orig_stdout
    f.close()

    return truegroups

group = None
for i in range(5):
    print i
    group = iteration(group)
