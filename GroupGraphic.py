import cvxpy as cvx
import numpy as np
from scipy import linalg
from sklearn.datasets import make_spd_matrix as spd
from sklearn.covariance import GraphLassoCV
import csv
import re


"""
GroupGraphicLasso with cvxpy

min -log(det(Phi)) + trace(S*Phi) + rho*sum(group,2) +alpha*(Phi,1)

"""

# def GroupGraphicLasso(X, rho, alpha, groups):
#     """
#         S is the empirical covariance matrix.
#     """
#     S = np.cov(X.T)
#     assert S.shape[0] == S.shape[1]
#     n = S.shape[0]
#     Phi = cvx.Semidef(n)
#     group_pennal=[]
#     for group in groups:
#         group_pennal.append(cvx.norm(Phi[group,group],"fro"))
#     obj = cvx.Minimize(-(cvx.log_det(Phi) - cvx.trace(S*Phi) - rho*sum(group_pennal) - alpha*cvx.norm(Phi,1)))
#     constraints = []
#     prob = cvx.Problem(obj,constraints)
#     prob.solve(solver=cvx.SCS, eps=1e-5)
#     return Phi.value

def nonGroupIndex(group,n):
    return [i for i in range(n) if i not in group]

def GroupGraphicLasso(X, rho, alpha ,groups):
    """
        S is the empirical covariance matrix.
    """
    S = np.cov(X.T)
    assert S.shape[0] == S.shape[1], "Matrix must be square"
    n = S.shape[0]

    #Phi = cvx.Variable(n, n)
    Phi = cvx.Semidef(n)
    rest = cvx.Semidef(n)
    group_pennal=[]
    non_one_pennal=[]
    A=set()
    for group in groups:
        group_pennal.append(cvx.norm(Phi[group,group],"fro"))
        non_index=nonGroupIndex(group, n)
        non_one_pennal.append(cvx.norm(Phi[group][:,non_index],1))
        A.update(set(group))
    non_block = [i for i in range(n) if i not in A]
    if len(non_block) > 0:
        block_onenorm = cvx.norm(Phi[:,non_block],1)
        obj = cvx.Minimize(-(cvx.log_det(Phi) - cvx.trace(S*Phi) - rho*sum(group_pennal) - alpha * sum(non_one_pennal)-
                        alpha * block_onenorm))
    else:
        obj = cvx.Minimize(-(cvx.log_det(Phi) - cvx.trace(S*Phi) - rho*sum(group_pennal) - alpha * sum(non_one_pennal)))
    constraints = []

    prob = cvx.Problem(obj,constraints)
    prob.solve(solver=cvx.SCS, eps=1e-5)
    return Phi.value




def GraphicLasso(X):
    model = GraphLassoCV()
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_
    return prec_


"""
Simulate toy data

addOneGroup: simulate SPD matrix with group index.
stick: Stick small pieces SPD matrix into large matrix.
createToyData_spd: create a group SPD matrix. ranNum: random seed. size: # of observation, # of feature.
"""

def addOneGroup(one_group):
    return spd(len(one_group))

def stick(group,groupMatrix,X):
    for i in xrange(groupMatrix.shape[0]):
        for j in xrange(groupMatrix.shape[1]):
            X[group[i],group[j]] = groupMatrix[i,j]

def createToyData_spd(ranNum=20,size=(100,20),
                 groups=[np.array(range(10)),np.array(range(10,20))]):
    prng = np.random.RandomState(ranNum)
    n_samples,n_features=size[0],size[1]
    prec = np.zeros((n_features,n_features))
    for g in groups:
        onegroup=addOneGroup(g)
        stick(g,onegroup,prec)
    for i in range(n_features):
        prec[i,i] += 1.
    cov = linalg.inv(prec)
    X = prng.multivariate_normal(np.zeros(size[1]), cov, size=size[0])
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X,prec


"""
Create database with Createdata.ipynb
"""
def createList(alpha,prec_):
    NList = []
    EList = []
    Node = []
    numN = 0
    for i in range(prec_.shape[0]):
        for j in range(i+1,prec_.shape[0]):
            if prec_[i,j]>alpha:
                if i not in Node:
                    Node.append(i)
                    Name = "v"+" "+str(numN)+" "+str(i+1)
                    NList.append(Name)
                    numN = numN+1
                if j not in Node:
                    Node.append(j)
                    Name = "v"+" "+str(numN)+" "+str(j+1)
                    NList.append(Name)
                    numN = numN+1
                EList.append("e"+" "+str(Node.index(i))+" "+str(Node.index(j))+" "+str(i+1)+str(j+1))

    return NList,EList


def gSpanList(X,num,List,group,rho, alpha):
    if group == None:
        prec_= GraphicLasso(X)
    else:
        prec_ = GroupGraphicLasso(X, rho, alpha ,group)

    NList,EList = createList(1e-5,prec_)
    List = List + ["t # "+str(num)]+NList+EList
    return List

def saveDatabase(List):
    database = r"database.txt"
    with open(database,"w") as db:
        for line in List:
            db.write(line+"\n")



"""
gspan:
"""

"""
extract subgraph
cleanMap: extract all frequent embedding group
addTrp: add subgraph
embedding: return all frequent mbedding.
"""

def cleanMap(mapDic,group):
    CleanDic = {}
    for key in mapDic.keys():
        fromL = key.split()[2]
        Flabel = re.findall('\d+', fromL)[0]
        toL = key.split()[4]
        Tlabel = re.findall('\d+', toL)[0]

        e1 = np.array([int(Flabel),int(Tlabel)])
        e2 = np.array([int(Tlabel),int(Flabel)])

        Ekey = str(Flabel)+"#"+str(Tlabel)

        for i in group:
            if np.array_equal(e1,i) or np.array_equal(e2,i):
                graphID = re.findall(r"\bid=[\w]*", mapDic[key])
                CleanDic[Ekey] = graphID

    return CleanDic

def addTrp(CleanDic, group):
    for i in group:
        if len(i)>2:
            i = np.sort(i)
            a = str(i[0])+"#"+str(i[1])
            b = str(i[1])+"#"+str(i[2])
            c = str(i[0])+"#"+str(i[2])

            keyFind = [a,b,c]
            result = [a in CleanDic.keys(),b in CleanDic.keys(),c in CleanDic.keys()]
            s = sum(result)

            if s>1:
                List = []
                key = []
                d = str(i[0])+"#"+str(i[1])+"#"+str(i[2])

                for j in range(3):
                    if result[j]:
                        key.append(keyFind[j])

                for m in CleanDic[key[0]]:
                    if m in CleanDic[key[1]]:
                        List.append(m)

                CleanDic[d] = List
    return CleanDic

def embedding(group):
    mapDic = {}
    for key, val in csv.reader(open("map.csv")):
    	mapDic[key] = val
    CleanDic = cleanMap(mapDic,group)
    subgraph = addTrp(CleanDic, group)

    return subgraph

"""
Transfer gspan result to group
split: split edge label base on vertex
transforGroupInfo: extract group
transformat: transfer list group to array group

"""

def split(x, v_list):
    """split x into a list of two numbers

    """
    for i in range(len(v_list)):
        v_list[i] = str(v_list[i])

    if len(x) == 2:
        return [x[0],x[1]]
    if len(x)==3:
        return [x[:1],x[1:]]
    if len(x)==4:
        if x[0:1] in v_list and x[0:1]  in v_list and (x[0:2] not in v_list or x[2:] not in v_list):
            return [x[0:1], x[1:]]
        if x[0:2] in v_list and x[2:] in v_list and (x[0:1] not in v_list or x[1:] not in v_list):
            return [x[0:2],x[2:]]
        if x[0:2] in v_list and x[2:] in v_list and x[0:1] in v_list and x[1:] in v_list:
            return [x[0:2],x[2:]],[x[0:1], x[1:]]
    if len(x)==5:
        return [x[:2],x[2:]]
    if len(x)==6:
        return [x[:3],x[3:]]

def transforGroupInfo(groupfile):
    graph_c=[]
    subgraph=[]
    v_list=[]
    with open(groupfile,'r') as gf:
        for line in gf.readlines():
            line_list = line.split(' ')
            #print line_list
            if line[0] == 't':
                #print 't'
                if len(subgraph) > 0:
                    graph_c.append(subgraph)
                v_list=[]
                subgraph = []
            elif line[0]== 'e':
                #print x[3][:-1]
                subgraph.append(split(line_list[3][:-1],v_list))
            elif line[0] == 'v' :
                v_list.append(int(line_list[2][:-1]))
            else :
                continue
    return graph_c

def transformat(graph_c):
    groups=[]
    truegroups=[]
    for g in graph_c:
        subg=set()
        subg1 = set()
        for conpon in g:
            for v in conpon:
                subg.add(int(v))
                subg1.add(int(v)-1)
        groups.append(list(subg))
        truegroups.append(list(subg1))
    #print groups
    groups = [np.array(g) for g in groups]
    truegroups = [np.array(g) for g in truegroups]

    return groups,truegroups
