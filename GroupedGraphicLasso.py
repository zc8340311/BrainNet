import numpy as np
import scipy as sci
from scipy import linalg, optimize
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.datasets import make_sparse_spd_matrix

MAX_ITER = 200

def group_lasso(X, y, alpha, groups, max_iter=MAX_ITER, rtol=1e-6,
             verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.
    Solves problem of the form:
               .5 * |Xb - y| + n_samples * alpha * Sum(w_j * |b_j|)
    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.
    y : array of shape (n_samples,)
    alpha : float or array
        Amount of penalization to use.
    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.
    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution.
    Returns
    -------
    x : array
        vector of coefficients
    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb
    """

    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))

    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    alpha = alpha * X.shape[0]

    # .. use integer indices for groups ..

    group_labels=groups

    H_groups = [np.dot(X[:, g].T, X[:, g]) for g in group_labels]

    eig = map(linalg.eigh, H_groups)

    Xy = np.dot(X.T, y)
    initial_guess = np.zeros(len(group_labels))

    def f(x, qp2, eigvals, alpha):
        return 1 - np.sum( qp2 / ((x * eigvals + alpha) ** 2))
    def df(x, qp2, eigvals, penalty):
        # .. first derivative ..
        return np.sum((2 * qp2 * eigvals) / ((penalty + x * eigvals) ** 3))

    if X.shape[0] >= X.shape[1]:
        H = np.dot(X.T, X)
    else:
        H = None

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        for i, g in enumerate(group_labels):
            # .. shrinkage operator ..
            eigvals, eigvects = eig[i]
            w_i = w_new.copy()
            w_i[g] = 0.

            if H is not None:
                X_residual = np.dot(H[g], w_i) - Xy[g]
            else:
                X_residual = np.dot(X.T, np.dot(X[:, g], w_i)) - Xy[g]
            qp = np.dot(eigvects.T, X_residual)
            if len(g) < 2:
                # for single groups we know a closed form solution
                w_new[g] = - np.sign(X_residual) * max(abs(X_residual) - alpha, 0)
            else:
                if alpha < linalg.norm(X_residual, 2):
                    initial_guess[i] = optimize.newton(f, initial_guess[i], df, tol=.5,
                                args=(qp ** 2, eigvals, alpha))
                    w_new[g] = - initial_guess[i] * np.dot(eigvects /  (eigvals * initial_guess[i] + alpha), qp)
                else:
                    w_new[g] = 0.


        # .. dual gap ..
        max_inc = linalg.norm(w_old - w_new, np.inf)
        if True: #max_inc < rtol * np.amax(w_new):
            residual = np.dot(X, w_new) - y
            group_norm = alpha * np.sum([linalg.norm(w_new[g], 2)
                         for g in group_labels])
            if H is not None:
                norm_Anu = [linalg.norm(np.dot(H[g], w_new) - Xy[g]) \
                           for g in group_labels]
            else:
                norm_Anu = [linalg.norm(np.dot(H[g], residual)) \
                           for g in group_labels]
            if np.any(norm_Anu > alpha):
                nnu = residual * np.min(alpha / norm_Anu)
            else:
                nnu = residual
            primal_obj =  .5 * np.dot(residual, residual) + group_norm
            dual_obj   = -.5 * np.dot(nnu, nnu) - np.dot(nnu, y)
            dual_gap = primal_obj - dual_obj
            if verbose:
                pass
                #print 'Relative error: %s' % (dual_gap / dual_obj)
            if np.abs(dual_gap / dual_obj) < rtol:
                break

    return w_new

def index(group, index):
    new_g=[]
    p=list(set(range(len(index)+1))-set(index))[0]
    for g in groups:
        temp=[]
        for i in g:
            if i>p:
                temp.append(i-1)
            if i<p:
                temp.append(i)
        new_g.append(temp)
    return map(np.array,new_g)
def glasso(S,rho,groups):
    n,p=S.shape
    maxIter=100
    tol=1e-6


    W=S+ rho * np.eye(p)
    W_old = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_old[i,j]=W[i,j]

    ttttt=False
    for round in range(1,maxIter):
        for j in range(p-1,-1,-1):
            jminus=range(0,p)
            del jminus[j]

            D,V = np.linalg.eig(W[jminus,:][:,jminus])
            x1 = np.dot(V,np.diag(np.sqrt(D)))
            X = np.dot(x1,V.T)   # W_11^(1/2)
            y1 = np.dot(V,np.diag(1./np.sqrt(D)))
            y2 = np.dot(y1,V.T)
            Y = np.dot(y2, S[jminus,j])     # W_11^(-1/2) * s_12

            #interface changed
            indexed_groups=index(groups,jminus)

            b=group_lasso(X, Y, rho, indexed_groups, max_iter=MAX_ITER)
            W[jminus,j] = np.dot(W[jminus,:][:,jminus], b)
            W[j,jminus] = W[jminus,j].T


        if np.linalg.norm(W - W_old,1) < tol:
            print np.linalg.norm(W - W_old,1)
            ttttt = True
            break
        else:
            W_old=W
        if not ttttt :
            print "glasso may not converge"

    return np.linalg.inv(W)
def add_group(data,group):
    '''add group to toy data'''
    for i in group:
        for j in group:
            if i==j:
                pass
            else:
                data[i,j] += 0.25
def set_zero(data):
    '''to visualize clearly, we set all diagonal element to 0'''
    z=np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i!=j:
                z[i][j]=data[i][j]
    return z
def plotnet(connectionM):
    '''plot the connection Matrix, we didnot use this function in this file'''
    plt.figure(figsize=(10,7))
    plt.title("Connection")
    G=nx.Graph()
    elist=[(i,j) for i in range(connectionM.shape[0]) for j in range(connectionM.shape[1]) if connectionM[i,j]==1]
    G.add_edges_from(elist)
    #also can use pos=nx.sprint_layout(G)
    pos=nx.circular_layout(G)
    nx.draw_networkx_nodes(G,pos,node_size=700)
    nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
    nx.draw_networkx_edges(G,pos,edgelist=elist)
    plt.axis('off')
def createToyData(ranNum=20,
                  size=(100,20),
                 groups=[np.array([0,1,2,3,4,5]),np.array([0,1,2]),np.array([3,4,5]),np.array([6,7,8,9])]
                 ):
    '''creat toy test data , parameter:
            ranNum: random state
            size: (num1,num2) num1 mean examples, num2 means features.
            groups: we require groups in a list, each element is a numpy.array() object example:[np.array([0,1,2,3,4,5])]
        this function return :
            X: toy data X
            prec: ground truth precision matrix
    '''
    n_samples = size[0]
    n_features = size[1]
    ##random state
    prng = np.random.RandomState(ranNum)
    prec = np.zeros(shape=(n_features,n_features))
    for i in range(n_features):
        prec[i,i]+=1.

    #groups=[np.array([0,1,2,3,4,5]),np.array([0,1,2]),np.array([3,4,5]),np.array([6,7,8,9]),np.array([6,7]),np.array([8,9])]
    for g in groups:
        add_group(prec,g)
    cov = linalg.inv(prec)
    ##using random state to create X
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X,prec

if __name__=="__main__":
    ##setting groups
    groups=[np.array([0,1,2,3,4,5]),np.array([0,1,2]),np.array([3,4,5]),np.array([6,7,8,9]),np.array([6,7]),np.array([8,9]),
           np.array([10,11,12,13,14,15]),np.array([10,11,12]),np.array([13,14,15]),np.array([16,17,18,19]),np.array([16,17]),
            np.array([18,19]),np.array([6,7,8,9,10,11,12,13,14,15])]
    groups1=[np.array([0,1]),np.array([1,4]),np.array([4,5]),np.array([4,7])]
    ##create toy data
    X,prec=createToyData(groups=groups)
    pennal_list=np.arange(0.001,0.03,0.005)

    ggl_list=[]
    gl_list=[]
    for pennal in pennal_list:
        ###grouped graphic lasso
        S=np.cov(X.T)
        print S.shape
        test=glasso(S,pennal,groups)
        ggl_list.append(test)
        ###graphic lasso
        model = GL(alpha=pennal)
        model.fit(X)
        #cov_ = model.covariance_
        prec_ = model.precision_
        gl_list.append(prec_)

    pennelty_number=len(ggl_list)
    fig,ax = plt.subplots(ncols=pennelty_number,nrows=2)
    for i in range(2):
        for j in range(pennelty_number):
                if i<1:
                    value=set_zero(abs(ggl_list[j]))
                    ax[i][j].imshow(value)
                else:
                    value=set_zero(abs(gl_list[j]))
                    ax[i][j].imshow(value)
    fig.set_size_inches(17,5)
    plt.show()

    pennelty_number=len(gl_list)
    fig,ax = plt.subplots(ncols=pennelty_number,nrows=2)
    for i in range(2):
        for j in range(pennelty_number):
                if i<1:
                    value=set_zero(abs(ggl_list[j]))
                    ax[i][j].imshow(value)
                else:
                    value=set_zero(abs(gl_list[j]))
                    ax[i][j].imshow(value)
    fig.set_size_inches(17,5)
    plt.show()


    
