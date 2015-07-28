__author__ = 'alvarobrandon'

import numpy
from scipy.sparse import csr_matrix


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                if R[i,j] > 0:
                    eij = R[i,j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                if R[i,j] > 0:
                    e = e + pow(R[i,j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i,k],2) + pow(Q[k,j],2))
        if e < 0.001:
            break
    return P, Q.T

data = numpy.random.randint(6,size=15)
rows = numpy.random.randint(50,size=15)
cols = numpy.random.randint(50,size=15)
m = csr_matrix((data,(rows,cols)),shape=(50,50))

N = m.shape[0] ### Number of users
M = m.shape[1] ### Number of items
K = 2
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)


nP, nQ = matrix_factorization(m, P, Q, K)
