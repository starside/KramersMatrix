"""
Created by Joshua Kelly, copyright 2015
This is open source software, using the The MIT License (MIT)
Contact me at joshua.kelly@physics.ucla.edu or inst.zombie@gmail.com

This software reqired graph-tools and Cython
https://graph-tool.skewed.de/

The MIT License (MIT)

Copyright (c) 2011-2016 Twitter, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import sys
from graph_tool.all import *
cimport numpy as np
cimport cython

DTYPE=np.int32
ctypedef np.int32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def floydwarshall(np.ndarray[DTYPE_t, ndim=2] mat, np.ndarray[DTYPE_t, ndim=2] parent, int numVertices):
    assert mat.dtype == DTYPE and parent.dtype == DTYPE
    cdef unsigned int nv = numVertices
    cdef int pInf = 2*nv
    cdef unsigned int i,j,k
    cdef int a,v
    for i in range(nv):
        for j in range(nv):
            if i == j or mat[i,j] == pInf:
                parent[i,j] = -1
            else:
                parent[i,j] = i
    for k in range(nv):
        for i in range(nv):
            for j in range(nv):
                a = mat[i,j]
                if a < pInf:
                    continue
                v = mat[i,k] + mat[k,j]
                if v < a:
                    mat[i,j] = v
                    parent[i,j] = parent[k,j]
                    continue

cdef void enq(int *q):
    q[0] = 1

def tq():
    cdef int que[10]
    myq = []
    enq(que)
    for i in range(10):
        myq.append(que[i])        
    print myq

def buildAdjacency(g):
    nv = g.num_vertices()
    cdef int pInf = 2*nv
    m = np.ones((nv,nv),dtype=np.int32)*pInf
    p = np.zeros((nv,nv),dtype=np.int32)
    adjacency = np.zeros((nv,nv),dtype=np.int32)
    edgelabels = np.ones((nv,nv),dtype=np.int32)*(-1)
    currentEdge = 0
    for e in g.edges():
        i = e.source()
        j = e.target()
        m[i][j] = 1
        m[j][i] = 1
        adjacency[i][j] = 1
        adjacency[j][i] = 1
        edgelabels[i][j] = currentEdge  #label the edges
        edgelabels[j][i] = currentEdge
        currentEdge += 1
    return m,p,edgelabels,adjacency

def countConnections(adj,nv):
    count = 0
    for i in range(nv):
        for j in range(nv):
            if adj[i][j] == 1:
                count += 1
    return count

def initCountdir(adj,nv):
    memo = np.zeros(nv+2*countConnections(adj,nv),dtype=np.int32)
    vind = np.zeros(nv,dtype=np.int32)
    offset = 0
    for i in range(nv):
        vind[i] = offset
        offset += 1
        for j in range(nv):  #count the number of branches on this node
            if adj[i][j] == 1:
                memo[vind[i]] += 1 #number of branches
                memo[offset] = j  #The ignore branch label
                memo[offset+1] = 0  #The data
                offset += 2
    return vind,memo


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void storeMemo(int node,int ig,int value,np.ndarray[DTYPE_t, ndim=1] memo,np.ndarray[DTYPE_t, ndim=1] vind):
    cdef int numBranchesIndex, numBranches, i
    numBranchesIndex = int(vind[node])
    numBranches = int(memo[numBranchesIndex])
    for i in range(numBranches):
        if memo[numBranchesIndex+1+i*2] == ig:
            memo[numBranchesIndex+1+1+i*2] = value

#Store value in node,ig
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int getMemo(int node,int ig,np.ndarray[DTYPE_t, ndim=1] memo,np.ndarray[DTYPE_t, ndim=1] vind):
    cdef int numBranchesIndex, numBranches, i
    numBranchesIndex = int(vind[node])
    numBranches = int(memo[numBranchesIndex])
    for i in range(numBranches):
        if memo[numBranchesIndex+1+i*2] == ig:
            return memo[numBranchesIndex+1+1+i*2]
    return -1 #If not found

#Memoizing recursive Countdir function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef countdir(np.ndarray[DTYPE_t, ndim=2] mat, int nv, int node, int ignore, np.ndarray[DTYPE_t, ndim=1] memo, np.ndarray[DTYPE_t, ndim=1] vind):
    cdef int rv, branchsum, v
    cdef unsigned int i
    rv = getMemo(node,ignore,memo,vind)
    if rv == -1:
        sys.stderr.write("Countdir cannot take non-adjacent vertices\n")
        exit(0)
    if rv > 0:
        return rv
    branchsum = 0
    for i in range(nv):
        v = mat[node,i]
        if v != 0 and i != ignore:
            branchsum += countdir(mat,nv,i,node,memo,vind)
    storeMemo(node,ignore,1+branchsum,memo,vind)
    return 1 + branchsum

def _findgkm(g):
    cdef np.ndarray[DTYPE_t, ndim=2] fwar,parent,edgelist,adjacency
    cdef np.ndarray[np.double_t, ndim=2] gkm
    cdef np.ndarray[DTYPE_t, ndim=1] vi,memo
    cdef unsigned int i,j
    cdef int Nv,Ne,pathEdgeLength,ve1,vb1,edge1,edge2,ve2,vb2
    Nv = g.num_vertices()
    Ne = g.num_edges()
    fwar,parent,edgelist,adjacency = buildAdjacency(g)  #initialize a variety of matrices
    vi, memo = initCountdir(adjacency, Nv )  #Initialize the memo
    floydwarshall(fwar,parent, Nv )       #Run the fw-algorithm.  This is written in Cython
    gkm = np.zeros((Ne,Ne),dtype=np.double)         #Initialize G(k,m) matrix
    #Run over all vertex pairs
    for i in range(Nv):
        for j in range(i):
            if i == j:  #Cannot use diagonals
                continue
            pathEdgeLength = fwar[i,j]
            ve1,vb1 = j,int(parent[i,j])    #Find vertices on one connecting edge
            edge1 = edgelist[ve1,vb1]
            ve2,vb2 = i,int(parent[j,i])   #Find vertices on the other edge
            edge2 = edgelist[ve2,vb2]
            gkm[edge1,edge2] = 1.0*countdir(adjacency,Nv,ve2,vb2,memo,vi)*countdir(adjacency,Nv,ve1,vb1,memo,vi)
            gkm[edge2,edge1] = gkm[edge1,edge2]
    return gkm#, edgelist



