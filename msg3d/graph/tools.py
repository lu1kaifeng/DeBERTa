import numpy as np
import torch


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def k_adjacency_torch(A, k, with_self=False, self_factor=1):
    #assert isinstance(A, np.ndarray)
    I = torch.eye(A.shape[1], dtype=A.dtype).to(A)
    I = I[None,:,:].repeat(A.shape[0],1,1)
    if k == 0:
        return I
    Ak = torch.clamp(torch.linalg.matrix_power(A + I, k), max=1.,min=0.) \
       - torch.clamp(torch.linalg.matrix_power(A + I, k - 1), max=1.,min=0.)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def walk_path_torch(A, n):
    nonzero = torch.nonzero(A)
    ret = [ [findPaths(A[b,:,:],u,n) for u in range(A.shape[1])] for b in range(A.shape[0])]
    return ret
    pass

def walk_all_paths_torch(A, n,in_channels):
    last_edge = torch.zeros([A.shape[0], n - 1, A.shape[1], A.shape[1]]).long().to(A.device)
    for k in range(1, n):
        paths = walk_path_torch(A, k)
        for i, bat in enumerate(paths):
            for j, begin in enumerate(bat):
                for path in begin:
                     last_edge[i,k - 1,j,path[-1]] = A[i, path[-2], path[-1]]
    return last_edge

def findPaths(G:torch.Tensor,u:int,n:int):
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G[u,:].nonzero() for path in findPaths(G,neighbor.item(),n-1) if u not in path]
    return paths

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def normalize_adjacency_matrix_torch(A):
    node_degrees = A.sum(-1)
    node_degrees = node_degrees.clamp(min=1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    norm_degs_matrix = torch.eye(A.shape[1])[None,:,:].repeat(A.shape[0],1,1).to(A) * degs_inv_sqrt[:,:,None].repeat(1,1,A.shape[1])
    return (norm_degs_matrix @ A @ norm_degs_matrix).to(torch.float32)

def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A