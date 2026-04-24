import torch as th
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob

def A_system(parents, device="cuda", ws=41, j_noise=0., bones=None, multiple=1.):
    
    bone_vector = th.load('j.pt') * multiple
    
    if bones is not None:
        for j in range(bone_vector.shape[0]):
            bone_vector[j] *= bones[j]

    bone_vector += th.randn_like(bone_vector) * j_noise

    joints = [15,20,21]
    bone_matrix = th.zeros(len(joints), 66)
    for i in range(len(joints)):
        k = joints[i]
        while parents[k] != -1:
            bone_matrix[i, parents[k]*3:parents[k]*3+3] = bone_vector[k,:]
            k = parents[k]

    bone_matrix = th.kron(th.eye(3), bone_matrix)
    A = th.kron(th.eye(ws).to(device), bone_matrix.to(device))
    
    return A

def S_system(device="cuda", ws=41):
    joints = [15,20,21]
    S = th.eye(len(joints))
    for i in range(1, len(joints)+1):
        S[i-1, i%len(joints)] = -1.

    # print(S)

    S = th.kron(th.eye(3), S)
    S_full = th.kron(th.eye(ws).to(device), S.to(device))
    # print(S.shape)

    return S_full