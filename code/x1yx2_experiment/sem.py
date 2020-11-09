# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Plot SEM
# python main.py --n_iterations 50000 --n_reps 1 --n_samples 1500 --dim 10 --k 4 --env_list 3 --env_rat 1:1:1 --seed 123  --cond_in eq_conf --plot 1

import random
import torch
from itertools import chain, combinations
import numpy as np

class SEM_X1YX2X3(object):
    # 
    def __init__(self, p, k, ones=True, scramble=False, hetero=True, hidden=False, shuffle=True):
        self.hetero = hetero
        self.hidden = hidden
        self.k = k
        self.p = p
        self.shuffle = shuffle

        r1 = (1./k)**0.5; r2 = 0
        ksi = r1* torch.rand(1)[0]
        self.sigma = 2*torch.rand(1)[0] + 1e-6
        self.alpha = (ksi*self.sigma) / (1. - k*ksi**2)**0.5
        self.beta = ksi / (self.sigma**2 + k*self.alpha**2)**0.5
        print("alpha ", self.alpha, "beta", self.beta)
        self.v = 1
        self.sigma_w = (self.v**2 - self.beta**2*(self.sigma**2 + k*self.alpha**2*self.v**2))**0.5

        self.theta_pl = torch.diag(torch.tensor(np.random.choice([-self.alpha, self.alpha], k, p=[0.5, 0.5])))
        print("var(Y) = ", self.sigma**2 + k*self.alpha**2*self.v**2)
        print("cov(X1,Y) = ", self.alpha*self.v**2)
        print("cov(X2,Y) = ", self.beta*(self.sigma**2 + k*self.alpha**2*self.v**2))
        print("theta+\n", self.theta_pl)
        
        if not scramble:
            self.scramble = torch.eye(p)


    def solution(self):
        w = torch.cat((self.theta_pl.sum(1), torch.zeros(self.p - self.k))).view(-1, 1)
        return w, self.scramble

    def __call__(self, n, nenv):
        def get_var(x_all, s):
            cov = np.round(np.cov(x_all, rowvar=False), decimals=2)
            print("var(%s)"%s)
            print(cov)
        def get_var_cov(x_all, y_all, s):
            #get_var(x_all, s)
            A = x_all; b=y_all.squeeze()
            cov = np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)
            print('cov(%s,Y)'%s)
            print(cov)
        def shuffle_columns(x, dim):
            reorder_cols = list(range(dim))
            random.shuffle(reorder_cols)
            print("environment", reorder_cols)
            idx = torch.repeat_interleave(torch.tensor(reorder_cols).view(1,-1), n, dim=0)
            return torch.gather(x, 1, idx), reorder_cols

        epsilon = torch.normal(mean=torch.zeros(n), std=torch.ones(n)*self.sigma).view(n,1) 
        w = torch.normal(mean=torch.zeros(n*self.k), std=torch.ones(n*self.k)*self.sigma_w).view(n,  self.k)
        # sample X1 from the fixed distribution: causal
        x1 = torch.normal(mean=torch.zeros(n*self.k), std=torch.ones(n*self.k)*self.v).view(n,  self.k)
        
        y = (x1 @ self.theta_pl).sum(1, keepdim=True) + epsilon
        get_var(y, "y")
        get_var_cov(x1,y, "x1")
        self.theta_mi = torch.diag(torch.tensor(np.random.choice([-self.beta, self.beta], self.k, p=[0.5, 0.5])))
        print("theta-\n", self.theta_mi)
        # sample X2 from changing distribution: effects
        x2 = y.repeat(1, self.k) @ self.theta_mi + w
        get_var_cov(x2, y, "x2")
        x3 = torch.normal(mean=torch.zeros(n*(self.p - 2*self.k)), \
            std=torch.ones(n*(self.p - 2*self.k))*self.v).view(n, self.p - 2*self.k)
        get_var_cov(x3, y, "x3")
        x23 = torch.cat((x2, x3), 1)
        
        if self.shuffle:
            x23, reorder_cols = shuffle_columns(x23, x23.numpy().shape[1])
        envs_order = list(range(self.k)) 
        for i, val in  enumerate(reorder_cols):
            if val in range(self.k):
                envs_order += [i+self.k]
        xs = torch.cat((x1, x23), 1)
        print("envs order ", envs_order, reorder_cols)
    
        return xs, y, set(envs_order)




class IRM_ERM_SimpleEnvs(object):
    def __init__(self, dim, k=None, ones=True, scramble=False, hetero=True, hidden=False):
        self.dim = dim 
        self.betas = []


    def solution(self):
        return self.betas

    def __call__(self, n, env):
        x = torch.normal(mean=torch.zeros(n*self.dim), std=torch.ones(n*self.dim)).view(n,  self.dim)

        beta = torch.randn(self.dim, 1) * (env+1)

        y = torch.matmul(x, beta)

        self.betas += [beta]

        return x, y, beta



class ChainEquationModel(object):
    def __init__(self, dim, k=None, ones=True, scramble=False, hetero=True, hidden=False):
        self.hetero = hetero
        self.hidden = hidden
        self.dim = dim // 2

        if ones:
            self.wxy = torch.eye(self.dim)
            self.wyz = torch.eye(self.dim)
        else:
            self.wxy = torch.randn(self.dim, self.dim) / dim
            self.wyz = torch.randn(self.dim, self.dim) / dim

        if scramble:
            self.scramble, _ = torch.qr(torch.randn(dim, dim))
        else:
            self.scramble = torch.eye(dim)

        if hidden:
            self.whx = torch.randn(self.dim, self.dim) / dim
            self.why = torch.randn(self.dim, self.dim) / dim
            self.whz = torch.randn(self.dim, self.dim) / dim
        else:
            self.whx = torch.eye(self.dim, self.dim)
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)

    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim))).view(-1, 1)
        return w, self.scramble

    def __call__(self, n, env):
        h = torch.randn(n, self.dim) * env
        x = h @ self.whx + torch.randn(n, self.dim) * env

        if self.hetero:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim)
        else:
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim) * env

        return torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True)
