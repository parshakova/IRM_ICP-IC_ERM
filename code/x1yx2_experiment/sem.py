# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#



import random
import torch
from itertools import chain, combinations
import numpy as np

def popul_erm_solution(betas_e, sigmas_e):
        invSigma = torch.inverse(torch.stack(sigmas_e).sum(0))
        res = 0
        for beta_e, sigma_e in zip(betas_e, sigmas_e):
            res += sigma_e @ beta_e
        res = invSigma @ res
        return res

class SEM_X1YX2X3(object):
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
        print("sigma_w\n", self.sigma_w)
        np.set_printoptions(suppress=True)
        if not scramble:
            self.scramble = torch.eye(p)


    def solution(self):
        w = torch.cat((self.theta_pl.sum(1), torch.zeros(self.p - self.k))).view(-1, 1)
        return w, self.scramble

    def get_sigma_e_beta(self, xs, theta_mi, theta_pl, envs_order):
        sigma_e = torch.diag(torch.ones(self.p)*self.v**2)
        k = self.k

        # [k+1,...,2k] x [1,...,k]=> \theta_{+j} \theta_{-i} v^2 = \Sigma_ij
        outer_prod = self.v**2*(theta_mi @ theta_pl.T )
        sigma_e[k:2*k, 0:k] = outer_prod
        # [1,...,k] x [k+1,...,2k]=> \theta_{+i} \theta_{-j} v^2 = Sigms_ij
        sigma_e[0:k, k:2*k] = outer_prod.T

        # [k+1,...,2k] x k+1,...,2k] => {1}_{i=j}v^2 + {1}_{i\neq j}\text{sgn}_{(\theta_{-i}\theta_{-j})}(v^2 - \sigma^2_w)
        sign_theta_mi = theta_mi / self.beta

        outer_prod = (self.v**2 - self.sigma_w**2) * (sign_theta_mi @ sign_theta_mi.T) + torch.diag(torch.ones(self.k)* (self.sigma_w**2))
        beta_e = torch.cat((theta_pl, torch.zeros((self.p-self.k, 1))), dim=0)
        sigma_e[k:2*k, k:2*k] = outer_prod


        sigma_e = sigma_e[:, envs_order]
        sigma_e = sigma_e[envs_order, :]
        print("sigma_e")
        print(envs_order)
        print(np.round(sigma_e.numpy(),  decimals=4))

        print("beta_e")
        print(beta_e.squeeze().numpy())

        return sigma_e, beta_e


    def __call__(self, n, nenv):
        def get_var(x_all, s):
            cov1 = np.round(np.cov(x_all, rowvar=False), decimals=4)
            print("var(%s)"%s)
            print(cov1)
            # A = x_all
            # cov2 = np.dot((A - A.mean(axis=0, keepdim=True)).T, A - A.mean(axis=0, keepdim=True)) / (A.shape[0]-1)

        def get_var_cov(x_all, y_all, s):
            #get_var(x_all, s)
            A = x_all; b=y_all.squeeze()
            cov = np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)
            print('cov(%s,Y)'%s)
            print(cov)

        def shuffle_columns(x, dim):
            reorder_cols = list(range(dim))
            random.shuffle(reorder_cols)
            idx = torch.repeat_interleave(torch.tensor(reorder_cols).view(1,-1), n, dim=0)
            return torch.gather(x, 1, idx), reorder_cols

        epsilon = torch.normal(mean=torch.zeros(n), std=torch.ones(n)*self.sigma).view(n,1) 
        w = torch.normal(mean=torch.zeros(n*self.k), std=torch.ones(n*self.k)*self.sigma_w).view(n,  self.k)
        # sample X1 from the fixed distribution: causal
        x1 = torch.normal(mean=torch.zeros(n*self.k), std=torch.ones(n*self.k)*self.v).view(n,  self.k)
        
        y = (x1 @ self.theta_pl).sum(1, keepdim=True) + epsilon
        self.theta_mi = torch.diag(torch.tensor(np.random.choice([-self.beta, self.beta], self.k, p=[0.5, 0.5])))
        print("theta-\n", self.theta_mi)
        # sample X2 from changing distribution: effects
        x2 = y.repeat(1, self.k) @ self.theta_mi + w
        get_var_cov(x1, y, "x1")
        get_var_cov(x2, y, "x2")
        x3 = torch.normal(mean=torch.zeros(n*(self.p - 2*self.k)), \
            std=torch.ones(n*(self.p - 2*self.k))*self.v).view(n, self.p - 2*self.k)
        get_var_cov(x3, y, "x3")
        x23 = torch.cat((x2, x3), 1)

        unshuffled_xs = torch.cat((x1, x23), 1)
        
        if self.shuffle:
            x23, reorder_cols = shuffle_columns(x23, x23.numpy().shape[1])
        envs_order = list(range(self.k)) 
        all_cols_order = list(range(self.k)) 
        for i, val in  enumerate(reorder_cols):
            all_cols_order += [val + self.k]
            if val < self.k:
                envs_order += [i+self.k]
        xs = torch.cat((x1, x23), 1)
        print("envs order ", envs_order, reorder_cols)
        get_var(xs, "X[1:p]")
        theta_mi, theta_pl = self.theta_mi.sum(1, keepdim=True), self.theta_pl.sum(1, keepdim=True)
        sigma_e, beta_e = self.get_sigma_e_beta(unshuffled_xs, theta_mi, theta_pl, all_cols_order)

        return xs, y, beta_e, sigma_e, set(envs_order)




class IRM_ERM_SimpleEnvs(object):
    def __init__(self, dim, k=None, ones=True, scramble=False, hetero=True, hidden=False):
        self.dim = dim 
        self.betas = []; self.sigmas = []
        self.sigma = 0.05


    def solution(self):
        return self.betas

    def __call__(self, n, env):
        sigma_e = torch.eye(self.dim)
        x = torch.normal(mean=torch.zeros(n*self.dim), std=torch.ones(n*self.dim)).view(n,  self.dim)
        epsilon = torch.normal(mean=torch.zeros(n), std=torch.ones(n)*self.sigma).view(n,  1)

        beta = torch.randn(self.dim, 1) * (env+1)

        y = torch.matmul(x, beta) + epsilon

        self.betas += [beta]
        self.sigmas += [sigma_e]

        return x, y, beta, sigma_e



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
