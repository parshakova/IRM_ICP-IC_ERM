# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import math

from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind

from torch.autograd import grad

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization(object):
    def __init__(self, environments, args, orders=None):
        best_reg = 0
        best_err = 1e6

        environments = environments[::-1]
        x_val = environments[-1][0]
        y_val = environments[-1][1]
        print("IRM reg n_size", x_val.numpy().shape)

        self.pairs = []

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.5f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()

            self.pairs += [(reg, self.phi.clone())]

        self.phi = best_phi
        self.reg = best_reg

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()
        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 10000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))
        #print("w", self.w.data.numpy())

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


class InvariantCausalPrediction(object):
    def __init__(self, environments, args, orders=None):
        def get_pvalue(subset, x_all, y_all, e_all):
            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)
            return p_value


        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []



        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]
        omega_set = set(range(dim))

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            p_value = get_pvalue(subset, x_all, y_all, e_all)

            if p_value > self.alpha:
                if args["cond_in"]!= 'n':
                    # do regression for the residual variables: S^c
                    # if for all j \in S^c are not causal = accept S
                    # otw reject S
                    # causal when the resilduals are
                    x_s = x_all[:, subset]
                    reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

                    y_S = (y_all - reg.predict(x_s)).ravel()


                    flag = False
                    gammas = []
                    js = omega_set - set(subset)

                    for j in sorted(list(js)):
                        gammas = []
                        x_j =  x_all[:, j:j+1]
                        if args["cond_in"] == 'eq':
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                reg_je = LinearRegression(fit_intercept=False).fit(x_j[e_idx, :], y_S[e_idx])
                                gamma_e = reg_je.coef_
                                #gamma_e = (y_S[e_idx] - reg_je.predict(x_j[e_idx, :])).ravel()
                                for gamma in gammas:
                                    if np.allclose(gamma, gamma_e, rtol=1e-1, atol=1e-2):
                                        flag = True
                                        #break
                                
                                gammas += [gamma_e]

                        elif args["cond_in"] == 'pval':
                            p_value_j = get_pvalue([j], x_all, y_S, e_all)
                            if p_value_j > self.alpha:
                                flag = True

                        if subset == tuple(range(args["k"]-1)):
                            print(j, subset, js)
                            print([g[0] for g in gammas])

                        #if flag:
                            #print(gammas)
                            #break

                    if not flag:
                        accepted_subsets.append(set(subset))
                        if args["verbose"]:
                                print("Accepted subset:", subset, "p_value:", p_value)

                else:
                    accepted_subsets.append(set(subset))
                    L = []
                    for e in range(len(orders)):
                        L += [set(subset).intersection(orders[e])]
                    if args["verbose"]:
                                print("Accepted subset:", subset, "L", L, "p_value:", p_value)

            else:
                if args["verbose"]:
                    pass
                    #print("NOT Accepted subset:", subset, "p_value:", p_value)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args, orders=None):
        def pretty(vector):
            vlist = vector.view(-1).tolist()
            return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()
       
        print("*"*30)
        cov = np.round(np.cov(y_all, rowvar=False), decimals=2)
        print('var(Y)')
        print(cov)
        cov = np.round(np.cov(x_all, rowvar=False), decimals=2)
        #print('var(X)')
        #print(cov)
        A = x_all; b=y_all.squeeze()
        cov = np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)
        print('cov(X,Y)')
        print(np.round(cov, decimals=2))

        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w).contiguous().view(-1, 1)

        for e, (x, y) in enumerate(environments):

            w = LinearRegression(fit_intercept=False).fit(x.numpy(), y.numpy()).coef_

            w = torch.Tensor(w).contiguous().view(-1, 1)       
            print("e{} {} {} ".format(
                e,
                "ERM",
                pretty(w)))     


    def solution(self):
        return self.w
