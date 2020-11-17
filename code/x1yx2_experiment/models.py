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
from scipy.stats import chisquare
from scipy.stats import chi2
import scipy.optimize
import statsmodels.api as sm

from torch.autograd import grad

import matplotlib
import matplotlib.pyplot as plt


class EarlyStopping(object):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.past = None
        self.early_stop = False

    def __call__(self, err):
        if self.past != None:
            if err > self.past:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
                self.past = err
        else:
            self.past = err



def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization(object):
    def __init__(self, environments, args, orders=None, betas = None, orig_risk=False):
        best_reg = 0
        best_err = 1e6
        population = "popul" in args["setup_sem"]

        if population != True:
            # train from data - empricial risk minimization
            environments = environments[::-1]
            x_val = environments[-1][0]
            y_val = environments[-1][1]
            print("IRM reg n_size", x_val.numpy().shape)

        self.pairs = []; self.betas = []; self.ws = []

        
        if population == 1:
            regs = [0.5, 1, 10, 50, ]
            #regs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        else:
            regs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        for reg in regs:
            if population == 1:
                # population risk minimization
                if orig_risk == False:
                    err = self.train_popul(betas, args, reg=reg).item()
                elif orig_risk == True :
                    err = self.train_popul2(betas, args, reg=reg, init_true=args["phi_init"]).item()
            else:
                # empricial risk minimization
                self.train(environments[:-1], args, reg=reg)
                err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.5f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
                best_w = self.w.clone()

            self.pairs += [(reg, self.phi.clone())]
            self.betas += [self.phi @ self.w]
            self.ws += [self.w]

        self.phi = best_phi
        self.w = best_w
        self.reg = best_reg
        self.beta = self.phi @ self.w

    def train_popul2(self, betas, args, reg=0, patience=10, init_true=0):
        dim_x = betas[0].size(0)

        if init_true == 1:
            self.phi = torch.nn.Parameter(torch.diag(torch.mean(torch.stack(betas),dim=0).squeeze()))
            if args["train_w"] == 1:
                self.w = torch.nn.Parameter(torch.ones(dim_x, 1))
            else:
                self.w = torch.ones(dim_x, 1)
                self.w.requires_grad = True
        else:
            self.phi = torch.nn.Parameter(torch.randn(dim_x, dim_x))
            if args["train_w"] == 1:
                self.w = torch.nn.Parameter(torch.randn(dim_x, 1))
            else:
                self.w = torch.randn(dim_x, 1)
                self.w.requires_grad = True

        

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=patience)
        Id =  torch.eye(dim_x, dim_x)

        for iteration in range(args["n_iterations"]):
            error = 0

            for beta_e in betas:
                
                M = (Id + reg* torch.inverse(self.phi.T @ self.phi))
                error += ((beta_e - self.phi @ self.w).T @ M @ (beta_e - self.phi @ self.w) )
                if (M!=M).sum() > 0:
                    print(M!= M)
                    print("inv", torch.inverse(self.phi.T @ self.phi))
                    print("M", M)
                    assert 1 == 0
            opt.zero_grad()
            err = error
            err.backward()
            opt.step()

            if args["verbose"] and iteration % 10000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} |  {}".format(iteration,
                                                                      reg,
                                                                      error.item(),
                                                                      w_str))

            early_stopping(err.item())
        
            if early_stopping.early_stop:
                print("Early stopping at %d"%iteration)

                break
        return err

    def train_popul(self, betas, args, reg=0, patience=10):
        dim_x = betas[0].size(0)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
         
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=patience)

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0

            for beta_e in betas:
                error += loss(self.phi @ self.w, beta_e)
                penalty += (  self.phi.T @ (beta_e - self.phi @ self.w) ).pow(2).mean()

            opt.zero_grad()
            if args["setup_sem"] == "irm_popul":
                err = error + reg * penalty
            else:
                err = reg * error + (1 - reg) * penalty
            #
            err.backward()
            opt.step()

            if args["verbose"] and iteration % 10000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))

            early_stopping(err.item())
        
            if early_stopping.early_stop:
                print("Early stopping at %d"%iteration)

                break
        return err

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
            # null hypothesis = S is causal
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

        x_all = []; y_all = []; e_all = []


        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all); y_all = np.vstack(y_all); e_all = np.hstack(e_all)

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


                    js = omega_set - set(subset)

                    exists_causal_j = False

                    for j in sorted(list(js)):
                        gammas = []
                        x_j =  x_all[:, j:j+1]
                        flag = False # flag == False only if j in S^c is causal  
                        if args["cond_in"] == 'eq_abs':
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                reg_je = LinearRegression().fit(x_j[e_idx, :], y_S[e_idx])
                                gamma_e = reg_je.coef_[0]
                                # check if gamma_e 's are close to each other
                                for gamma in gammas:
                                    if np.abs(gamma-gamma_e) > 1e-1:
                                        flag = True
                                
                                gammas += [gamma_e]

                        elif args["cond_in"] == 'eq_conf':
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                res = sm.OLS(endog=y_S[e_idx], exog=x_j[e_idx, :]).fit()
                                # gamma is 1D
                                # get confidence interval for gamma_ek and combine it with c.i. of gamma_em, 
                                # i.e. (gamma_1 - gamma_2) +/- delta
                                # if |gamma_1 - gamma_2| > 2*delta  ==>  gamma_1 != gamma_2
                                ci_gamma_e = res.conf_int(0.05).squeeze()
                                gamma_e = np.mean(ci_gamma_e)
                                delta_e = ci_gamma_e[1] - gamma_e
                                
                                for (gamma_ej, delta_ej) in gammas:

                                    delta = np.sqrt(delta_e**2 + delta_ej**2)
                                    if np.abs(gamma_e - gamma_ej) > 2*delta:
                                        # outside of confidence interval
                                        flag = True
                                
                                gammas += [(gamma_e, delta_e)]

                        elif args["cond_in"] == 'eq_chi':
                            # gamma is 1D
                            # get p value for chi^2(m-1) distribution 
                            # if p < alpha  ==>  gamma_1 != ... != gamma_m
                            gammas = []
                            var_e = []
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                #reg_je = LinearRegression().fit(x_j[e_idx, :], y_S[e_idx])
                                #gamma_e = reg_je.coef_[0]
                                res = sm.OLS(endog=y_S[e_idx], exog=x_j[e_idx, :]).fit()
                                ci_gamma_e = res.conf_int(0.05).squeeze()
                                gamma_e = np.mean(ci_gamma_e)
                                var_e = ((ci_gamma_e[1] - ci_gamma_e[0])/4.)**2
                                
                                gammas += [gamma_e]

                            gammas = np.array(gammas)
                            var_e = np.array(var_e)
                            k = gammas.shape[0]
                            gamma_bar = np.ones(k)*gammas.mean()
                            #val2, p_val = chisquare(gammas, f_exp=gamma_bar, ddof=0)
                            val = (np.divide((gammas - gamma_bar)**2, var_e)).sum()
                            p_val = 1 - chi2.cdf(val, k-1)
                            #print("chi p_val = ", p_val, val,"S=", subset,"j=", j ) #val, )

                            if p_val < self.alpha:
                                # outside of confidence interval
                                flag = True

                        elif args["cond_in"] == 'pval':
                            p_value_j = get_pvalue([j], x_all, y_S, e_all)
                            # if [j] is causal => null hypothesis is not rejected
                            if p_value_j < self.alpha:
                                flag = True

                        if flag == False:
                            # j is causal
                            exists_causal_j = True
                            break

                    if exists_causal_j == False:
                        # exists mismatched pair for all j's => all gammas are NOT similar for that j in S^c
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
    def __init__(self, environments, args, orders=None, betas=None):
        def pretty(vector):
            vlist = vector.view(-1).tolist()
            return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()
       
        #print("*"*30)
        cov = np.round(np.cov(y_all, rowvar=False), decimals=2)
        #print('var(Y)')
        #print(cov)
        cov = np.round(np.cov(x_all, rowvar=False), decimals=2)
        #print('var(X)')
        #print(cov)
        A = x_all; b=y_all.squeeze()
        cov = np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0]-1)
        #print('cov(X,Y)')
        #print(np.round(cov, decimals=2))

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
