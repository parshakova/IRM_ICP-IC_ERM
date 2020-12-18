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
import scipy
from sem import popul_erm_solution

from torch.autograd import grad

import matplotlib
import matplotlib.pyplot as plt


def check_for_nans(vec):
    vec = vec.view(-1)
    return torch.isnan(vec).sum() > 0

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


class BestParameters(object):
    def __init__(self):
        self.past = None
        self.params = []

    def __call__(self, err, params):
        if self.past != None:
            if err < self.past:
                self.params = [par.clone() for par in params]
                self.past = err
        else:
            self.params = [par.clone() for par in params]
            self.past = err




def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization(object):
    def __init__(self, environments, args, orders=None, betas = None, orig_risk=False, sigmas=None):
        best_reg = 0; env = []
        best_err = float('inf')
        population = "popul" in args["method_reg"]

        if population != True:
            # train from data - empricial risk minimization
            environments = environments[::-1]
            x_val = environments[-1][0]
            y_val = environments[-1][1]
            print("IRM reg n_size", x_val.numpy().shape)
            env = environments[:-1]

        self.pairs = []; self.betas = []; self.ws = []

        if population:
            if args["method_reg"] == "irm_popul":
                regs = [10.]
            else:
                regs = [1e-3]
        else:
            if args["method_reg"] == "all":
                #regs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                regs = [1e-5, 1e-3, 1e-1]
            else:
                regs = [ 1e-4, 1e-1]

        all_grads = {}
        eps = 1e-04

        for reg in regs:

            loss_function = self.create_loss_func(betas, sigmas, args, reg, orig_risk, env)

            err, grads = self.train_model(betas, sigmas, args, loss_function, env=env, reg=reg, \
                                            init_true=args["phi_init"], eps=eps, orig_risk = orig_risk)
            if not population:
                err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            all_grads[reg] = grads

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
        self.all_grads = all_grads


    def train_model(self, betas, sigmas, args, loss_function, env=None, reg=0, patience=15, init_true=0, eps=1e-05, orig_risk = False):
        dim_x = betas[0].size(0)

        train_w = args["train_w"] == 1
        opt = self.initialize_weights(dim_x, args, init_true = init_true, train_w = train_w, betas = betas, sigmas=sigmas)

        loss = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=patience)
        Id =  torch.eye(dim_x, dim_x)

        grads = []
        watch = BestParameters()

        for iteration in range(args["n_iterations"]):

            error, penalty, err = loss_function(self.phi, self.w)
                
            w_str = pretty(self.solution())
            if "popul" in args["method_reg"] and orig_risk ==True: 
                summary = "{:05d} | {:.5f} | {:.5f} |  {}".format(iteration, reg, err.item(), w_str)
            else:
                summary = "{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration, reg, error, penalty,
                                                                      w_str)

            gr = self.make_optimization_step(err, opt, iteration, args, summary, loss_function)
            early_stopping(err.item())
            if check_for_nans(gr): 
                print(iteration, "nans in gradient")
                break
            gr = torch.norm(gr)
            if args["verbose"] and iteration % 10 == 0:
                print(summary)
            watch(gr, self.parameters)

            if iteration % args["grad_freq"] == 0:
                grads += [ gr.item()]

            if gr < eps: # or early_stopping.early_stop 
                print("Early stopping at %d"%iteration)
                break

        print("iteration = ", iteration, "gradient = ", watch.past)
        self.phi = watch.params[0].clone()
        if args["train_w"] == 1:
            self.w = watch.params[1].clone()

        return watch.past.item(), grads


    def create_loss_func(self, betas, sigmas, args, reg, orig_risk, environments):
        def get_loss(phi, w):
            error = 0; penalty = 0; err = 0
            if "popul" in args["method_reg"]:
                # population risk minimization
                if orig_risk == False:
                    # optimize loss with transformed regularization term
                    err, grads = self.train_popul_transformed_reg(betas, sigmas, args, reg=reg, init_true=args["phi_init"], eps=eps)                
                    for beta_e, sigma_e in zip(betas, sigmas):
                        error += ((phi @ w - beta_e).T @ sigma_e @ (phi @ w - beta_e)).sum()
                        penalty += 4*(phi.T @ sigma_e @ (beta_e - phi @ w) ).pow(2).sum()#.mean()

                    if args["method_reg"] == "irm_popul":
                        err = error + reg * penalty
                    else:
                        err = reg * error + (1 - reg) * penalty
                elif orig_risk == True :
                    # optimize loss with original regularization term
                    for beta_e, sigma_e in zip(betas, sigmas):
                         
                        invPhi = torch.inverse(phi.T @ sigma_e @ phi)
                        PhiD = invPhi @ phi.T @ sigma_e
                        M = (sigma_e + reg * (PhiD.T @ PhiD))
                        err = err + ((beta_e - phi @ w).T @ M @ (beta_e - phi @ w) ).squeeze()
            else:
                # empricial risk minimization
                for x_e, y_e in environments:
                    error_e = (x_e @ phi @ w - y_e).pow(2).mean()
                    penalty += grad(error_e, w, create_graph=True)[0].pow(2).sum()
                    error += error_e
                err = (reg * error + (1 - reg) * penalty)
            return error, penalty, err

        return get_loss


    def initialize_weights(self, dim_x, args, init_true = 0, betas = None, sigmas=None, train_w= False):
        if init_true == 1:
            # initialize parameters to true ones (ERM)
            eps = torch.normal(mean=torch.zeros(dim_x**2), std=torch.ones(dim_x**2)*1e-5).view(dim_x, dim_x)
            u_hat = popul_erm_solution(betas, sigmas) 
            self.phi = torch.nn.Parameter(torch.diag(u_hat.squeeze()) + eps)
            params = [self.phi]
            if train_w:
                # train parameters w
                self.w = torch.nn.Parameter(torch.ones(dim_x, 1))
                params += [self.w]
            else:
                # set parameters w = 1 and fix them 
                self.w = torch.ones(dim_x, 1)
                self.w.requires_grad = True
        else:
            # randomly initialize parameters Phi
            self.phi = torch.nn.Parameter(torch.randn(dim_x, dim_x))
            params = [self.phi]
            if train_w:                
                self.w = torch.nn.Parameter(torch.randn(dim_x, 1))
                params += [self.w]
            else:
                self.w = torch.randn(dim_x, 1)
                self.w.requires_grad = True

        if args["optim"] == "adam":
            opt = torch.optim.Adam(params, lr=args["lr"])
        elif args["optim"] == "sgd":
            opt = torch.optim.SGD(params, lr=args["lr"], momentum=0.9)
        elif args["optim"] == "newton":
            opt = torch.optim.SGD(params, lr=args["lr"], momentum=0.9)
            self.parameters = params

        print("init_w: train_w = ", train_w, " init_Phi: init_true = ", init_true==1)
        return opt

    
    def update_theta(self, dtheta, params):
        # return unflatten (theta + dtheta)
        curr = 0
        for p in params:
            Nsize = torch.flatten(p.data).size()[0]
            dpar = dtheta[curr:curr+Nsize].reshape(p.data.size())
            p.data = p.data + dpar
            curr += Nsize

    def zero_grad_newton(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def jacobian(self, y, x, create_graph=False, flatten_x = False):                                        
        jac = []                                                                                          
        flat_y = y.view(-1)                                                                          
        grad_y = torch.zeros_like(flat_y)                                                               
        for i in range(len(flat_y)):                                                                      
            grad_y[i] = 1.                                                                                
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            if not flatten_x:
                grad_x = grad_x.view(x.shape) 
            else:
                grad_x = grad_x.view(-1)
            jac.append(grad_x)                                                          
            grad_y[i] = 0. 
        if not flatten_x:                                                                               
            return torch.stack(jac).view(y.shape + x.shape)  
        else:
            return torch.stack(jac) 
    
    def newton_step_update(self, loss, loss_function, args):                                             

        def hessian_grad_all_params(y, thetas):
            # x = list(parameters) not flattened 
            Ndim = 0
            for p in thetas:
                Ndim += torch.flatten(p.data).size()[0]

            hessian = torch.zeros((Ndim, Ndim))
            grad_vec = torch.zeros((Ndim, 1))
            i_size = 0

            for i, pi in enumerate(thetas):
                grad_i = self.jacobian(y, pi, create_graph = i==0)
                pi_size = torch.flatten(pi.data).size()[0] 
                grad_vec[i_size:i_size+pi_size] = grad_i.data.view(-1, 1)
                j_size = 0
                for j, pj in enumerate(thetas):
                    hess_ij = self.jacobian(grad_i, pj, flatten_x=True, create_graph = j==0)
                    pj_size = torch.flatten(pj.data).size()[0]  
                    hessian[i_size:i_size+pi_size, j_size:j_size+pj_size] = hess_ij
                    hessian[j_size:j_size+pj_size, i_size:i_size+pi_size] = hess_ij.T
                    j_size += pj_size

                i_size += pi_size

            return hessian, grad_vec

        self.zero_grad_newton(self.parameters)

        # take the second gradient
        hess, grad_vec = hessian_grad_all_params(loss, self.parameters)
        # newton step
        print("k=", np.linalg.cond(hess.data.numpy()))
        dtheta = - torch.inverse(hess) @ grad_vec.view(-1, 1)

        self.update_theta(dtheta, self.parameters)

        print("hess", hess.data.numpy())

        error_hess = self.test_hessian(hess, grad_vec, loss_function, args)
        print("#"*5 + " Hessian norm ", error_hess)

        return grad_vec

    def test_hessian(self, hess, grad_vec, loss_function, args):
        def jacobian_all_params(y, thetas):
            # x = list(parameters) not flattened 
            Ndim = 0
            for p in thetas:
                Ndim += torch.flatten(p.data).size()[0]

            grad_vec = torch.zeros((Ndim, 1))
            i_size = 0

            for i, pi in enumerate(thetas):
                grad_i = self.jacobian(y, pi, create_graph=True)
                pi_size = torch.flatten(pi.data).size()[0] 
                grad_vec[i_size:i_size+pi_size] = grad_i.data.view(-1, 1)

                i_size += pi_size

            return grad_vec

        # Hv = 1/e * (g(x + ev) - g(x))
        g1 = grad_vec

        eps = 1#1e-6
        v = eps*torch.randn(grad_vec.shape)


        params = [p.clone() for p in self.parameters]

        if args["train_w"] == 1:
            phi, w = params
        else:
            phi = params[0]
            w = self.w
        _, _, loss = loss_function(phi, w)
        self.zero_grad_newton(params)
        g12 = jacobian_all_params(loss, params)

        print("g1", g1.squeeze())
        print("g1_repeat", g12.squeeze())

        self.update_theta(v, params)
        print("phi", self.phi, "\nphi2", params[0], "\nv", v.squeeze())

        if args["train_w"] == 1:
            phi, w = params
        else:
            phi = params[0]
            w = self.w

        _, _, loss = loss_function(phi, w)
        self.zero_grad_newton(params)
        g2 = jacobian_all_params(loss, params)

        rhs = 1./eps * (g2 - g1)

        print((hess @ v).squeeze())
        print(rhs.squeeze())

        return torch.norm(hess @ v - rhs).item()
        

    def make_optimization_step(self, err, opt, iteration, args, summary, loss_function):

        if args["optim"] == "newton":
            grads = self.newton_step_update(err, loss_function, args)
            
            if args["train_w"] == 1:
                self.phi, self.w = self.parameters
            else:
                self.phi = self.parameters[0]
        else:
            opt.zero_grad()
            err.backward()
            opt.step()

            grads = self.phi.grad.view(-1)
            
            params = [self.phi]
            if args["train_w"] == 1:
                grads = torch.cat([grads, self.w.grad.view(-1)], 0)
                params += [self.w]
            

            self.parameters = params

        #print(self.parameters)
        return grads


    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


class InvariantCausalPrediction(object):

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)


    def __init__(self, environments, args, orders=None, betas = None, sigmas = None):

        def get_pvalue(subset, x_all, y_all, e_all):
            # null hypothesis   H0: S is causal
            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                # probability that res_in and res_out have same distribution
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
                # keep null hypothesis => S is causal
                if args["cond_in"] == 'n':
                    # no conditional independence
                    # accept the set S using original ICP test
                    # without conditional independence term
                    accepted_subsets.append(set(subset))
                    L = []
                    for e in range(len(orders)):
                        L += [set(subset).intersection(orders[e])]
                    if args["verbose"]:
                        print("Accepted subset:", subset, "L", L, "p_value:", p_value)
                else:
                    # enforce CI (conditional independence)
                    # do regression for the residual variables: S^c
                    #       if for all j \in S^c are not causal => accept S
                    #       otw reject S
                    # j is causal when the residuals are similar

                    x_s = x_all[:, subset]
                    reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

                    y_S = (y_all - reg.predict(x_s)).ravel()

                    # S^c = {j: j \notin S}
                    js = omega_set - set(subset)

                    exists_causal_j = False

                    for j in sorted(list(js)):
                        gammas = []
                        x_j =  x_all[:, j:j+1]
                        flag = False # flag == False only if j in S^c is causal  
                        if args["cond_in"] == 'eq_abs':
                            # check if gamma_e 's are close to each other
                            # if |gamma_1 - gamma_2| small
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                reg_je = LinearRegression().fit(x_j[e_idx, :], y_S[e_idx])
                                gamma_e = reg_je.coef_[0]
                                for gamma in gammas:
                                    if np.abs(gamma-gamma_e) > 1e-1:
                                        flag = True                                
                                gammas += [gamma_e]

                        elif args["cond_in"] == 'eq_conf_delta':
                            # regression coefficient for x_j: gamma is 1D
                            # get confidence interval for gamma_ek and combine it with c.i. of gamma_em, 
                            # i.e. (gamma_1 - gamma_2) +/- delta
                            # if |gamma_1 - gamma_2| > 2*delta  ==>  gamma_1 != gamma_2
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                res = sm.OLS(endog=y_S[e_idx], exog=x_j[e_idx, :]).fit()
                                # student t distribution; one std from mean
                                ci_gamma_e = res.conf_int(0.318/len(environments)).squeeze()
                                #ci_gamma_e = res.conf_int(0.318).squeeze()
                                gamma_e = np.mean(ci_gamma_e)
                                # variance
                                delta_e = (ci_gamma_e[1] - gamma_e)
                                
                                for (gamma_ej, delta_ej) in gammas:
                                    delta = np.sqrt(delta_e**2 + delta_ej**2)
                                    # correction for the number ofg environments
                                    if np.abs(gamma_e - gamma_ej) > 2*delta:
                                        # outside of confidence interval
                                        flag = True                                
                                gammas += [(gamma_e, delta_e)]

                        elif args["cond_in"] == 'eq_conf_var':
                            # regression coefficient for x_j: gamma is 1D
                            # get confidence interval for gamma_ek and combine it with c.i. of gamma_em, 
                            # i.e. (gamma_1 - gamma_2) +/- delta
                            # if |gamma_1 - gamma_2| > 2*delta  ==>  gamma_1 != gamma_2
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                X = sm.add_constant(x_j[e_idx, :])
                                res = sm.OLS(endog=y_S[e_idx], exog=X).fit()
                                # student t distribution
                                b0, b1 = res.params[0], res.params[1] 
                                N = y_S[e_idx].shape[0]
                                p = 2

                                residuals = np.expand_dims(y_S[e_idx], axis=1) - b0 - b1*x_j[e_idx, :]
                                sigma2_hat = (1./(N - p - 1)*np.dot(residuals.T, residuals)).squeeze()
                                var_b1 = sigma2_hat * np.linalg.inv(np.dot(x_j[e_idx, :].T, x_j[e_idx, :]))
                                var_e = np.sqrt(var_b1/ N).squeeze()

                                gamma_e = res.params[-1]

                                
                                for (gamma_ej, var_ej) in gammas:
                                    std = np.sqrt((var_e + var_ej)/N)
                                    # correction for the number of environments
                                    tval = scipy.stats.t.ppf(1-args["alpha"]/len(environments), N-1)
                                    tval = scipy.stats.t.ppf(1-args["alpha"], N-1)
                                    if np.abs(gamma_e - gamma_ej) / std > tval:
                                        # gammas are different
                                        flag = True                                
                                gammas += [(gamma_e, var_e)]

                        elif args["cond_in"] == 'eq_chi_var':
                            # gamma is 1D
                            # get p value for chi^2(m-1) distribution 
                            # if p < alpha  ==>  gamma_1 != ... != gamma_m
                            gammas = []
                            var_e = []
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]

                                X = sm.add_constant(x_j[e_idx, :])
                                res = sm.OLS(endog=y_S[e_idx], exog=X).fit()
                                # student t distribution
                                b0, b1 = res.params[0], res.params[1] 
                                N = y_S[e_idx].shape[0]
                                p = 2
                                gamma_e = res.params[-1]
                                residuals = np.expand_dims(y_S[e_idx], axis=1) - b0 -  b1*x_j[e_idx, :]
                                sigma2_hat = (1./(N - p - 1)*np.dot(residuals.T, residuals)).squeeze()
                                var_b1 = (sigma2_hat * np.linalg.inv(np.dot(x_j[e_idx, :].T, x_j[e_idx, :]))).squeeze()

                                var_e += [var_b1]                                
                                gammas += [gamma_e]

                            gammas = np.array(gammas)
                            var_e = np.array(var_e)
                            k = gammas.shape[0]
                            gamma_bar = np.ones(k)*gammas.mean()
                            #Chi2 test
                            val = (np.divide((gammas - gamma_bar)**2, var_e)).sum()
                            p_val = 1 - chi2.cdf(val, k-1)
                            #print("chi p_val = ", p_val, val,"S=", subset,"j=", j ) #val, )

                            if p_val < self.alpha:
                                # outside of confidence interval
                                flag = True

                        elif args["cond_in"] == 'eq_chi_delta':
                            # gamma is 1D
                            # get p value for chi^2(m-1) distribution 
                            # if p < alpha  ==>  gamma_1 != ... != gamma_m
                            gammas = []
                            var_e = []
                            for e in range(len(environments)):
                                e_idx = np.where(e_all == e)[0]
                                #reg_je = LinearRegression().fit(x_j[e_idx, :], y_S[e_idx]); gamma_e = reg_je.coef_[0]
                                res = sm.OLS(endog=y_S[e_idx], exog=x_j[e_idx, :]).fit()
                                ci_gamma_e = res.conf_int(0.05).squeeze()
                                gamma_e = np.mean(ci_gamma_e)
                                var_e += [((ci_gamma_e[1] - ci_gamma_e[0])/4.)**2]                                
                                gammas += [gamma_e]

                            gammas = np.array(gammas)
                            var_e = np.array(var_e)
                            k = gammas.shape[0]
                            gamma_bar = np.ones(k)*gammas.mean()
                            #Chi2 test
                            val = (np.divide((gammas - gamma_bar)**2, var_e)).sum()
                            p_val = 1 - chi2.cdf(val, k-1)
                            #print("chi p_val = ", p_val, val,"S=", subset,"j=", j ) #val, )

                            if p_val < self.alpha:
                                # outside of confidence interval
                                flag = True

                        elif args["cond_in"] == 'pval':
                            p_value_j = get_pvalue([j], x_all, y_S, e_all)/len(environments)
                            # if null hypothesis => [j] is causal
                            if p_value_j < self.alpha:
                                # j is not causal
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
                if args["verbose"]:
                    pass
                    #print("*** NOT Accepted subset:", subset, "p_value:", p_value)

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


    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args, orders=None, betas=None, sigmas = None):
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
