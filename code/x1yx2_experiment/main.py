# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem import ChainEquationModel,SEM_X1YX2X3,IRM_ERM_SimpleEnvs
from models import *

import matplotlib.pyplot as plt
import argparse
import torch
import math
import numpy
import random


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = torch.where(w != 0)[0].view(-1)
    i_noncausal = torch.where(w == 0)[0].view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2)
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2)
    else:
        error_noncausal = 0

    return error_causal.mean().item(), error_noncausal.mean().item(), error_causal, error_noncausal

def find_betas(args):
    def compute_u_rhs(u, betas, reg):
        # u for IRM
        a = 0; b = 0
        lamb = 1-reg
        eta = reg
        for beta_e in betas:
            dot_prod = torch.matmul(u.T, beta_e - u).squeeze()
            a += (eta-lamb*dot_prod)*beta_e
            b += (eta-2*lamb*dot_prod)
        res = 1./b * a
        return res

    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)
        random.seed(1)

    if args["setup_sem"] == "irm_erm_simple":
        setup_str = ""

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "IRM": InvariantRiskMinimization
    }
    if int(args["env_list"]) > 1:
        all_methods["IRM"] = InvariantRiskMinimization
    

    methods = all_methods

    all_sems = []
    all_solutions = []
    all_environments = []
    n_env = int(args["env_list"])

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "irm_erm_simple":
            sem = IRM_ERM_SimpleEnvs(args["dim"])

            env_list = range(n_env)
            # ratios are forced to be the same
            n = args["n_samples"]
            print("sample in each envs ", n)
            environments = []
            betas = []
            for e in env_list:
                res = sem(n, e)
                environments += [res[:2]]
                betas += [res[2]]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    sol_dict = {}
    p = args["dim"]

    ones = torch.ones(p,1)
    P = 1./p * (torch.matmul(ones, ones.T))
    P_ort = torch.eye(p) - P
    for sem, environments in zip(all_sems, all_environments):
        sem_betas = sem.solution()

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)

            if method_name == "IRM":
                print("******** IRM *********")
                #phi = method.phi
                #lamb = method.reg
                sol_dict[method_name] = []
                for (lamb,phi) in method.pairs:
                    # verify whether Phi_ort = 0
                    #     phi = 1/p(u 1^T)
                    print(lamb)
                    u = torch.matmul(phi, ones)
                    phi_ort = torch.matmul(phi, P_ort)
                    print("||Phi_ort|| = ", torch.norm(phi_ort))
                    u_hat = compute_u_rhs(u, betas[1:], lamb)
                    print("u = ", u, "\nu_hat = ", u_hat)
                    print("||u-u_hat|| = ", torch.norm(u-u_hat))
                    sol_dict[method_name] += [(lamb, torch.norm(phi_ort), torch.norm(u-u_hat))]


            elif method_name == "ERM":
                print("******** ERM *********")
                u_hat = 1./n_env * sum(betas)
                u = method.w
                print("u = ", u, "\nu_hat = ", u_hat)
                print("||u-u_hat|| = ", torch.norm(u-u_hat))
                sol_dict[method_name]  = torch.norm(u-u_hat)

    return sol_dict

def find_betas_popul(args):
    def compute_u_rhs(u, betas, lamb, eta):
        # u for IRM
        a = 0; b = 0
        for beta_e in betas:
            dot_prod = torch.matmul(u.T, beta_e - u).squeeze()
            a += (eta-lamb*dot_prod)*beta_e
            b += (eta-2*lamb*dot_prod)
        res = 1./b * a
        return res

    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)
        random.seed(1)

    if args["setup_sem"] == "irm_erm_simple":
        setup_str = ""

    all_methods = {
        "IRM": InvariantRiskMinimization,
        "ERM": EmpiricalRiskMinimizer
    }
    if int(args["env_list"]) > 1:
        all_methods["IRM"] = InvariantRiskMinimization
    

    methods = all_methods

    all_sems = []
    all_solutions = []
    all_environments = []
    n_env = int(args["env_list"])

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "irm_erm_popul":
            sem = IRM_ERM_SimpleEnvs(args["dim"])

            env_list = range(n_env)
            # ratios are forced to be the same
            n = args["n_samples"]
            print("sample in each envs ", n)
            environments = []
            betas = []
            for e in env_list:
                res = sem(n, e)
                environments += [res[:2]]
                betas += [res[2]]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    sol_dict = {}
    p = args["dim"]

    ones = torch.ones(p,1)
    P = 1./p * (torch.matmul(ones, ones.T))
    P_ort = torch.eye(p) - P
    for sem, environments in zip(all_sems, all_environments):
        sem_betas = sem.solution()

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args, betas = betas)

            if method_name == "IRM":
                print("******** IRM *********")
                
                sol_dict[method_name] = []
                for (reg,phi) in method.pairs:
                    # verify whether Phi_ort = 0
                    #     phi = 1/p(u 1^T)
                    print(reg)
                    u = torch.matmul(phi, ones)
                    phi_ort = torch.matmul(phi, P_ort)
                    print("||Phi_ort|| = ", torch.norm(phi_ort))
                    #lamb = reg; eta = 1
                    eta = reg; lamb = 1-reg
                    u_hat = compute_u_rhs(u, betas, lamb, eta)
                    print("u = ", u, "\nu_hat = ", u_hat)
                    print("||u-u_hat|| = ", torch.norm(u-u_hat))
                    sol_dict[method_name] += [(reg, torch.norm(phi_ort), torch.norm(u-u_hat))]

                beta_irm = method.beta


            elif method_name == "ERM":
                print("******** ERM *********")
                u_hat = 1./n_env * sum(betas)
                u = method.w
                print("u = ", u, "\nu_hat = ", u_hat)
                print("||u-u_hat|| = ", torch.norm(u-u_hat))
                sol_dict[method_name]  = [torch.norm(u-u_hat), torch.norm(beta_irm - method.w).detach().item(), \
                                                                        torch.norm(beta_irm - u_hat).detach().item()]

    return sol_dict

def find_betas_irm_popul(args):
    def compute_u_rhs(u, betas, lamb, eta):
        # u for IRM
        a = 0; b = 0
        for beta_e in betas:
            dot_prod = torch.matmul(u.T, beta_e - u).squeeze()
            a += (eta-lamb*dot_prod)*beta_e
            b += (eta-2*lamb*dot_prod)
        res = 1./b * a
        return res

    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)
        random.seed(1)

    setup_str = ""

    all_methods = {
        "IRM": InvariantRiskMinimization
    }
    if int(args["env_list"]) > 1:
        all_methods["IRM"] = InvariantRiskMinimization
    

    methods = all_methods

    all_solutions = []
    n_env = int(args["env_list"])

    if args["setup_sem"] == "irm_popul":
        sem = IRM_ERM_SimpleEnvs(args["dim"])

        env_list = range(n_env)
        # ratios are forced to be the same
        n = args["n_samples"]
        print("sample in each envs ", n)
        environments = []
        betas = []
        for e in env_list:
            res = sem(n, e)
            environments += [res[:2]]
            betas += [res[2]]
    else:
        raise NotImplementedError


    p = args["dim"]

    ones = torch.ones(p,1)
    P = 1./p * (torch.matmul(ones, ones.T))
    P_ort = torch.eye(p) - P
    
    sem_betas = sem.solution()

    

    barycenter = 1./n_env * sum(betas)
    seeds = [123, 222]
    sol_dict = {s:{init:[] for init in [0,1]} for s in seeds}
    phi_sol_dict = {s:{0:[]} for s in seeds}

    for seed in seeds:        
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        torch.set_num_threads(1)
        random.seed(1)

        args["train_w"] = 1
        if seed == seeds[0]:
            inits = [0, 1]
        else:
            inits = [0]

        for init in inits:
            args["phi_init"] = init
            method = all_methods["IRM"](environments, args, betas = betas, orig_risk=True)

            idx = len(method.pairs)//2
            reg, phi = method.pairs[idx]
            w, w_beta = method.ws[idx], method.betas[idx]
            
            sol_dict[seed][init] = (reg, None, w_beta, torch.norm(w_beta - barycenter))

       
        method = all_methods["IRM"](environments, args, betas = betas, orig_risk=False)

        idx = len(method.pairs)//2
        reg, phi = method.pairs[idx]
        w, w_beta = method.ws[idx], method.betas[idx]
        
        phi_ort = torch.matmul(phi, P_ort)
        print("||Phi_ort|| = ", torch.norm(phi_ort))
        phi_sol_dict[seed][0] = (reg, torch.norm(phi_ort), w_beta, torch.norm(w_beta - barycenter))


    return sol_dict, phi_sol_dict


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)
        random.seed(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_ones={}_hidden={}_hetero={}_scramble={}".format(
            args["setup_ones"],
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "simple":
        setup_str = ""
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization
    }
    if int(args["env_list"]) > 1:
        all_methods["IRM"] = InvariantRiskMinimization
    

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     ones=args["setup_ones"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"])

            env_list = [float(e) for e in args["env_list"].split(",")]
            environments = [sem(args["n_samples"], e) for e in env_list]
        elif args["setup_sem"] == "simple":
            sem = SEM_X1YX2X3(args["dim"],args["k"], args["env_shuffle"])

            env_list = range(int(args["env_list"]))
            ratios = list(map(int, args["env_rat"].split(':')))
            n = args["n_samples"]
            n_samples = [math.ceil(ni*1.0/sum(ratios)*n) for ni in ratios]
            print("sample in envs ", n_samples)
            environments = []
            env_orders = []
            for e in env_list:
                res = sem(n_samples[e], e)
                environments += [res[:2]]
                env_orders += [res[2]]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    sol_dict = {}
    for sem, environments in zip(all_sems, all_environments):
        sem_solution, sem_scramble = sem.solution()

        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem_solution), 0, 0)
        ]
        #sol_dict["SEM"] = [sem_solution.numpy(), 0, 0]

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args, env_orders)

            method_solution = sem_scramble @ method.solution()

            err_causal, err_noncausal, ecaus, enoncaus = errors(sem_solution, method_solution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(
                setup_str,
                method_name,
                pretty(method_solution),
                err_causal,
                err_noncausal))

            sol_dict[method_name] = [method_solution.detach().view(-1).numpy(), \
                                                ecaus.detach().view(-1).numpy(), enoncaus.detach().view(-1).numpy()]

        all_solutions += solutions

    return all_solutions, sol_dict

def plot_results(sol_dict, ns, dim, args):
    """
    sol_dict[n][method_name] = [beta, err_causal, err_noncausal]
    method_name in {IRM, ICP, ERM} 
    """
    method_names = ["ERM", "ICP", "IRM"]

    fig, axs = plt.subplots(nrows=1, ncols=len(ns), 
                                    figsize=(5*len(ns), 5))

    width = 0.2
    pos = [-1, 0, 1]
    ind = np.arange(dim)
    for i,n in enumerate(ns):
        for k, method in enumerate(method_names):
            x = sol_dict[n][method][0]
            axs[i].bar(ind + pos[k]*width, x, width, label=method)

        axs[i].set_xticks(ind)
        axs[i].set_xticklabels(('x%d'%j for j in range(x.shape[0])))
        axs[i].legend(prop={'size': 10})
        axs[i].set_title('N = %d'%n)

    plt.savefig('plot_betas%s.png'%args["env_rat"])
    plt.show()
    
    # plot bars for causal - non causal errors
    fig, axs = plt.subplots(nrows=2, ncols=len(ns)) 
                                    #figsize=(5*len(ns), 5))

    width = 0.5
    hatch = [None, "//"]
    ind = np.arange(len(method_names))
    for i,n in enumerate(ns):
        for k in range(2):
            x_means = []; x_vars = []; colors = []
            for m, method in enumerate(method_names):
                x_means += [np.mean(sol_dict[n][method][k+1])]
                x_vars += [np.std(sol_dict[n][method][k+1])]
                colors += ["C" + str(m)]
            axs[k,i].bar(ind, x_means, width, yerr=np.array(x_vars), log=True, color=colors, hatch=hatch[k])

            axs[k,i].set_xticks(ind)
            axs[k,i].set_xticklabels(tuple(method_names))
            #axs[k,i].legend(prop={'size': 10})
            if k == 0:
                axs[k,i].set_title('Caus N = %d'%n)
            else:
                axs[k,i].set_title('Nonc N = %d'%n)


    #plt.savefig('plot_caus_noncaus%s.png'%args["env_rat"])
    plt.show()



def plot_results_envs(sol_dict, envs, dim, args):
    """
    sol_dict[e][method_name] = [beta, err_causal, err_noncausal]
    method_name in {IRM, ICP, ERM} 
    """
    method_names = ["ERM", "ICP", "IRM"]

    fig, axs = plt.subplots(nrows=1, ncols=len(envs), 
                                    figsize=(5*len(envs), 5))

    width = 0.2
    pos = [-1, 0, 1]
    ind = np.arange(dim)
    for i,e in enumerate(envs):
        for k, method in enumerate(method_names):
            x = sol_dict[e][method][0]
            axs[i].bar(ind + pos[k]*width, x, width, label=method)

        axs[i].set_xticks(ind)
        axs[i].set_xticklabels(('x%d'%j for j in range(x.shape[0])))
        axs[i].legend(prop={'size': 10})
        axs[i].set_title('|E| = %d'%e)

    plt.savefig('plot_betas%s.png'%args["env_rat"])
    plt.show()
    
    # plot bars for causal - non causal errors
    fig, axs = plt.subplots(nrows=2, ncols=len(envs)) 
                                    #figsize=(5*len(envs), 5))

    width = 0.5
    hatch = [None, "//"]
    ind = np.arange(len(method_names))
    for i,e in enumerate(envs):
        for k in range(2):
            x_means = []; x_vars = []; colors = []
            for m, method in enumerate(method_names):
                x_means += [np.mean(sol_dict[e][method][k+1])]
                x_vars += [np.std(sol_dict[e][method][k+1])]
                colors += ["C" + str(m)]
            axs[k,i].bar(ind, x_means, width, yerr=np.array(x_vars), log=True, color=colors, hatch=hatch[k])

            axs[k,i].set_xticks(ind)
            axs[k,i].set_xticklabels(tuple(method_names))
            #axs[k,i].legend(prop={'size': 10})
            if k == 0:
                axs[k,i].set_title('Caus N = %d'%e)
            else:
                axs[k,i].set_title('Nonc N = %d'%e)


    #plt.savefig('plot_caus_noncaus%s.png'%args["env_rat"])
    plt.show()



def plot_results_irm_erm_popul(sol_dict, envs):
    """
    sol_dict[e][IRM] = [lamb, ||Phi_ort||, ||u-u_hat||]
    sol_dict[e][ERM] = [||u-u_hat||, ||u_hat_irm - u_hat_erm ||, ||u_hat_irm - u_mean ||]
    method_name in {IRM, ICP, ERM} 
    """
    method_names = ["ERM", "IRM"]
    ncols = 4
    fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                                    figsize=(5*ncols, 5))

    width = 0.3
    pos = [-1./2, 1./2]
    ind = np.arange(len(envs))
    phi_ort_norms = []
    regs = []
    for el in sol_dict[envs[0]]["IRM"]:
        regs += [el[0]]
    for i, method in enumerate(method_names):
        if method == "ERM":
            xs = []
        else:
            xs = {reg:[] for reg in regs}
            phi_ort_norms = {reg:[] for reg in regs}
        for e in envs:
            if method == "ERM":
                xs += [sol_dict[e][method][0]]

            elif method == "IRM":
                
                for (reg, phi_norm, u_norm) in sol_dict[e][method]:
                    xs[reg] += [u_norm] 
                    phi_ort_norms[reg] += [phi_norm]

        if method == "ERM":
            axs[0].plot(ind, xs, label  = method)
        elif method == "IRM":
            for reg in regs:
                axs[0].plot(ind, xs[reg], label  = method+" %.1e"%reg)

    axs[0].set_xticks(ind)
    axs[0].set_xticklabels((str(e) for e in envs))
    axs[0].legend(prop={'size': 10})
    axs[0].set_title("||u-u_hat||")

    for reg in regs:
        axs[1].plot(ind, phi_ort_norms[reg], label  = "%.1e"%reg)
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels((str(e) for e in envs))
    axs[1].set_title("||Phi_ort||")
    axs[1].legend(prop={'size': 10})

    diffs = []
    for e in envs:
        diffs += [sol_dict[e]["ERM"][1]]
    axs[2].plot(ind, diffs)
    axs[2].set_xticks(ind)
    axs[2].set_xticklabels((str(e) for e in envs))
    axs[2].set_title("||beta_irm - beta_erm||")

    diffs = []
    for e in envs:
        diffs += [sol_dict[e]["ERM"][2]]
    axs[3].plot(ind, diffs)
    axs[3].set_xticks(ind)
    axs[3].set_xticklabels((str(e) for e in envs))
    axs[3].set_title("||beta_irm - beta_mean||")

    plt.savefig('irm_erm_curves_dim%d.png'%args["dim"])
    plt.show()


def plot_results_irm_popul_orig(sol_dict, envs):
    """
    sol_dict[e][seed][init] += [(reg, torch.norm(phi_ort), w_beta, torch.norm(w_beta - barycenter))]
    method_name in {IRM} 
    """
    method = "IRM"
    ncols = 3
    fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                                    figsize=(5*ncols, 5))

    width = 0.3
    pos = [-1./2, 1./2]
    ind = np.arange(len(envs))
    phi_ort_norms = {}
    seeds = [123, 222]
    inits = [0, 1]
    regs = []
    xs = {}
    
    for el in sol_dict[envs[0]][seeds[0]][0]:
        regs += [el[0]]
    w_norms = {reg:{init:{} for init in inits} for reg in regs}

    for seed in seeds:
        for  init in inits:
            if init == 1 and seed != seeds[0]:
                continue
            xs[str(seed)+"|"+str(init)] = {reg:[] for reg in regs}
            phi_ort_norms[str(seed)+"|"+str(init)] = {reg:[] for reg in regs}
            for e in envs:
                for (reg, phi_norm, w_beta, w_norm) in sol_dict[e][seed][init]:
                    xs[str(seed)+"|"+str(init)][reg] += [w_norm] 
                    phi_ort_norms[str(seed)+"|"+str(init)][reg] += [phi_norm]
                    if seed == seeds[0]:
                        w_norms[reg][init][e] = w_beta.detach().numpy()
            if init == 0:
                linestyle = ":"
            else:
                linestyle = "-"
            for reg in regs:
                axs[0].plot(ind, xs[str(seed)+"|"+str(init)][reg], label = str(seed)+"|"+str(init)+"|%.1e"%reg, linestyle=linestyle)


    axs[0].set_xticks(ind)
    axs[0].set_xticklabels((str(e) for e in envs))
    axs[0].legend(prop={'size': 8})
    axs[0].set_title("||beta_irm - beta_mean||")

    for seed in seeds:
        for  init in inits:
            if init == 1 and seed != seeds[0]:
                continue
            if init == 0:
                linestyle = ":"
            else:
                linestyle = "-"
            for reg in regs:
                axs[1].plot(ind, phi_ort_norms[str(seed)+"|"+str(init)][reg], label  = str(seed)+"|"+str(init)+"|%.1e"%reg, linestyle=linestyle)
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels((str(e) for e in envs))
    axs[1].set_title("||Phi_ort||")
    axs[1].legend(prop={'size': 8})

    for reg in regs:
        norms = []
        for e in envs:
            norms += [np.linalg.norm(w_norms[reg][0][e]-w_norms[reg][1][e])]
        axs[2].plot(ind, norms, label  = "%.1e"%reg)
    axs[2].set_xticks(ind)
    axs[2].set_xticklabels((str(e) for e in envs))
    axs[2].set_title("||beta_irm - beta_penal||")
    axs[2].legend(prop={'size': 8})

    plt.savefig('irm_curves_dim%d.png'%args["dim"])
    plt.show()


def plot_results_irm_popul(sol_dict, envs):
    """
    sol_dict[wphi][e][seed][init] = (reg, torch.norm(phi_ort), w_beta, torch.norm(w_beta - barycenter))
    method_name in {IRM} 
    compare IRM R(phi) with R(w,phi) optimization
    """
    method = "IRM"
    ncols = 3
    fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                                    figsize=(5*ncols, 5))

    width = 0.3
    pos = [-1./2, 1./2]
    ind = np.arange(len(envs))
    phi_ort_norms = {}
    seeds = [123, 222]
    inits = [0, 1]
    regs = []
    modes = ["wphi", "phi"]
    
    w_norms = {t:{init:{s:{} for s in seeds} for init in inits} for t in modes}

    for seed in seeds:
        for  init in inits:
            if init == 1 and seed != seeds[0]:
                continue

            for wmode in modes:
                if wmode == "phi" and init != 0:
                    continue
                xs = []
                for e in envs:
                    (reg, phi_norm, w_beta, w_norm) = sol_dict[wmode][e][seed][init]
                    xs += [w_norm] 
                    #phi_ort_norms[str(seed)+"|"+str(init)+"%s|"%wmode] = phi_norm
                    w_norms[wmode][init][seed][e] = w_beta.detach().numpy()
                if init == 0:
                    linestyle = ":"
                else:
                    linestyle = "-"
                axs[0].plot(ind, xs, label = "%s|"%wmode+str(seed)+"|"+str(init)+"|%.1e"%reg, linestyle=linestyle)



    axs[0].set_xticks(ind)
    axs[0].set_xticklabels((str(e) for e in envs))
    axs[0].legend(prop={'size': 8})
    axs[0].set_title("||beta_irm - beta_mean||")


    for s in seeds:
        norms = []
        for e in envs:
            norms += [np.linalg.norm(w_norms["phi"][0][s][e]-w_norms["wphi"][1][seeds[0]][e])]
        axs[1].plot(ind, norms, label  = s)
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels((str(e) for e in envs))
    axs[1].set_title("||beta_irm - beta_penal_init||")
    axs[1].legend(prop={'size': 8})

    for s in seeds:
        norms = []
        for e in envs:
            norms += [np.linalg.norm(w_norms["phi"][0][s][e]-w_norms["wphi"][0][s][e])]
        axs[2].plot(ind, norms, label  = s)
    axs[2].set_xticks(ind)
    axs[2].set_xticklabels((str(e) for e in envs))
    axs[2].set_title("||beta_irm - beta_penal||")
    axs[2].legend(prop={'size': 8})


    plt.savefig('irm_curves_dim%d.png'%args["dim"])
    plt.show()



def plot_results_irm_erm(sol_dict, ns):
    """
    sol_dict[n][IRM] = [lamb, ||Phi_ort||, ||u-u_hat||]
    method_name in {IRM, ICP, ERM} 
    """
    method_names = ["ERM", "IRM"]
    ncols = 2
    fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                                    figsize=(5*ncols, 5))

    width = 0.3
    pos = [-1./2, 1./2]
    ind = np.arange(len(ns))
    phi_ort_norms = []
    regs = []
    for el in sol_dict[ns[0]]["IRM"]:
        regs += [el[0]]
    for i, method in enumerate(method_names):
        if method == "ERM":
            xs = []
        else:
            xs = {reg:[] for reg in regs}
            phi_ort_norms = {reg:[] for reg in regs}
        for n in ns:
            if method == "ERM":
                xs += [sol_dict[n][method]]

            elif method == "IRM":
                
                for (reg, phi_norm, u_norm) in sol_dict[n][method]:
                    xs[reg] += [u_norm] 
                    phi_ort_norms[reg] += [phi_norm]

        if method == "ERM":
            axs[0].plot(ind, xs, label  = method)
        elif method == "IRM":
            for reg in regs:
                axs[0].plot(ind, xs[reg], label  = method+" %.1e"%reg)

    axs[0].set_xticks(ind)
    axs[0].set_xticklabels((str(n) for n in ns))
    axs[0].legend(prop={'size': 10})
    axs[0].set_title("||u-u_hat||")

    for reg in regs:
        axs[1].plot(ind, phi_ort_norms[reg], label  = "%.1e"%reg)
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels((str(n) for n in ns))
    axs[1].set_title("||Phi_ort||")
    axs[1].legend(prop={'size': 10})

    plt.savefig('irm_erm_curves.png')
    #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--double_pval', type=int, default=0)
    parser.add_argument('--cond_in', type=str, default='n')
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--methods', type=str, default="ERM,ICP,IRM")
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--env_list', type=str, default="3")#default=".2,2.,5.")
    parser.add_argument('--env_rat', type=str, default="1:10:100")
    parser.add_argument('--env_shuffle', type=int, default=1)
    parser.add_argument('--setup_sem', type=str, default="simple")
    parser.add_argument('--setup_ones', type=int, default=1)
    parser.add_argument('--phi_init', type=int, default=0)
    parser.add_argument('--setup_hidden', type=int, default=0)
    parser.add_argument('--popul', type=int, default=0)
    parser.add_argument('--train_w', type=int, default=0)
    parser.add_argument('--setup_hetero', type=int, default=0)
    parser.add_argument('--setup_scramble', type=int, default=0)
    args = dict(vars(parser.parse_args()))

    if args["setup_sem"] == "irm_popul":
        args["popul"] = 2
        sol_dict = {"wphi":{}, "phi":{}}
        n = 5000
        envs = [3, 5, 8, 10, 12, 15]
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            args["n_samples"] = n
            args["env_list"] = str(e)
            n_sol_dict, phi_sol_dict = find_betas_irm_popul(args)
            sol_dict["wphi"][e] = n_sol_dict
            sol_dict["phi"][e] = n_sol_dict
        plot_results_irm_popul(sol_dict, envs)

    elif args["setup_sem"] == "irm_erm_popul":
        args["popul"] = 1
        sol_dict = {}
        n = 5000
        envs = [3, 5, 8, 10, 11, 12, 15]
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            args["n_samples"] = n
            args["env_list"] = str(e)
            n_sol_dict = find_betas_popul(args)
            sol_dict[e] = n_sol_dict
        plot_results_irm_erm_popul(sol_dict, envs)

    elif args["setup_sem"] == "irm_erm_simple":
        sol_dict = {}
        ns = [1000, 3000, 5000, 10000, 50000]
        for n in ns:
            args["n_samples"] = n
            n_sol_dict = find_betas(args)
            sol_dict[n] = n_sol_dict
        plot_results_irm_erm(sol_dict, ns)


    elif args["plot"]==1:
        sol_dict = {}
        ns = [150, 600, 1500, 2400, 3000, 5000]
        for n in ns:
            args["n_samples"] = n
            all_solutions, n_sol_dict = run_experiment(args)
            print("n = %d"%n)
            print("\n".join(all_solutions))
            sol_dict[n] = n_sol_dict
        plot_results(sol_dict, ns, args["dim"], args)

    elif args["plot"]==2:
        sol_dict = {}
        n_each = 1000
        envs = [3, 5, 8, 10, 12, 15]  
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            n = n_each*e
            args["n_samples"] = n
            args["env_list"] = str(e)
            args["env_rat"] = "1"+":1"*(e-1)
            all_solutions, n_sol_dict = run_experiment(args)
            print("n = %d"%n)
            print("\n".join(all_solutions))
            sol_dict[e] = n_sol_dict
        plot_results_envs(sol_dict, envs, args["dim"], args)

    else:
        all_solutions,_ = run_experiment(args)
        print("\n".join(all_solutions))

# Plot SEM
# python main.py --n_iterations 50000 --n_reps 1 --dim 10 --k 4 --env_list 3 --env_rat 1:1:1 --seed 123  --cond_in eq_conf --plot 1 --setup_sem irm_erm_simple
# IRM vs ERM
# python main.py --n_iterations 100000 --n_reps 1 --n_samples 1500 --dim 5 --env_list 3 --env_rat 1:1:1 --seed 123 --setup_sem irm_erm_popul

