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

import faulthandler


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


def setup_models_environments(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)
        random.seed(args["seed"])

    if args["setup_sem"] == "chain":
        setup_str = "chain_ones={}_hidden={}_hetero={}_scramble={}".format(
            args["setup_ones"],
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    else:
        setup_str = ""
    
    all_methods = {
            "ERM": EmpiricalRiskMinimizer,
            "ICP": InvariantCausalPrediction,
            "IRM": InvariantRiskMinimization
    }
    if  args["method_reg"] != "all": 
        del all_methods["ICP"]
    

    all_sems = []
    betas = []; env_orders = []
    all_environments = []
    n_env = int(args["env_list"])

    if args["setup_sem"] == "chain":
        sem = ChainEquationModel(args["dim"], ones=args["setup_ones"], hidden=args["setup_hidden"], scramble=args["setup_scramble"], hetero=args["setup_hetero"])
    elif args["setup_sem"] == "causal":

        sem = SEM_X1YX2X3(args["dim"],args["k"], args["env_shuffle"])
    elif args["setup_sem"] == "indep":
        sem = IRM_ERM_SimpleEnvs(args["dim"])

    print("SEM type: ", args["setup_sem"])
    
    for rep_i in range(args["n_reps"]):
        if args["method_reg"] in ["irm_popul", "irm_erm_popul", "irm_erm_nsample"]:
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

        elif args["method_reg"] == "chain":
            env_list = [float(e) for e in args["env_list"].split(",")]
            environments = [sem(args["n_samples"], e) for e in env_list]

        elif args["method_reg"] == "all":

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

    return all_sems, all_environments, setup_str, betas, all_methods, env_orders



# args["method_reg"] == "all"
def solve_irm_erm_icp(args):

    all_solutions = []
    all_sems, all_environments, setup_str, _, methods, env_orders = setup_models_environments(args)

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
            if method_name == "IRM":
                sol_dict["IRM_grad"] = [method.all_grads]
            

        all_solutions += solutions

    return all_solutions, sol_dict


def compute_u_rhs(u, betas, lamb, eta):
        # u for IRM
        a = 0; b = 0
        for beta_e in betas:
            dot_prod = torch.matmul(u.T, beta_e - u).squeeze()
            a += (eta - lamb*dot_prod)*beta_e
            b += (eta - 2*lamb*dot_prod)
        res = a / b
        return res

# args["method_reg"] == "irm_erm_nsample"
def solve_irm2_erm(args):

    n_env = int(args["env_list"])
    all_sems, all_environments, setup_str, betas, methods, env_orders = setup_models_environments(args)

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
                sol_dict[method_name] = []
                for (lamb,phi) in method.pairs:
                    # verify whether Phi_ort = 0
                    #     phi = 1/p(u 1^T)
                    print(lamb)
                    u = torch.matmul(phi, ones)
                    phi_ort = torch.matmul(phi, P_ort)
                    print("||Phi_ort|| = ", torch.norm(phi_ort))
                    u_hat = compute_u_rhs(u, betas[1:], lamb=(1-lamb)*4./args["dim"], eta=lamb)
                    print("||u-u_hat|| = ", torch.norm(u-u_hat))
                    sol_dict[method_name] += [(lamb, torch.norm(phi_ort), torch.norm(u-u_hat))]
                sol_dict["IRM_grad"] = [method.all_grads]

            elif method_name == "ERM":
                print("******** ERM *********")
                u_hat = 1./n_env * sum(betas)
                u = method.w
                print("||u-u_hat|| = ", torch.norm(u-u_hat))
                sol_dict[method_name]  = torch.norm(u-u_hat)

    return sol_dict


# args["method_reg"] == "irm_erm_popul"
def solve_irm2_erm_popul(args):

    n_env = int(args["env_list"])
    all_sems, all_environments, setup_str, betas, methods, env_orders = setup_models_environments(args)

    sol_dict = {}
    p = args["dim"]

    ones = torch.ones(p,1)
    P = 1./p * (torch.matmul(ones, ones.T))
    P_ort = torch.eye(p) - P
    for sem, environments in zip(all_sems, all_environments):
        sem_betas = sem.solution()

        for method_name in ["IRM", "ERM"]:
            method_constructor = methods[method_name]
            if method_name == "IRM":
                print("******** IRM *********")
                method = method_constructor(environments, args, betas = betas)
                
                sol_dict[method_name] = []
                for (reg,phi) in method.pairs:
                    # verify whether Phi_ort = 0
                    #     phi = 1/p(u 1^T)
                    print(reg)
                    u = torch.matmul(phi, ones)
                    phi_ort = torch.matmul(phi, P_ort)
                    print("||Phi_ort|| = ", torch.norm(phi_ort))
                    
                    u_hat = compute_u_rhs(u, betas, lamb=(1-reg)*4./args["dim"], eta=reg)
                    print("||u-u_hat|| = ", torch.norm(u-u_hat))
                    sol_dict[method_name] += [(reg, torch.norm(phi_ort), torch.norm(u-u_hat))]

                beta_irm = method.beta
                sol_dict["IRM_grad"] = [method.all_grads]


            elif method_name == "ERM":
                print("******** ERM *********")
                u_hat = 1./n_env * sum(betas)
                sol_dict[method_name]  = [torch.norm(beta_irm - u_hat).detach().item()]

    return sol_dict


# args["method_reg"] == "irm_popul"
def solve_irm1_irm2_popul(args):

    n_env = int(args["env_list"])
    all_sems, all_environments, setup_str, betas, methods, env_orders = setup_models_environments(args)

    p = args["dim"]

    ones = torch.ones(p,1)
    P = 1./p * (torch.matmul(ones, ones.T))
    P_ort = torch.eye(p) - P
    sem_betas = all_sems[0].solution()
    
    barycenter = torch.mean(torch.stack(betas), dim=0)
    sol_dict = {}
    sol_dict["erm"] = barycenter


    inits = [0, 1]

    for init in inits:
        args["phi_init"] = init

        args["train_w"] = 1        
        method = methods["IRM"](all_environments[0], args, betas = betas, orig_risk=True)
        idx = 0
        reg, phi = method.pairs[idx]
        w, w_beta = method.ws[idx], method.betas[idx]
        sol_dict["irm1_"+str(init)] = w_beta

   
        args["train_w"] = 0
        method = methods["IRM"](all_environments[0], args, betas = betas, orig_risk=False)
        idx = 0
        reg, phi = method.pairs[idx]
        w, w_beta = method.ws[idx], method.betas[idx]
        sol_dict["irm2_"+str(init)] = w_beta

    return sol_dict




def plot_irm_erm_icp(sol_dict, ns, dim, args):
    """
    sol_dict[n][method_name] = [beta, err_causal, err_noncausal]
    method_name in {IRM, ICP, ERM} 
    sol_dict[n][IRM] = [beta, err_causal, err_noncausal, all_grads]
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

    if args["plot_grad"] == 1:
        plot_grads_irm("n", sol_dict, ns, args)
    

def plot_grads_irm(name, sol_dict, ns, args):

    fig, axs = plt.subplots(nrows=1, ncols=max(len(ns), 2)) 
    for i,n in enumerate(ns):
        for reg, grads in sol_dict[n]["IRM_grad"][-1].items():
            axs[i].plot(args["grad_freq"]*np.array(range(len(grads))), grads, label  = "%.1e"%reg)
        axs[i].set_title("%s=%d"%(name, n))
        axs[i].set_yscale("log")
        axs[i].set_ylabel("||grad||")
        axs[i].set_xlabel("iter")
        axs[i].legend(prop={'size': 6})

    plt.show()


def plot_envs(sol_dict, envs, dim, args):
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

    if args["plot_grad"] == 1:
        plot_grads_irm("|E|", sol_dict, envs, args)


def plot_irm_erm(sol_dict, ns):
    """
    sol_dict[n][IRM] = [lamb, ||Phi_ort||, ||u-u_hat||, {reg:||grad||}]
    sol_dict[n][ERM] = [||u-u_hat||]
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

    plt.show()

    if args["plot_grad"] == 1:
        plot_grads_irm("n", sol_dict, ns, args)


def plot_irm_erm_popul(sol_dict, envs):
    """
    sol_dict[e][IRM] = [lamb, ||Phi_ort||, ||u-u_hat||]
    sol_dict[e][ERM] = [||u_irm - u_erm ||]
    method_name in {IRM, ERM} 
    """
    method_names = ["ERM", "IRM"]
    ncols = 3
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
            continue
        xs = {reg:[] for reg in regs}
        phi_ort_norms = {reg:[] for reg in regs}
        for e in envs:                
            for (reg, phi_norm, u_norm) in sol_dict[e][method]:
                xs[reg] += [u_norm] 
                phi_ort_norms[reg] += [phi_norm]

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
        diffs += [sol_dict[e]["ERM"][0]]
    axs[2].plot(ind, diffs)
    axs[2].set_xticks(ind)
    axs[2].set_xticklabels((str(e) for e in envs))
    axs[2].set_title("||beta_irm - beta_erm||")

    plt.savefig('irm_erm_curves_dim%d.png'%args["dim"])
    plt.show()

    if args["plot_grad"] == 1:
        plot_grads_irm("|E|", sol_dict, envs, args)


def plot_irm1_irm2_popul(sol_dict, envs):
    """
    sol_dict[e][method] = w_beta
    method_name in {IRM} 
    compare IRM1 R(phi,w) with IRM2 R(phi) optimization and ERM
    """

    # plot betas for all available methods
    fig, axs = plt.subplots(nrows=1, ncols=len(envs)+1, 
                                    figsize=(5*(len(envs)+1), 5))

    width = 0.1
    pos = [-2, -1,0, 1, 2]
    ind = np.arange(args["dim"])
    method_names = list(sol_dict[envs[0]].keys())
    for i,e in enumerate(envs):
        for k, method in enumerate(method_names):
            beta = sol_dict[e][method].detach().numpy().squeeze()
            axs[i].bar(ind + pos[k]*width, beta, width, label=method)

        axs[i].set_xticks(ind)
        axs[i].set_xticklabels(('x%d'%j for j in range(beta.shape[0])))
        axs[i].legend(prop={'size': 6})
        axs[i].set_title('|E| = %d'%e)


    
    # plot the differences ||beta_irm - beta_erm ||
    ind = np.arange(len(envs))
    
    for method in method_names:
        if method == "erm": continue
        xs = []
        for e in envs:
            w_beta = sol_dict[e][method]
            xs += [torch.norm(w_beta - sol_dict[e]["erm"]).detach().numpy().squeeze()] 

        axs[len(envs)].plot(ind, xs, label = method)

    axs[len(envs)].set_xticks(ind)
    axs[len(envs)].set_xticklabels((str(e) for e in envs))
    axs[len(envs)].legend(prop={'size': 6})
    axs[len(envs)].set_title("||beta_irm - beta_erm||")


    plt.savefig('diff_irm12_erm_dim%d.png'%args["dim"])
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--cond_in', type=str, default='n', \
        help="conditional independence; options: <eq_abs> (absolute differences), <eq_conf> (pairwise confidence intervals)," \
        + " <eq_chi> (Chi^2 test), <pval> (pvalue for set j)")
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--methods', type=str, default="ERM,ICP,IRM")
    parser.add_argument('--alpha', type=float, default=0.05, help = "alpha for hypothesis testing")
    parser.add_argument('--env_list', type=str, default="3")
    parser.add_argument('--env_rat', type=str, default="1:10:100", help="ratios of samples between environments")
    parser.add_argument('--env_shuffle', type=int, default=1)
    parser.add_argument('--setup_sem', type=str, default="causal", help = "<causal>, <indep>")
    parser.add_argument('--method_reg', type=str, default="all", help = "<irm_popul>, <irm_erm_popul>, <irm_erm_simple>, <all>")
    parser.add_argument('--setup_ones', type=int, default=1)
    parser.add_argument('--phi_init', type=int, default=0)
    parser.add_argument('--setup_hidden', type=int, default=0)
    parser.add_argument('--popul', type=int, default=0, help="optimize for population loss or empirical loss")
    parser.add_argument('--train_w', type=int, default=0, help = "train parameters w or keep them fixed (w=1)")
    parser.add_argument('--setup_hetero', type=int, default=0)
    parser.add_argument('--setup_scramble', type=int, default=0)
    parser.add_argument('--grad_freq', type=int, default=100, help="frequency for recording grad norm values of IRM")
    parser.add_argument('--plot_grad', type=int, default=0, help = "plot gradient norms for IRM")
    args = dict(vars(parser.parse_args()))
    faulthandler.enable()

    if args["method_reg"] == "irm_popul":
        # IRM1: Optimize Phi, w
        # IRM2: Optimize Phi, w=1
        # IRM1 vs IRM2 vs ERM       
        args["popul"] = 2
        n = 5000
        envs = [3, 5, 8, 10, 15]
        sol_dict = {}
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            args["n_samples"] = n
            args["env_list"] = str(e)
            sol_dict[e] = solve_irm1_irm2_popul(args)
        plot_irm1_irm2_popul(sol_dict, envs)


    elif args["method_reg"] == "irm_erm_popul":
        # IRM2 vs ERM population
        args["popul"] = 1
        sol_dict = {}
        n = 5000
        envs = [3, 5, 8, 10, 11, 12, 15]
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            args["n_samples"] = n
            args["env_list"] = str(e)
            n_sol_dict = solve_irm2_erm_popul(args)
            sol_dict[e] = n_sol_dict
        plot_irm_erm_popul(sol_dict, envs)


    elif args["method_reg"] == "irm_erm_nsample":
        # IRM2 vs ERM
        sol_dict = {}
        ns = [1000, 3000, 5000, 10000, 50000]
        for n in ns:
            args["n_samples"] = n
            n_sol_dict = solve_irm2_erm(args)
            sol_dict[n] = n_sol_dict
        plot_irm_erm(sol_dict, ns)


    elif args["plot"]==1 and args["method_reg"] == "all":
        # plot IRM, ICP, ERM results subject to 
        # number of samples in each environment
        sol_dict = {}
        ns = [150, 600, 1500, 2400, 3000, 5000]
        for n in ns:
            args["n_samples"] = n
            all_solutions, n_sol_dict = solve_irm_erm_icp(args)
            print("n = %d"%n)
            print("\n".join(all_solutions))
            sol_dict[n] = n_sol_dict
        plot_irm_erm_icp(sol_dict, ns, args["dim"], args)


    elif args["plot"]==2 and args["method_reg"] == "all":
        # plot IRM, ICP, ERM results subject to 
        # number of environments, while keeping fixed number of samples in each environment
        sol_dict = {}
        n_each = 1000
        envs = [3, 5, 8, 10, 12, 15]  
        for e in envs:
            print("#"*10 + " %d envs "%e + "#"*10)
            n = n_each*e
            args["n_samples"] = n
            args["env_list"] = str(e)
            args["env_rat"] = "1"+":1"*(e-1)
            all_solutions, n_sol_dict = solve_irm_erm_icp(args)
            print("n = %d"%n)
            print("\n".join(all_solutions))
            sol_dict[e] = n_sol_dict
        plot_envs(sol_dict, envs, args["dim"], args)

    elif args["method_reg"] == "all":
        all_solutions, n_sol_dict = solve_irm_erm_icp(args)
        print("\n".join(all_solutions))
        sol_dict = {args["n_samples"]:n_sol_dict}
        ns = [args["n_samples"]]

        if args["plot_grad"] == 1:
            plot_grads_irm("n", sol_dict, ns, args)

# Plot SEM
# python main.py --n_iterations 50000 --n_reps 1 --dim 10 --k 4 --env_list 3 --env_rat 1:1:1 --seed 123  --cond_in eq_conf --plot 1 --setup_sem irm_erm_simple
# IRM vs ERM
# python main.py --n_iterations 100000 --n_reps 1 --n_samples 1500 --dim 5 --env_list 3 --env_rat 1:1:1 --seed 123 --setup_sem irm_erm_popul

