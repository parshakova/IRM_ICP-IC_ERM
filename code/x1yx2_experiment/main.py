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
    def compute_u_rhs(u, betas, lamb):
        # u for IRM
        a = 0; b = 0
        for beta_e in betas:
            dot_prod = torch.matmul(u.T, beta_e - u).squeeze()
            a += (1-lamb*dot_prod)*beta_e
            b += (1-2*lamb*dot_prod)
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

def plot_results(sol_dict, ns, dim):
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
                
                for (reg, u_norm, phi_norm) in sol_dict[n][method]:
                    xs[reg] += [u_norm] 
                    phi_ort_norms[reg] += [phi_norm]

        if method == "ERM":
            axs[0].plot(ind, xs, label  = method)
        elif method == "IRM":
            for reg in regs:
                axs[0].plot(ind, xs[reg], label  = method+" %.5f"%reg)

    axs[0].set_xticks(ind)
    axs[0].set_xticklabels((str(n) for n in ns))
    axs[0].legend(prop={'size': 10})
    axs[0].set_title("||u-u_hat||")

    for reg in regs:
        axs[1].plot(ind, phi_ort_norms[reg], label  = "%.5f"%reg)
    axs[1].set_xticks(ind)
    axs[1].set_xticklabels((str(n) for n in ns))
    axs[1].set_title("||Phi_ort||")

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
    parser.add_argument('--setup_hidden', type=int, default=0)
    parser.add_argument('--setup_hetero', type=int, default=0)
    parser.add_argument('--setup_scramble', type=int, default=0)
    args = dict(vars(parser.parse_args()))

    if args["setup_sem"] == "irm_erm_simple":
        sol_dict = {}
        ns = [1000, 3000, 5000, 10000, 50000]
        for n in ns:
            args["n_samples"] = n
            n_sol_dict = find_betas(args)
            sol_dict[n] = n_sol_dict
        plot_results_irm_erm(sol_dict, ns)


    elif args["plot"]:
        sol_dict = {}
        ns = [150, 600, 1500, 2400, 3000]
        for n in ns:
            args["n_samples"] = n
            all_solutions, n_sol_dict = run_experiment(args)
            print("n = %d"%n)
            print("\n".join(all_solutions))
            sol_dict[n] = n_sol_dict
        plot_results(sol_dict, ns, args["dim"])
    else:
        all_solutions,_ = run_experiment(args)
        print("\n".join(all_solutions))
