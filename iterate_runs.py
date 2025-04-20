"""
Iterate over a specific set of parameters (size, topology, model),
by repeatedly calling the main file.

Values are stored and then used to compute mean, stdev and their ratio.
"""

import os.path
import os
import numpy as np

N_ITER = 100
MODEL = 'paper'
TOPOLOGY = 'True'
N_SITES = 10

globals()['topological'] = TOPOLOGY
globals()['n_sites'] = N_SITES
globals()['model'] = MODEL

initial_gs = np.empty(2**N_SITES)
db_eigvals = []
db_gs_fidelity = []
db_ent_entropy = []
db_string_par = [[], []]

for iteration in range(N_ITER):
    if not os.path.exists("main_ED.py"):
        raise FileNotFoundError('main_ED.py does not exist.')    

    with open('main_ED.py') as file:
        # exec(file.read(), globals={'topological': TOPOLOGY, 'n_sites': 4, 'model': 'paper'})
        exec(file.read())
    if iteration == 0:
        initial_gs = paper.eigvecs[:, 0]

    db_eigvals.append(paper.eigvals)
    db_gs_fidelity.append(initial_gs.dot(paper.eigvecs[:, 0]))
    db_ent_entropy.append(ent_entropy)
    db_string_par[0].append(z_parameter)
    db_string_par[1].append(x_parameter)

    if iteration % 10 == 0:    print(f"Done iteration num. {iteration + 1}")

aggr_eigvals = [
    np.average(np.array(db_eigvals), axis=0),
    np.std(np.array(db_eigvals), axis=0)
]
aggr_eigvals.append(aggr_eigvals[1] / aggr_eigvals[0])
aggr_gs_fidelity = [
    np.average(np.array(db_gs_fidelity)),
    np.std(np.array(db_gs_fidelity))
]
aggr_gs_fidelity.append(aggr_gs_fidelity[1] / aggr_gs_fidelity[0])
aggr_ent_entropy = [
    np.average(np.array(db_ent_entropy), axis=0),
    np.std(np.array(db_ent_entropy), axis=0)
]
aggr_ent_entropy.append(aggr_ent_entropy[1] / aggr_ent_entropy[0])
aggr_string_par = [[], []]
aggr_string_par[0] = [
    np.average(np.array(db_string_par[0])),
    np.std(np.array(db_string_par[0]))
]
aggr_string_par[0].append(aggr_string_par[0][0] / aggr_string_par[0][1])
aggr_string_par[1] = [
    np.average(np.array(db_string_par[1])),
    np.std(np.array(db_string_par[1]))
]
aggr_string_par[1].append(aggr_string_par[1][0] / aggr_string_par[1][1])
print('done')