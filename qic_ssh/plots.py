import json
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import Counter

def f_exp(x, A, b):
    return A * np.exp(- (x / 2 - 1) / b)

def E_to_num(namefile: str, topological: bool):
    with open(namefile, 'r') as file:
        data = json.load(file)
    # cast contents to array for plt.eventplot
    data = [np.array(content) for content in data]

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    plt.eventplot(data, orientation='vertical', linelengths=0.8)
    # missing dots for the degeneracy of GS

    if topological:
        ax.set_title(f"Eigspectrum for {len(data)-1} topological sites")
    else:
        ax.set_title(f"Eigspectrum for {len(data)-1} trivial sites")
    ax.set_xlabel("Particle number")
    ax.set_ylabel("Energy")
    ax.set_yticks([])

    plt.show()
    # fig.savefig(f"E_to_num.pdf")

def obc_effect(topological: bool, paper: bool, orig: str):
    # for all possible sizes, check if folder exists and read data
    db_energies = list()

    if topological:
        version = 'topo'
    else:
        version = 'triv'
    
    if paper:
        model = 'paper'
    else:
        model = 'ideal'

    if 'ED' in orig:
        min_x = 6
    elif 'MPS' in orig:
        min_x = 16

    sizes = np.arange(min_x, 17, 2) if min_x == 6 else np.arange(min_x, 33, 4)
    for size in sizes:
        data_path = f"DATA_SERVER/{orig}/data_{size}_{version}_{model}"
        if not os.path.exists(data_path):
            continue
        
        # read in the eigenvalues of the problem
        energies = np.loadtxt(data_path + f"/Eigvals{size}_{model}_topo{topological}.data",
            delimiter=',')
        # save first 4 values 
        db_energies.append(np.abs(energies[0:4]) / np.max(np.abs(energies[0:4])))
    
    # plot the 4 lowest energies
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    # Fit the hybridized energy with an exponential
    y_exp_dec = np.array([1 - np.min(Energies_given_N) for Energies_given_N in db_energies])
    x_exp_dec = np.array([min_x + 2 * ii for ii in range(len(db_energies))])
    popt, pcov, infodict, _, _ = curve_fit(f_exp, x_exp_dec, y_exp_dec, full_output=True)
    x_plot = list(range(len(db_energies)))
    f_x = np.arange(-0.5, np.max(x_exp_dec) // 2, 0.01)
    plt.plot(
        f_x,
        - f_exp(f_x * 2 + min_x, popt[0], popt[1]) + 1,
        '--',
        color='0.5',
        label=f'$\\tau = {popt[1]:.3f} \\pm {np.sqrt(pcov[1, 1]):.3f}$'
    )

    plt.eventplot(
        db_energies,
        orientation='vertical',
        linelengths=0.8,
        colors=[f"C{ii}" for ii in range(len(db_energies))]
    )

    plt.plot(
        x_plot,
        [np.min(Energies_given_N) for Energies_given_N in db_energies],
        '+k',
        label=''
    )

    mode = 'topological' if topological else 'trivial'
    alg = 'ED' if min_x == 6 else 'MPS'
    
    ax.set_title(f"Lowest energies for {mode} sites, {alg}")
    ax.set_xlabel("Sites number")
    ax.set_xticks(range(len(db_energies)), [min_x + 2 * ii for ii in range(len(db_energies)) ])
    ax.set_ylabel("Normalized Energy")
    # ax.set_ylim([np.min([*db_energies]) - 0.0005, 1.0005])
    ax.set_xlim([- 1, len(db_energies)])
    # ax.set_yscale('log')

    ax.legend()
    plt.show()

def ent_spectr(ed_sizes, mps_sizes):
    # Entanglement spectrum
    db_spectr = []
    for orig, size in zip(
        [*['DATA_ED'] * len(ed_sizes), *['DATA_MPS'] * len(mps_sizes)],
        [*ed_sizes, *mps_sizes]
    ):
        for mode in ['topo', 'triv']:
            if not os.path.exists(f'DATA_SERVER/{orig}/data_{size}_{mode}_paper'):
                continue

            spectrum = np.loadtxt(
                f'DATA_SERVER/{orig}/data_{size}_{mode}_paper' +
                    f"/EntEntropy{size}_paper_topoTrue.data",
                delimiter=',') if mode == 'topo' else np.loadtxt(
                f'DATA_SERVER/{orig}/data_{size}_{mode}_paper' +
                    f"/EntEntropy{size}_paper_topoFalse.data",
                delimiter=',')
            
            if orig == 'DATA_MPS':
                spectrum *= np.log2(np.exp(1.))
            # dump it
            db_spectr.append(spectrum)

    # now count the entries
    degeneracies = [Counter(spectrum) for spectrum in db_spectr]

    fig, ax = plt.subplots()

    ax.eventplot(
        db_spectr,
        orientation='vertical',
        linelengths=0.8,
        colors=['blue', 'red'] * len([*ed_sizes, * mps_sizes])
    )

    # Annotate degeneracies
    for jj, degeneracy in enumerate(degeneracies):
        for y_val, count in degeneracy.items():
            if count > 1:  # Only annotate if degenerate (optional)
                ax.text(jj, y_val, f'{count}',
                        va='center', ha='center', fontsize=9, color='black')

    ax.set_title('Entanglement spectrum for different chain sizes')
    ax.set_xlabel('Number of sites')
    ax.set_ylabel('Ent. entropy')
    ax.set_yticks([])
    ax.set_xticks(
        list(range(len(db_spectr))),
        [*np.repeat(ed_sizes, 2), *np.repeat(mps_sizes, 2)]
    )
    plt.show()
    print('gg')


ent_spectr([6, 8, 10, 12, 14], [16, 20, 22, 24, 28, 32])