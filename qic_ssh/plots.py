import json
import os.path
import matplotlib.pyplot as plt
import numpy as np

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

    plt.show()
    # fig.savefig(f"E_to_num.pdf")

def obc_effect(topological: bool, paper: bool):
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

    for size in range(16):
        data_path = f"DATA_SERVER/data_{size}_{version}_{model}"
        if not os.path.exists(data_path):
            continue
        
        # read in the eigenvalues of the problem
        energies = np.loadtxt(data_path + f"/Eigvals{size}_{model}_topo{topological}.data",
            delimiter=',')
        # save first 4 values 
        db_energies.append(np.abs(energies[0:4]) / np.max(np.abs(energies[0:4])))
    
    # plot the 4 lowest energies
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    plt.eventplot(
        db_energies,
        orientation='vertical',
        linelengths=0.8,
        colors=[f"C{ii}" for ii in range(len(db_energies))]
    )

    if topological:
        ax.set_title("Lowest energies for topological sites")
    else:
        ax.set_title("Lowest energies for trivial sites")
    ax.set_xlabel("Sites number")
    ax.set_xticks(range(len(db_energies)), [4 + 2 * ii for ii in range(len(db_energies)) ])
    ax.set_ylabel("Normalized Energy")
    # ax.set_ylim([0.95, 1])
    # ax.set_yscale('log')

    plt.show()
