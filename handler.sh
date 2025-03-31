#! /bin/bash
# Handler for different simulations - QIAC exam

source "../bin/activate"
# parameters
# declare -a sizes=("4" "6" "8" "10" "12" "14")
declare -a sizes=("4")

for size in "${sizes[@]}"
do
    # Start topological + ideal
    rm -r "data_${size}_topo_ideal"
    mkdir "data_${size}_topo_ideal"

    python "main.py" --topological True --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_topo_ideal"

    # Start trivial + ideal
    rm -r "data_${size}_triv_ideal"
    mkdir "data_${size}_triv_ideal"

    python "main.py" --topological False --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_triv_ideal"

    # Start topological + paper
    rm -r "data_${size}_topo_paper"
    mkdir "data_${size}_topo_paper"

    python "main.py" --topological True --n_sites "$size" --model "paper"
    mv *.data "data_${size}_topo_paper"

    # Start trivial + paper
    rm -r "data_${size}_triv_paper"
    mkdir "data_${size}_triv_paper"

    python "main.py" --topological False --n_sites "$size" --model "paper"
    mv *.data "data_${size}_triv_paper"
done