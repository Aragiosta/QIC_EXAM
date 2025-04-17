#! /bin/bash
# Handler for different simulations - QIAC exam

source "../bin/activate"
# parameters
# declare -a sizes=("6" "8" "10" "12" "14" "16")
declare -a sizes=("4")

# ED part
for size in "${sizes[@]}"
do
    # Start topological + ideal
    rm -r "data_${size}_topo_ideal"
    mkdir "data_${size}_topo_ideal"

    python "main_ED.py" --topological True --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_topo_ideal"

    # Start trivial + ideal
    rm -r "data_${size}_triv_ideal"
    mkdir "data_${size}_triv_ideal"

    python "main_ED.py" --topological False --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_triv_ideal"

    # Start topological + paper
    rm -r "data_${size}_topo_paper"
    mkdir "data_${size}_topo_paper"

    python "main_ED.py" --topological True --n_sites "$size" --model "paper"
    mv *.data "data_${size}_topo_paper"

    # Start trivial + paper
    rm -r "data_${size}_triv_paper"
    mkdir "data_${size}_triv_paper"

    python "main_ED.py" --topological False --n_sites "$size" --model "paper"
    mv *.data "data_${size}_triv_paper"
done

rm -r "DATA_ED/"
mkdir DATA_ED
mv data_* DATA_ED/

# declare -a sizes=("16" "22" "28" "36" "44" "52")
declare -a sizes=("16")

# MPS part
for size in "${sizes[@]}"
do
    # Start topological + ideal
    rm -r "data_${size}_topo_ideal"
    mkdir "data_${size}_topo_ideal"

    python "main_MPS.py" --topological True --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_topo_ideal"

    # Start trivial + ideal
    rm -r "data_${size}_triv_ideal"
    mkdir "data_${size}_triv_ideal"

    python "main_MPS.py" --topological False --n_sites "$size" --model "ideal"
    mv *.data "data_${size}_triv_ideal"

    # Start topological + paper
    rm -r "data_${size}_topo_paper"
    mkdir "data_${size}_topo_paper"

    python "main_MPS.py" --topological True --n_sites "$size" --model "paper"
    mv *.data "data_${size}_topo_paper"

    # Start trivial + paper
    rm -r "data_${size}_triv_paper"
    mkdir "data_${size}_triv_paper"

    python "main_MPS.py" --topological False --n_sites "$size" --model "paper"
    mv *.data "data_${size}_triv_paper"
done

rm -r "DATA_MPS/"
mkdir DATA_MPS
mv data_* DATA_MPS/
