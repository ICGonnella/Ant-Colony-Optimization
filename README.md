# Directories

[a relative link](./ACO)

### ACO
see the internal `README.md` file ( implementation of the 
[Ant-Colony-Optimization methods](https://github.com/ICGonnella/Ant-Colony-Optimization/tree/master/ACO)

### MLACO
empty directory that has to be replaced by the clone of
the github [ MLACO repository](https://github.com/yuansuny/MLACO):

type: git clone https://github.com/yuansuny/MLACO.git

### py2opt_local
code that implements the 2-opt local search, taken from
the [py2opt repository](https://github.com/pdrm83/py2opt) (with minor changes in the results' 
visualization w.r.t. the original implementation)

# .py files

### Orienteering Problem solved with Gurobi tools
python file to solve a chosen OP with Gurobi
optimization tools.
The OP can be chosen either choosing a dimensionality
and generating a DataFrame of random points with random
gains (with the function `generate_dataset()` of the 
`data_preprocessing.py` file of the ACO package), or loading
a previously generated OP problem of size 50 or 100 from
the github 
[MLACO repository](https://github.com/yuansuny/MLACO/tree/main/Datasets), 
with the function `load_problem()`
of the `data_preprocessing.py` file of the ACO package.

### Orienteering Problem solved with ACO method
python file to solve a chosen OP with ACO methods.
It explains how to run all the possible configurations of ACO
algorithms implemented in the ACO package.
The OP can be chosen in the same way explained for the
Orientiring Problem solved with Gurobi tools file

# Others

### train_data.csv
.csv training data for the ML model 

### trained_lr.sav
Logistic Regressor trained with the `train_data.csv` set

### trained_svm.sav
SVM model trained with the `train_data.csv` set

