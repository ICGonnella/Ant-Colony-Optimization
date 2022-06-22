# ACO (Ant-Colony-Optimization)
Ant-Colony-Optimization technique is purely inspired from the 
foraging behaviour of ant colonies (ants are social insects that 
communicate with each other using pheromone).
The underlying principle of ACO is to observe and reproduce their 
movement from nests to possible food sources, which is usually done
finding the shortest possible path.
Initially, ants start to move randomly in search of food around 
their nests. This randomized search opens up multiple routes from 
the nest to the food source. Then, based on the quality and quantity of the food, ants carry a 
portion of the food back, leaving necessary pheromone concentration 
on its return path. 
Therefore, the probability that the following ants will select 
a specific path will depend on these pheromone trials left by the previous
ones.

The best time-performance could be obtained parallelizing the code, 
which has not been done in the context of this project.

## Implementation
Here, the `AS` (Ant System) and `MMAS` (Max-Min Ant System) methods 
have been implemented. 

### ant.py
Class that implements an Ant object. 
It contains the methods that allow a single ant to create a path from 
a starting point to an ending one according to a certain probability 
distribution, depending on its current position and on the 
paths previously built by the other ants of the Ant Colony.

### population.py
Class that implements the Population of the Ant Colony, made up by
a certain number of Ant objects.
It contains the methods that implement the AS and MMAS techniques, 
also with different internal configurations. 
Indeed, the `ML_` parameter of the Population class controls whether
or not, and how, a Machine Learning model can be used to improve the
ACO algorithms performance. 
According to the ["Boosting ant colony optimization via solution prediction and machine learning"](https://www.sciencedirect.com/science/article/abs/pii/S0305054822000636) paper, 
the three ways in which this can be done
are executed by setting `ML_ = 1` (ACO_\ni) , `ML_ = 2` (ACO_\hat{\ni}) and
`ML_ = 3` (ACO_\tau), while `ML_ = 0` executes the "traditional" 
ACO algorithms. 
Another parameter, `LS_` , controls whether or not a 2-opt local search is
performed each time an Ant discovers a new path. 

In addition, `MMAS` algorithm can be performed in two ways: 
by letting the best ant at each iteration
update the `tau_matrix`, or by letting the "global" best ant 
(considering all the iterations done so far) update
the `tau_matrix`. This is done by specifying
`global_ = False` in the first case, and 
`global_ = True` in the second one.

The Population class also implements methods to compute the five
different features serving in input the ML model, in case `ML_ != 0`.

### t_set.py
Useful functions for generating the training and testing sets for the
ML model

### data_preprocessing.py
Useful functions for generating an Orientiring Problem of a certain 
dimensionality `n`, or loading a OP problem of 50 or 100 dimension from
the github [repository](https://github.com/yuansuny/MLACO)

### ml.py
Useful functions to obtain the probability matrix derived from the ML
predictions.