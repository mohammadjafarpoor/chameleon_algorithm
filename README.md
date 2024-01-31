# chameleon_clustering_algorithm
Python 3.5+ implementation of the clustering algorithm CHAMELEON[1].

This repository is a modification and improvement of https://github.com/Moonpuck/chameleon_cluster.git

Fixed lots of bugs and accorded with the results in the paper.

Depends on METIS for Python.

## Installing
1. Install requirements.

```
pip install networkx
```
```
pip install metis
```
```
pip install tqdm
```
```
pip install numpy
```
```
pip install pandas
```
```
pip install matplotlib
```
```
pip install seaborn
```

2. Append the directory to your python path. replace `<Library PATH>` with the directory of the file you've just cloned.
```
import sys
sys.path.append(<Library PATH>)
```

3. Run sample code

```
python -i main.py
```

## References

[1] Karypis, George, Eui-Hong Han, and Vipin Kumar. "Chameleon: Hierarchical clustering using dynamic modeling." *Computer* 32.8 (1999): 68-75.
http://ieeexplore.ieee.org/abstract/document/781637/
