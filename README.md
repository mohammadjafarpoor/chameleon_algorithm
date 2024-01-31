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
2. If you're still having issues with `METIS`, just run this code to solve it:
```
import requests
import tarfile

# Download and extract the file
url = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path=".")

# Change working directory
%cd metis-5.1.0

# The remaining steps
!make config shared=1 prefix=~/.local/
!make install
!cp ~/.local/lib/libmetis.so /usr/lib/libmetis.so
!export METIS_DLL=/usr/lib/libmetis.so
!pip3 install metis-python
```
3. Append the cloned file's directory to your python path
```
import sys
sys.path.append(<Library PATH>)
```

4. Run sample code

```
python -i main.py
```

## References

[1] Karypis, George, Eui-Hong Han, and Vipin Kumar. "Chameleon: Hierarchical clustering using dynamic modeling." *Computer* 32.8 (1999): 68-75.
http://ieeexplore.ieee.org/abstract/document/781637/
