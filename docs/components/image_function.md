# Image Function

For image functions this version of Lassie relies heavily on machine learning pickers delivered by [SeisBench](https://github.com/seisbench/seisbench).

## PhaseNet Image Function

!!! abstract "Citation PhaseNet"
    *Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018).*

```python exec='on'
from lassie.utils import generate_docs
from lassie.images.phase_net import PhaseNet

print(generate_docs(PhaseNet()))
```
