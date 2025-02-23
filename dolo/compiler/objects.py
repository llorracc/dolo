from dolo.numeric.distribution import *

#
from dataclasses import dataclass
from dolang.language import language_element          # DSL decorator for model objects (snt3p5)

# not sure we'll keep that
import numpy as np
from typing import List, Union

Scalar = Union[int, float]

# not really a language element though
# @language_element


class Domain:
    pass


class CartesianDomain(Domain, dict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, w in kwargs.items():
            v = kwargs[k]
            self[k] = np.array(v, dtype=float)       # Store bounds as float arrays (snt3p5)

    def discretize(self, n=None):
        if n == None:
            n = [10] * (len(self.min))               # Default 10 points per dimension (snt3p5)
        from dolo.numeric.grids import UniformCartesianGrid

        return UniformCartesianGrid(self.min, self.max, n)  # Create state space grid (snt3p5)

    @property
    def states(self):
        l = tuple(self.keys())                       # Get state var names (snt3p5)
        return l

    @property
    def min(self):
        return np.array([self[e][0] for e in self.states])  # Lower bounds vector (snt3p5)

    @property
    def max(self):
        return np.array([self[e][1] for e in self.states])  # Upper bounds vector (snt3p5)


# these are dummy objects so far
#
# @language_element
# @dataclass
# class UNormal:
#     mu: float
#     sigma: float
#     signature = {'mu': 'float', 'sigma': 'float'}
#


# @language_element
# @dataclass
# class MvNormal:
#     Mu: List[float]
#     Sigma: List[List[float]]
#     signature = {'Mu': 'list(float)', 'Sigma': 'Matrix'}


# %%


@language_element
class Conditional:

    signature = {"condition": None, "type": None, "arguments": None}  # DSL type spec (snt3p5)

    def __init__(self, condition, type, arguments):
        self.condition = condition                    # Branch condition (snt3p5)
        self.type = type                             # Distribution type (snt3p5)
        self.arguments = arguments                    # Dist params (snt3p5)


@language_element
class Product:
    def __init__(self, *args: List):
        self.factors = args                          # Store product components (snt3p5)


@language_element
def Matrix(*lines):
    mat = np.array(lines, np.float64)                # Convert to float matrix (snt3p5)
    assert mat.ndim == 2
    return mat


@language_element
def Vector(*elements):
    mat = np.array(elements, np.float64)             # Convert to float vector (snt3p5)
    assert mat.ndim == 1
    return mat
