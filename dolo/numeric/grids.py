from functools import reduce
from operator import mul
from quantecon import cartesian
import numpy as np
from numpy import zeros

from typing import TypeVar, Generic, Dict

T = TypeVar("T")
S = TypeVar("S")


def prod(l):
    # Compute product of elements in list l (snt3p5)
    return reduce(mul, l, 1.0)


from dolo.numeric.misc import mlinspace


class Grid:
    # Base class for all grid types with common operations (snt3p5)

    def __mul__(self, rgrid):
        # Multiply two grids to create a product grid (snt3p5)
        return cat_grids(self, rgrid)

    @property
    def nodes(self):
        # Get array of all grid nodes (snt3p5)
        return self.__nodes__

    @property
    def n_nodes(self):
        # Get total number of nodes in grid (snt3p5)
        return self.__nodes__.shape[0]

    def node(self, i):
        # Get coordinates of i-th node (snt3p5)
        return self.__nodes__[i, :]


class ProductGrid(Grid, Generic[T, S]):
    # Grid representing product of two grids with optional dimension names (snt3p5)

    def __init__(self, g1: T, g2: S, names=None):
        # Initialize with two grids and optional dimension names (snt3p5)
        self.grids = [g1, g2]
        self.names = names

    def __getitem__(self, v):
        # Get grid by dimension name (snt3p5)
        return self.grids[self.names.index(v)]

    def __repr__(self):
        # String representation showing product of grids (snt3p5)
        return str.join(" × ", [e.__repr__() for e in self.grids])


class EmptyGrid(Grid):
    # Grid with no nodes, used as placeholder or base case (snt3p5)

    type = "empty"

    @property
    def nodes(self):
        # Empty grid has no nodes (snt3p5)
        return None

    @property
    def n_nodes(self):
        # Empty grid has zero nodes (snt3p5)
        return 0

    def node(self, i):
        # Empty grid has no nodes to return (snt3p5)
        return None

    def __add__(self, g):
        # Adding to empty grid returns the other grid (snt3p5)
        return g


class PointGrid(Grid):
    # Grid consisting of a single point (snt3p5)

    type = "point"

    def __init__(self, point):
        # Initialize with coordinates of the point (snt3p5)
        self.point = np.array(point)

    @property
    def nodes(self):
        # Point grid has no node array (snt3p5)
        return None

    @property
    def n_nodes(self):
        # Point grid has one node (snt3p5)
        return 1

    def node(self, i):
        # Point grid has no indexed nodes (snt3p5)
        return None


class UnstructuredGrid(Grid):
    # Grid with arbitrary node positions (snt3p5)

    type = "unstructured"

    def __init__(self, nodes):
        # Initialize with array of node coordinates (snt3p5)
        nodes = np.array(nodes, dtype=float)
        self.min = nodes.min(axis=0)  # Minimum coordinates in each dimension (snt3p5)
        self.max = nodes.max(axis=0)  # Maximum coordinates in each dimension (snt3p5)
        self.__nodes__ = nodes
        self.d = len(self.min)  # Number of dimensions (snt3p5)


class CartesianGrid(Grid):
    # Base class for Cartesian product grids (snt3p5)
    pass


class UniformCartesianGrid(CartesianGrid):
    # Cartesian grid with uniform spacing in each dimension (snt3p5)

    type = "UniformCartesian"

    def __init__(self, min, max, n=[]):
        # Initialize with min/max bounds and number of points per dimension (snt3p5)
        self.d = len(min)  # Number of dimensions (snt3p5)

        # Store bounds as numpy arrays (snt3p5)

        # this should be a tuple
        self.min = np.array(min, dtype=float)
        self.max = np.array(max, dtype=float)
        if len(n) == 0:
            self.n = np.zeros(n, dtype=int) + 20  # Default 20 points per dimension (snt3p5)
        else:
            self.n = np.array(n, dtype=int)

        # Generate uniformly spaced nodes (snt3p5)
        self.__nodes__ = mlinspace(self.min, self.max, self.n)

    # def node(i:)
    # pass

    def __add__(self, g):
        # Add two uniform Cartesian grids by concatenating dimensions (snt3p5)
        if not isinstance(g, UniformCartesianGrid):
            raise Exception("Not implemented.")

        n = np.array(tuple(self.n) + tuple(g.n))
        min = np.array(tuple(self.min) + tuple(self.min))
        max = np.array(tuple(self.max) + tuple(self.max))

        return UniformCartesianGrid(min, max, n)

    def __numba_repr__(self):
        # Get representation for Numba compilation (min, max, n) for each dimension (snt3p5)
        return tuple([(self.min[i], self.max[i], self.n[i]) for i in range(self.d)])


class NonUniformCartesianGrid(CartesianGrid):
    # Cartesian grid with arbitrary spacing in each dimension (snt3p5)

    type = "NonUniformCartesian"

    def __init__(self, list_of_nodes):
        # Initialize with list of node positions for each dimension (snt3p5)
        list_of_nodes = [np.array(l) for l in list_of_nodes]
        self.min = [min(l) for l in list_of_nodes]  # Minimum coordinates (snt3p5)
        self.max = [max(l) for l in list_of_nodes]  # Maximum coordinates (snt3p5)
        self.n = np.array([(len(e)) for e in list_of_nodes])  # Points per dimension (snt3p5)
        # this should be done only on request.
        self.__nodes__ = cartesian(list_of_nodes)  # Generate Cartesian product nodes (snt3p5)
        self.list_of_nodes = list_of_nodes

    def __add__(self, g):
        # Add two non-uniform Cartesian grids by concatenating dimensions (snt3p5)
        if not isinstance(g, NonUniformCartesianGrid):
            raise Exception("Not implemented.")
        return NonUniformCartesianGrid(self.list_of_nodes + g.list_of_nodes)

    def __numba_repr__(self):
        # Get representation for Numba compilation - list of node arrays (snt3p5)
        return tuple([np.array(e) for e in self.list_of_nodes])


class SmolyakGrid(Grid):
    # Sparse grid using Smolyak algorithm for high-dimensional interpolation (snt3p5)

    type = "Smolyak"

    def __init__(self, min, max, mu=2):
        # Initialize with bounds and sparsity parameter mu (snt3p5)
        from interpolation.smolyak import SmolyakGrid as ISmolyakGrid

        min = np.array(min)
        max = np.array(max)
        self.min = min
        self.max = max
        self.mu = mu  # Controls grid density/sparsity (snt3p5)
        d = len(min)
        sg = ISmolyakGrid(d, mu, lb=min, ub=max)  # Create interpolation grid (snt3p5)
        self.sg = sg
        self.d = d
        self.__nodes__ = sg.grid


def cat_grids(grid_1, grid_2):
    # Concatenate two grids along their dimensions (snt3p5)
    if isinstance(grid_1, EmptyGrid):
        return grid_2
    if isinstance(grid_1, CartesianGrid) and isinstance(grid_2, CartesianGrid):
        min = np.concatenate([grid_1.min, grid_2.min])
        max = np.concatenate([grid_1.max, grid_2.max])
        n = np.concatenate([grid_1.n, grid_2.n])
        return CartesianGrid(min, max, n)
    else:
        raise Exception("Not Implemented.")


# Compatibility functions for old interface (snt3p5)
def node(grid, i):
    # Get i-th node from grid (snt3p5)
    return grid.node(i)


def nodes(grid):
    # Get all nodes from grid (snt3p5)
    return grid.nodes


def n_nodes(grid):
    # Get total number of nodes in grid (snt3p5)
    return grid.n_nodes


if __name__ == "__main__":

    print("Cartsian Grid")
    grid = CartesianGrid([0.1, 0.3], [9, 0.4], [50, 10])
    print(grid.nodes)
    print(nodes(grid))

    print("UnstructuredGrid")
    ugrid = UnstructuredGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Non Uniform CartesianGrid")
    ugrid = NonUniformCartesianGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Smolyak Grid")
    sg = SmolyakGrid([0.1, 0.2], [1.0, 2.0], 2)
    print(nodes(sg))
    print(node(sg, 1))
    print(n_nodes(sg))
