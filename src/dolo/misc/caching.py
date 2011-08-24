import functools

class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
         self.func = func
         self.cache = {}

    def __call__(self, *args):

        args = (e for e in args)
        try:
            return self.cache[args]
        except KeyError:
            value = self.func(*args)
            self.cache[args] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)
    def __repr__(self):
         """Return the function's docstring."""
         return self.func.__doc__
    def __get__(self, obj, objtype):
         """Support instance methods."""
         return functools.partial(self.__call__, obj)


class cachedondisk(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):
         self.func = func
         self.fname = func.__name__
        
    def __call__(self, *args):
        import pickle
        hh = tuple(  hashable(e) for e in args )
        h = hash(hh)
        try:
            with file('.cache.{0}.{1}.pickle'.format(self.fname,h)) as f:
                value = pickle.load(f)
            return value
        except IOError:
            value = self.func(*args)
            # write file with h
            with file('.cache.{0}.{1}.pickle'.format(self.fname,h),'w') as f:
                pickle.dump(value,f)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)
    def __repr__(self):
         """Return the function's docstring."""
         return self.func.__doc__
    def __get__(self, obj, objtype):
         """Support instance methods."""
         return functools.partial(self.__call__, obj)

def clear_cache():

    import os
    
    try:
        os.system('rm .cache.*.pickle')
    except:
        pass


import collections


def hashable(obj):
    if hasattr(obj,'flatten'): # for numpy arrays
        return tuple( obj.flatten().tolist() )
    if isinstance(obj, collections.Hashable):
        return obj
    if isinstance(obj, collections.Mapping):
        items = [(k,hashable(v)) for (k,v) in obj.items()]
        return frozenset(items)
    if isinstance(obj, collections.Iterable):
        return tuple([hashable(item) for item in obj])
    return TypeError(type(obj))