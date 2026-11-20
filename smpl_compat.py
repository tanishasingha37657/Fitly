"""
Compatibility shim for using the old `chumpy`/`smplx` SMPL loading pipeline
on modern Python (3.11+) and NumPy (1.24+). Import this before anything
that imports smplx or chumpy.
"""
import inspect
from collections import namedtuple
import numpy as np

# inspect.getargspec() was removed in Python 3.11; chumpy still calls it.
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

# np.bool, np.int, np.float, etc. were removed in NumPy 1.24+; chumpy still
# imports them directly from the numpy namespace.
_np_aliases = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
    "nan": np.nan,
    "inf": np.inf,
}
for _name, _val in _np_aliases.items():
    if not hasattr(np, _name):
        setattr(np, _name, _val)