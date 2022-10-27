import json

NUMPY_INTS = [
    "int_",
    "intc",
    "intp",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
NUMPY_FLOATS = ["float_", "float16", "float32", "float64"]
NUMPY_SCALARS = NUMPY_INTS + NUMPY_FLOATS


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if o.__class__.__module__ == "numpy":
            if o.__class__.__name__ in NUMPY_INTS:
                return int(o)
            if o.__class__.__name__ in NUMPY_FLOATS:
                return float(o)
        return super().default(o)
