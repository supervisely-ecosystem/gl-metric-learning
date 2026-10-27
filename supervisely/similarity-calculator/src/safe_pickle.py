"""Load embedding .pkl files without letting them run code.

The files under /GL-MetricLearning/embeddings/ in Team Files are written by the
"Calculate Embeddings" app (supervisely/calculator) and hold only plain data:
{tag: [{'url': str, 'embedding': [float, ...] | None, 'bbox': [int, ...] | None,
'description': dict}, ...]}. Older files may also hold numpy arrays and scalars.

Any team member can put a file into that folder, and pickle.load() runs whatever
code a crafted file asks for. This unpickler only resolves the few globals that
such data needs (numpy array/scalar/dtype reconstruction and a couple of builtin
containers); everything else is rejected before it can be called.
"""
import pickle

_ALLOWED_GLOBALS = {
    # numpy arrays, scalars and dtypes (numpy 1.x module names)
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    # the same under numpy 2.x module names
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    # containers that are not native pickle opcodes in every protocol
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("collections", "OrderedDict"),
    # protocol 0-2 store bytes (e.g. numpy array data) as _codecs.encode(str, 'latin1')
    ("_codecs", "encode"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                "global '{}.{}' is not allowed in an embeddings file".format(module, name)
            )
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ImportError:  # file written with numpy 2, read with numpy 1
                module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load(file_obj):
    """Drop-in replacement for pickle.load() for embedding files."""
    return RestrictedUnpickler(file_obj).load()
