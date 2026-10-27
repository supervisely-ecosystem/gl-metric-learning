"""Offline tests for loading embedding .pkl files (no Supervisely server needed).

Run from the repo root inside supervisely/base-py-sdk:6.35.0 (Python 3.8):
    python -m unittest discover -s supervisely/similarity-calculator/tests -v
APP_SRC can point at another copy of src/ (e.g. the pre-fix code).
"""
import ast
import json
import logging
import os
import pickle
import sys
import tempfile
import types
import unittest

import numpy as np

SRC = os.environ.get(
    "APP_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)


def _import_sly_functions():
    # sly_globals talks to the server at import time and torch is not in the base
    # image, so replace them with stubs; the rest of sly_functions is imported as is.
    progress = types.ModuleType("sly_progress")

    class SlyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def refresh_params(self, *args, **kwargs):
            pass

        def next_step(self, *args, **kwargs):
            pass

        def reset_params(self, *args, **kwargs):
            pass

    progress.SlyProgress = SlyProgress
    globals_stub = types.ModuleType("sly_globals")
    globals_stub.api = None
    globals_stub.task_id = 0
    globals_stub.logger = logging.getLogger("test")
    sys.modules["sly_progress"] = progress
    sys.modules["sly_globals"] = globals_stub
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.path.insert(0, os.path.abspath(SRC))
    import sly_functions

    return sly_functions


f = _import_sly_functions()


class _RunsCommand:
    def __init__(self, command):
        self.command = command

    def __reduce__(self):
        return (os.system, (self.command,))


class _Exec:
    def __init__(self, code):
        self.code = code

    def __reduce__(self):
        return (exec, (self.code,))


def _calculator_style_payload():
    """Same data path as calculator/src/functions.py: serve -> str -> literal_eval -> pack_data."""
    embeddings = np.random.rand(2, 4).astype(np.float32)
    served = json.dumps(str([{"index": i, "embedding": list(e)} for i, e in enumerate(embeddings)]))
    by_index = {item["index"]: item["embedding"] for item in ast.literal_eval(json.loads(served))}
    return {
        "cola": [
            {"url": "http://x/1.jpg", "embedding": by_index[0], "bbox": [0, 0, 9, 9],
             "description": {"sku": "1"}},
            {"url": "http://x/2.jpg", "embedding": None, "bbox": [0, 0, 0, 0], "description": {}},
        ],
        None: [
            {"url": "http://x/3.jpg", "embedding": by_index[1], "bbox": None, "description": {}},
        ],
    }


def _numpy_payload():
    """Older files kept numpy arrays/scalars instead of lists of floats."""
    return {
        "pepsi": [
            {"url": np.str_("http://x/4.jpg"), "embedding": np.arange(4, dtype=np.float32),
             "bbox": [np.int64(1), 2, 3, 4], "description": {"w": np.float64(0.5)}},
        ]
    }


class EmbeddingsLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.marker = os.path.join(self.tmp, "PWNED")

    def _write(self, name, obj, protocol=pickle.DEFAULT_PROTOCOL):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=protocol)
        return path

    def test_legitimate_calculator_file_loads(self):
        payload = _calculator_style_payload()
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            path = self._write("ok_{}.pkl".format(protocol), payload, protocol)
            emb, ph = f.load_embeddings_to_memory([path])
            self.assertEqual(emb["label"], ["cola", None])
            self.assertEqual(emb["embedding"][0], payload["cola"][0]["embedding"])
            self.assertEqual(emb["url"], ["http://x/1.jpg", "http://x/3.jpg"])
            self.assertEqual(ph["label"], ["cola"])
            self.assertIsNone(ph["embedding"][0])

    def test_legitimate_numpy_file_loads(self):
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            path = self._write("np_{}.pkl".format(protocol), _numpy_payload(), protocol)
            emb, _ = f.load_embeddings_to_memory([path])
            np.testing.assert_array_equal(emb["embedding"][0], np.arange(4, dtype=np.float32))
            self.assertEqual(emb["description"][0]["w"], 0.5)

    def test_os_system_payload_is_not_executed(self):
        evil = self._write("evil.pkl", {"x": [{"url": _RunsCommand("touch " + self.marker),
                                               "embedding": [1.0], "bbox": None, "description": {}}]})
        good = self._write("good.pkl", _calculator_style_payload())
        try:
            emb, _ = f.load_embeddings_to_memory([evil, good])
        except pickle.UnpicklingError:
            emb = None
        self.assertFalse(os.path.exists(self.marker), "pickle payload executed os.system")
        if emb is not None:  # the legitimate file still loads, the crafted one is skipped
            self.assertEqual(emb["label"], ["cola", None])

    def test_exec_payload_is_not_executed(self):
        code = "open({!r}, 'w').write('x')".format(self.marker)
        evil = self._write("evil2.pkl", {"x": [{"url": "u", "embedding": [1.0], "bbox": None,
                                                "description": _Exec(code)}]})
        try:
            f.load_embeddings_to_memory([evil])
        except pickle.UnpicklingError:
            pass
        self.assertFalse(os.path.exists(self.marker), "pickle payload executed exec()")


if __name__ == "__main__":
    unittest.main()
