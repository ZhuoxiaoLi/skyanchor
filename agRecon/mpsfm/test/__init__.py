import importlib.util

from mpsfm.utils.tools import get_class

from .basetest import BaseTest


def get_test(name):
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            obj = get_class(path, BaseTest)
            if obj is not None:
                return obj

    raise RuntimeError(f'Test {name} not found in any of [{" ".join(import_paths)}]')
