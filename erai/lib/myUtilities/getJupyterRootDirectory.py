def getJupyterRootDirectory():
    import sys
    import os

    module_path = os.path.abspath(os.path.join('.'))

    return module_path