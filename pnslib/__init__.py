"""PnSLib.
Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import os

HOME = os.environ["HOMEPATH"]
PNSLIB_ROOT = os.path.join(HOME, ".pnslibres")
PNSLIB_DATA = os.path.join(PNSLIB_ROOT, "data")

if not os.path.isdir(PNSLIB_DATA):
    os.makedirs(PNSLIB_DATA)
