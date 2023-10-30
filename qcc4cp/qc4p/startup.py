# You can use this file to initialize an interactive python repl.
# For example, after building, set and adjust to your system:
#
#  export PYTHONPATH=$HOME/qcc:$HOME/qcc/bazel-bin/src/lib
#  export PYTHONSTARTUP=$HOME/qcc/src/lib/startup.py
#
#  python3
#  >> state.zeros(2)
#  State([1.+0.j 0.+0.j 0.+0.j 0.+0.j])

"""Initialize and load all qcc packages."""

# pylint: disable=unused-import
import numpy as np

from . import bell
from . import circuit
from . import helper
from . import ops
from . import state
from . import tensor

print('QCC: ' + __file__ + ': initialized')
