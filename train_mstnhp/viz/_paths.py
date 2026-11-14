"""Keep legacy visualization paths relative to train_mstnhp.

Call before importing project modules. CLI-relative paths and paths in YAML
files use train_mstnhp as their base, regardless of the launch directory.
Absolute paths are unaffected. This deliberately changes the process cwd;
these visualization programs are standalone scripts, not library modules.
"""

import os
from pathlib import Path
import sys


def configure_paths():
    training_dir = Path(__file__).resolve().parents[1]
    repo_dir = training_dir.parent
    # Prefer the simulator beside the training scripts, as before the move.
    for directory in (repo_dir, training_dir):
        value = str(directory)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    os.chdir(training_dir)
    return training_dir
