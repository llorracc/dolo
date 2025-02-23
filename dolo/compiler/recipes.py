"""
ideas :
-  recursive blocks           [by default]
- (order left hand side ?)    [by default]
- dependency across blocks
- dummy blocks that are basically substituted everywhere else
"""

import os, yaml, sys

if getattr(sys, "frozen", False):
    # we are running in a |PyInstaller| bundle
    DIR_PATH = sys._MEIPASS                          # Get bundled app path (snt3p5)
else:
    DIR_PATH, this_filename = os.path.split(__file__)

DATA_PATH = os.path.join(DIR_PATH, "recipes.yaml")   # Model spec templates path (snt3p5)

with open(DATA_PATH, "rt", encoding="utf-8") as f:
    recipes = yaml.safe_load(f)                      # Load model type definitions (snt3p5)
