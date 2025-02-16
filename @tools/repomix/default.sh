#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0)) ; cd $SCRIPT_DIR
echo "$SCRIPT_DIR"

# SCRIPT_DIR=/Volumes/Sync/GitHub/llorracc/dolo/@tools/repomix
cd "$SCRIPT_DIR/../.."
pwd

repomix --ignore "repomix*,linter.py,*experiments*,mkdocs.yml,.devcontainer,.github,.gitpod.yml,*.pyc,*.toml*,.git,*~"
mv repomix-output.txt @tools/repomix/repomix-all-including-jupyter.txt

cd examples
[[ ! -e notebooks_py ]] && mkdir notebooks_py

cd notebooks
for fnb in *.ipynb; do
    f=${fnb%.*}
    jupyter nbconvert --to script $fnb
    mv $f.py ../notebooks_py
done
cd ../..

repomix --ignore "repomix*,linter.py,*experiments*,mkdocs.yml,.devcontainer,.github,.gitpod.yml,*.pyc,*.toml*,.git,*~,**/*.ipynb,.git"
mv repomix-output.txt @tools/repomix/repomix-all-except-jupyter.txt

repomix --ignore "repomix*,linter.py,*experiments*,mkdocs.yml,.devcontainer,.github,.gitpod.yml,*.pyc,*.toml*,.git,*~,**/*.ipynb,.git,lmmcp.py,perturbation*,dolo/algos/improved_time_iteration.py,dolo/algos/bruteforce_lib.py,*models_*,notebooks,dynare*,linter.py,*experiments*,AI-prompts_MDP/**/*,.cursor/rules/*,distribution.py,perfect_foresight*,perturbation*,misc/*,tests/*,LICENSE,pyproject.toml,REDME.md,*.sh,__init__.py,docs/*,distribution.py,*.rst,discretization*"
mv repomix-output.txt @tools/repomix/repomix-core_nodocs.txt

repomix --include "docs/*,examples/models/*,examples/notebooks_py,examples/models_/*,bin/dolo"
mv repomix-output.txt @tools/repomix/repomix-docs.txt
rm -Rf examples/notebooks_py

