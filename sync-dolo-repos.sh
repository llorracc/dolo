#!/bin/bash

EF="/Volumes/Sync/GitHub/EconForge/dolo"
LL="/Volumes/Sync/GitHub/llorracc/dolo"

# Create temporary file for rsync patterns
TMPFILE=$(mktemp)

# Add specific patterns for the files we want
cat > "$TMPFILE" << EOL
# First exclude git and specstory
- .git/
- .git/**
- .specstory/
- .specstory/**

# Include directories we need
+ /examples/
+ /examples/notebooks/
+ /examples/models/
+ /dolo/
+ /dolo/compiler/

# Include specific files and patterns
+ *.py
+ *.yaml
+ *.md
+ *.rst
+ *.txt
+ *.ipynb

# Exclude specific patterns
- *.png
- *.jpg
- *.jpeg
- *.gif
- *.ico
- *.exe
- *.dll
- *.so
- *.dylib
- *.bin
- __pycache__/
- __pycache__/**
- *.pyc
- dolo.egg-info/
- dolo.egg-info/**
- models_with_errors/
- models_with_errors/**

# Exclude everything else
- *
EOL

echo "Using patterns:"
cat "$TMPFILE"

# First do a dry run and filter output to show only files
echo
echo "Files that will be modified:"
echo "--------------------------"
rsync -nav --itemize-changes --include-from="$TMPFILE" "$EF/" "$LL/" | grep -v '^.d' | grep -v 'sending incremental file list' | grep -v '^$' | grep -v '/$' | sed 's/^.*] //'

# Ask for confirmation
echo
echo "Above is the list of files that will be copied/modified."
read -p "Do you want to proceed with the actual copy? (y/n) " answer

if [[ $answer == "y" ]]; then
    echo "Executing rsync..."
    rsync -av --include-from="$TMPFILE" "$EF/" "$LL/"
    
    # Remove all .DS_Store files
    echo "Removing .DS_Store files..."
    find "$LL" -name ".DS_Store" -delete
    
    # Remove all .ipynb_checkpoints directories
    echo "Removing .ipynb_checkpoints directories..."
    find "$LL" -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
    
    # Create notebooksMDP directory if it doesn't exist
    echo "Creating notebooksMDP directory if needed..."
    mkdir -p "$LL/examples/notebooksMDP"
    
    # Move Python files from notebooks to notebooksMDP (only in LL)
    echo "Moving Python files to notebooksMDP in target directory..."
    cd "$LL/examples/notebooks" && \
    for pyfile in *.py; do
        if [ -f "$pyfile" ]; then
            echo "Moving $pyfile"
            mv "$pyfile" "../notebooksMDP/"
        fi
    done
    cd - > /dev/null
    
    # Report what was moved to notebooksMDP
    echo "Files now in notebooksMDP:"
    ls -l "$LL/examples/notebooksMDP"
    
    # Create modelsMDP directory if it doesn't exist
    echo "Creating modelsMDP directory if needed..."
    mkdir -p "$LL/examples/modelsMDP"
    
    # Move MDP model files (only in LL)
    echo "Moving MDP model files in target directory..."
    cd "$LL/examples/models" && \
    for mdpfile in *_mdp*; do
        if [ -f "$mdpfile" ]; then
            echo "Moving $mdpfile"
            mv "$mdpfile" "../modelsMDP/"
        fi
    done
    cd - > /dev/null
    
    # Report what was moved to modelsMDP
    echo "Files now in modelsMDP:"
    ls -l "$LL/examples/modelsMDP"
else
    echo "Operation cancelled."
fi

# Clean up
rm "$TMPFILE" 