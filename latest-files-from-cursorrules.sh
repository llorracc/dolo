#!/bin/bash
cd /Volumes/Sync/Github/EconForge/dolo
# Define the output file
output_file="latest.md"

# Create or clear the output file
> "$output_file"

# Use a while loop with process substitution instead of mapfile
while IFS= read -r file; do
    # Get file extension and clean path
    ext="${file##*.}"
    clean_path="${file#./}"
    
    # Write to output file
    {
        echo "<open_file>"
        echo "\`\`\`$ext:$clean_path"
        cat "$file"
        echo "\`\`\`"
        echo "</open_file>"
        echo ""
    } >> "$output_file"
done < <(find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "*.rst" -o -name "*.txt" \) \
    ! -path "*/.*" \
    ! -name "latest.md" \
    ! -name "*.png" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.gif" \
    ! -name "*.ico" \
    ! -name "*.exe" \
    ! -name "*.dll" \
    ! -name "*.so" \
    ! -name "*.dylib" \
    ! -name "*.bin" \
    ! -name "__init__.py" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    ! -path "*/dolo.egg-info/*" \
    ! -name "*.ipynb" \
    ! -path "*/models_with_errors/*")

echo "Latest contents have been written to $output_file."