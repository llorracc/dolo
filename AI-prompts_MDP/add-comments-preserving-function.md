# Instructions for Adding Documentation to Python Codebase Without Changing Functionality

CRITICAL WARNING: This is a documentation-only task. The ONLY allowed change is adding single-line comments with '# comment'.
ANY modification to the actual code (including whitespace, parameters, imports, etc.) is STRICTLY FORBIDDEN.

Before starting:
1. Read 'AI-prompts_MDP/files-commented.md' to identify which files have already been processed
2. Skip any files that are already listed in the "Completed Files" section
3. Only process files that are not yet listed as completed

Before submitting any changes:
1. Diff the original and modified files
2. Verify that the ONLY differences are added single-line comments
3. If ANY other changes are found, revert to original and try again

Read the file 'add-comments-to-these-files.md' to get a list of the files you will be changing 

then follow the instructions in the file 'add-preparation.md'

## Core Principle
Add ONLY single-line comments to explain code that isn't immediately obvious to an experienced software engineer.

## When NOT to Comment
Do NOT comment code that is self-explanatory to any experienced Python developer, such as:
1. Standard library imports (e.g., `import copy`, `import os`, `import re`)
2. Basic Python operations:
   ```python
   x = 1
   self.data = data
   name = "John"
   items = []
   for i in range(10):
   if x > 0:
   ```
3. Common programming patterns:
   ```python
   try:
       do_something()
   except Exception as e:
   
   with open(file) as f:
   
   def __init__(self, param):
       self.param = param
   ```

## When to Comment
Add comments ONLY when:
1. Using domain-specific or uncommon libraries:
   ```python
   from dolang.symbolic import parse_string  # Parse model eqn syntax (snt3p5)
   import sympy                              # Symbolic math processing (snt3p5)
   ```
2. Non-obvious business logic or domain concepts:
   ```python
   x = calculate_beta(params)  # Discount factor from utility fn (snt3p5)
   ```
3. Complex algorithms or mathematical operations:
   ```python
   z = x * y + alpha * beta  # Cobb-Douglas production fn (snt3p5)
   ```
4. Non-standard or project-specific patterns:
   ```python
   model.set_changed(all=True)  # Reset all cached values (snt3p5)
   ```

## Documentation Rules

### ALLOWED - You may ONLY:
1. Add single-line comments using `# comment` at end of line
2. Use standard abbreviations:
   ```
   vars      = variables
   params    = parameters
   eqns      = equations
   regex     = regular expression
   init      = initialize
   dict      = dictionary
   calc      = calculate
   eval      = evaluate
   aux       = auxiliary
   config    = configuration
   fn        = function
   expr      = expression
   ```
   and so on.

### FORBIDDEN - You must NEVER:
1. Add, modify, or remove docstrings (comments between triple quotes)
2. Add multi-line comments
3. DELETE any code, including:
   - Variable assignments
   - Function calls
   - Class attributes
   - Import statements
   - Whitespace
   - Blank lines
4. MODIFY any code, including:
   - Variable names
   - Function names
   - Class names
   - Parameter names
   - Attribute names
   - Import statements
   - Logic flow
5. CHANGE or DELETE any existing comments or docstrings

### Comment Alignment Rules
1. Comments within the same code block (same indentation level) MUST align:
   ```python
   def example():
       x = calculate_beta(a, b)     # Compute discount factor (snt3p5)
       y = process_utility(u, v)    # Apply CRRA transform (snt3p5)
       z = get_steady_state(x, y)   # Find equilibrium point (snt3p5)
   ```
2. Exception: If aligning would make any line exceed 120 characters

## DOUBLE-CHECK IT
0. Keep track of the names of any files you have modified in a file named AI-prompts_MDP/files-commented.md
   a. And make a copy in your memory of the original version of the file before it was modified
   b. As a verification step, you will need to compare the original version to your proposed modification
1. When you think you have finished a file:
   - Read the ENTIRE modified file and compare it to the original file
   - Verify each modification is ONLY adding comments
2. For each file:
   - Add ONLY `# comment` to explain non-obvious lines
   - Leave obvious lines uncommented
   - Never remove or modify existing code
   
3. After finishing your commenting of a file, pause and ask the user to review your work

## Final Verification:
- Verify file still parses correctly
- Check all functionality remains unchanged
- No line of the original code is removed
- The only alterations have been adding comments
