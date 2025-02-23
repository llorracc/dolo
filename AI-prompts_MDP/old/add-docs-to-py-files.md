<!-- Existing header or metadata, unchanged... -->

# Instructions for Adding Documentation to Python Codebase Without Changing Functionality

## Core Principle
This task is DOCUMENTATION-ONLY. You are acting as a documentation writer, not a code improver. Your role is to explain the code exactly as it exists, not to improve it.

## Preparation 
1. Read and understand the documentation in `docs/` and examples in `examples/`
2. Study how the codebase accomplishes its documented tasks
3. Note any references to external papers, algorithms, or methods

## Files to Document
- All ".py" files EXCEPT those listed in "Files to Ignore" below

## Files to Ignore
- serial_operations.py
- taylor_expansion.py
- tensor.py
- linter.py

** Folders to ignore **
- dolo/algos/misc/
- dolo/algos/numeric/discretization/
- dolo/algos/numeric/extern/
- dolo/algos/optimize/
- tests/
- experiments/

## Documentation Rules

### ALLOWED - You may ONLY:
1. ADD inline comments to uncommented lines using `# comment` at end of line
2. ADD docstrings below function/class definitions that lack them
3. ADD references to existing docs/examples
4. APPEND to existing docstrings (without modifying existing content)

### FORBIDDEN - You must NEVER:
1. DELETE any code, including:
   - Variable assignments
   - Function calls
   - Class attributes
   - Import statements
   - Whitespace
   - Blank lines
2. MODIFY any code, including:
   - Variable names
   - Function names
   - Class names
   - Parameter names
   - Attribute names
   - Import statements
   - Logic flow
3. CHANGE any existing comments or docstrings

## Required Process

1. Before ANY edit:
   - Read the ENTIRE file
   - List every line you plan to modify
   - Verify each modification is ONLY adding comments/docstrings
   - If unsure, DO NOT make the edit

2. For each file:
   PHASE 1: Inline Comments
   - Add ONLY `# comment` to explain unclear lines
   - Leave obvious lines uncommented
   - Never remove or modify existing code

   PHASE 2: Docstrings
   - Do not delete any existing docstring text
     - You can append to the docstring if that would be appropriate
   - Follow numpy docstring style
   - Include relevant references

## Step-by-Step Process
0. **Preparation**
    - Follow the instructions in the Preparation section above.
1. **Inline Comments**  
   - For each line lacking a comment, and for which a comment is appropriate, append a *short* inline comment explaining what the line does.  Do not add comments to lines whose entire purpose is obvious from the code.
     - The "entire purpose" of an import statement is not obvious because the reader does not know what the module is for.
   - Retain existing comments unchanged.  
   - **Do not add** or remove lines, reorder lines, or perform ANY code modifications.
2. After EVERY edit:
   - Compare original and new files line-by-line
   - Count code lines (excluding comments) - must match exactly
   - Verify every variable/attribute name remains identical
   - Check that only comments/docstrings were added
   - If any check fails, revert and try again
3. Final Verification:
   - Run any existing tests
   - Verify file still parses correctly
   - Check all functionality remains unchanged relative to original
   - No line of the original code is removed or altered. No new functional code is introduced.

After you have completed Phase 1, you confirm to the user that comments have been added to all the appropriate lines, and that no functioning has been changed

Phase 2: Docstrings
4. **Revise As Appropriate**
    - You should learn details about the codebase as you write the comments in Phase 1.
    - That means you should understand it better after you have finished than you did before you started.
    - You will use this new understanding to revise create or add to docstrings in Phase 2.
    - Again, do not change any lines of code, and do not remove any comments that were in the original code.
    - For any object that already has a docstring, DO NOT DELETE that docstring.  
      - You can append to it if extra explanation is needed. 
      - You can also add references to other documentation or examples that are relevant to the object.
      
5. **Keep Revising Until Satisfactory**
    - You should keep revising the documentation until you are satisfied that it is accurate, complete, and clear.
    - You should also check that the code is still working as intended.
    - If you find any errors or omissions, you should fix them.
    - If you find any parts that are not clear, you should revise them until they are clear.
    
6. **Report Back On Anything You Did Not Understand**
    - If you did not understand something, or you think there is something important you did not document, report it back.

## When in Doubt
- If unsure whether an edit is allowed, DO NOT make it
- Ask for clarification rather than risk changing functionality
- Remember: Your job is to explain the code, not improve it

---
