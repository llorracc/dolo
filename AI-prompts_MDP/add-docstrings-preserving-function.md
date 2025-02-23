<!-- Existing header or metadata, unchanged... -->

# Instructions for Adding Docstrings to Python Codebase Without Changing Functionality

## Core Principle
This task is DOCUMENTATION-ONLY. You are acting as a documentation writer, not a code improver. Your role is to explain the code exactly as it exists, not to improve it.

## Documentation Rules

### ALLOWED - You may ONLY:
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
   - Verify each modification is ONLY modifying docstrings
   - If unsure, DO NOT make the edit

2. For each file:
   - Do not delete any existing docstring text
     - You can append to the docstring if that would be appropriate
   - Follow numpy docstring style
   - Include relevant references

## Step-by-Step Process
4. **Revise As Appropriate**
    - You should learn details about the codebase as you write the docstrings
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
