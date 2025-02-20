<!-- Existing header or metadata, unchanged... -->

# Instructions for Adding Documentation to python codebase Without Changing Functionality

Below are instructions for adding comments to the code in the codebase. These steps must be followed **without altering** the code's functional or structural elements—**only** documentation (inline comments and docstrings) can be added.

---

## Preparation 

Read, and thoroughly understand, the documentation in `docs/` and the usage examples in `examples/`, with the aim of building insight about the way the elements of the codebase accomplish the tasks they are designed for and documented for.

## Scope of Edits

**Allowed**:
1. Adding inline comments to currently uncommented lines (always use `# ...` at the end of lines, or after existing code).
2. Adding new docstrings immediately below a function or class definition *only if* it lacks a docstring and one seems appropriate.
3. Citing relevant `.md` documentation in `docs/` or usage samples in `examples/` or yaml files in `models/`.

**Forbidden**:
1. Changing any existing comment text (you must leave them exactly as-is).
2. Removing or altering any lines of code (including whitespace or blank lines).
3. Adding or removing imports, renaming variables, or adjusting logic flow.
4. Changing function signatures or how they behave.
5. Modifying any code that controls functionality
6. Changing any existing docstring, except by appending material to the end

---

## Step-by-Step Process
0. **Preparation**
    - Follow the instructions in the Preparation section above.
1. **PHASE 1: Inline Comments**  
   - For each line lacking a comment, append a *short* inline comment explaining what the line does.  
   - Retain existing comments unchanged.  
   - **Do not add** or remove lines, reorder lines, or perform ANY code modifications.

2. **PHASE 2: Docstrings**  
   - Immediately after each function or class definition that does **not** already have a docstring, add a suitable docstring referencing relevant docs.  
   - Provide usage examples by referencing any place in `examples/models/` or `examples/notebooks_py/` that calls the method, if applicable.  
   - As before, do **not** delete or change any line of functional code. Only insert docstrings on lines that do **not** exist (i.e., right below the `def` or `class` statement) so that the **total line count does not change**.

3. **Verification**  
   - Ensure each import statement has a short inline comment describing its purpose.  
   - Ensure each function now has a docstring unless it is self-explanatory.
   - No line of the original code is removed or altered. No new functional code is introduced.

4. **Revise As Appropriate**
    - You should learn details about the codebase as you write these documentation comments.
    - That means you should understand it better after you have finished than you did before you started.
    - With your new understanding, revise the documentation to make it more accurate, complete, and clear.  
    - Again, do not change any lines of code, and do not remove any comments that were in the original code.

5. **Keep Revising Until Satisfactory**
    - You should keep revising the documentation until you are satisfied that it is accurate, complete, and clear.
    - You should also check that the code is still working as intended.
    - If you find any errors or omissions, you should fix them.
    - If you find any parts that are not clear, you should revise them until they are clear.
    
6. **Report Back On Anything You Did Not Understand**
    - If you did not understand something, or you think there is something important you did not document, report it back.

---