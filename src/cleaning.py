# src/cleaning.py

"""
Post-processing utilities for HumanEval code completions.

This module converts raw model outputs into strict, harness-compatible
function bodies by:

- Extracting <SOLUTION> blocks (for CoT)
- Removing markdown artifacts
- Removing redundant function definitions
- Removing stray docstrings
- Normalizing indentation relative to the function body

Any failure to produce valid code is surfaced as an empty string,
which is the correct behavior for HumanEval evaluation.
"""

import re
import textwrap
from typing import List


# ---------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------

def cleaner(code: str, entry_point: str, is_cot: bool) -> str:
    """
    Normalize a model-generated completion into a valid HumanEval function body.

    Parameters
    ----------
    code : str
        Raw model output (decoded text).
    entry_point : str
        Name of the target function for this HumanEval problem.
    is_cot : bool
        Whether the model was prompted in CoT mode.

    Returns
    -------
    str
        A properly indented function body, or an empty string on failure.
    """

    # ------------------------------------------------------------
    # A. Extract <SOLUTION> block (CoT only)
    # ------------------------------------------------------------
    if is_cot:
        if "<SOLUTION>" not in code:
            return ""
        code = code.split("<SOLUTION>", 1)[1]
        if "</SOLUTION>" in code:
            code = code.split("</SOLUTION>", 1)[0]

    # ------------------------------------------------------------
    # B. Remove markdown fences (do NOT split; replace safely)
    # ------------------------------------------------------------
    code = code.replace("```python", "").replace("```", "")

    # ------------------------------------------------------------
    # C. Remove redundant function definition
    # ------------------------------------------------------------
    def_pattern = re.compile(
        rf"^\s*def\s+{re.escape(entry_point)}(\s*\(|\s*:)"
    )

    lines: List[str] = []
    for line in code.split("\n"):
        if def_pattern.match(line):
            continue
        # HumanEval already imports typing if needed
        if line.strip().startswith("from typing"):
            continue
        lines.append(line)

    code = "\n".join(lines)

    # ------------------------------------------------------------
    # D. Remove the first docstring block (if any)
    # ------------------------------------------------------------
    code = re.sub(
        r'(\s*("""|\'\'\')[\s\S]*?\2)',
        "",
        code,
        count=1,
    )

    code = code.lstrip("\n\r")

    # ------------------------------------------------------------
    # E. Relative indentation normalization
    # ------------------------------------------------------------
    if not code.strip():
        return ""

    lines = code.split("\n")

    # Find the first real body line (skip imports)
    reference_line = None
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("import ", "from ")):
            reference_line = line
            break

    if reference_line is None:
        reference_line = next((l for l in lines if l.strip()), None)

    if reference_line is None:
        return ""

    body_indent = len(reference_line) - len(reference_line.lstrip())

    normalized_lines: List[str] = []
    for line in lines:
        if not line.strip():
            normalized_lines.append("")
            continue
        current_indent = len(line) - len(line.lstrip())
        new_indent = max(0, current_indent - body_indent)
        normalized_lines.append(" " * new_indent + line.lstrip())

    code = "\n".join(normalized_lines)

    # Final: indent as function body (4 spaces)
    return textwrap.indent(code, "    ")