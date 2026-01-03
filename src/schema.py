# src/schema.py

import re
import ast
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Regex definitions (single source of truth)
# ---------------------------------------------------------------------------

RE_START = re.compile(r"<START_WORKING_OUT>", re.IGNORECASE)
RE_END = re.compile(r"</END_WORKING_OUT>", re.IGNORECASE)
RE_SOL = re.compile(r"<SOLUTION>", re.IGNORECASE)
RE_SOL_END = re.compile(r"</SOLUTION>", re.IGNORECASE)

RE_SOLUTION = re.compile(
    r"<SOLUTION>\s*(.*?)\s*</SOLUTION>",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(text: str) -> Tuple[bool, str]:
    """
    Checks whether the model output follows the exact required XML-like schema.

    Returns:
        (is_valid, reason)
    """
    if not RE_START.search(text):
        return False, "Missing <START_WORKING_OUT>"
    if not RE_END.search(text):
        return False, "Missing </END_WORKING_OUT>"
    if not RE_SOL.search(text):
        return False, "Missing <SOLUTION>"
    if not RE_SOL_END.search(text):
        return False, "Missing </SOLUTION>"

    # Enforce ordering: reasoning must come before solution
    start_idx = RE_START.search(text).start()
    sol_idx = RE_SOL.search(text).start()

    if sol_idx < start_idx:
        return False, "Tag order incorrect (<SOLUTION> before reasoning block)"

    return True, "Schema valid"

# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------

def extract_solution(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts Python code inside <SOLUTION> ... </SOLUTION>.

    Returns:
        (code, status)

        code   -> extracted Python string, or None
        status -> textual reason (for debugging / logging)
    """
    match = RE_SOLUTION.search(text)
    if not match:
        return None, "No <SOLUTION> block found"

    code = match.group(1).strip()
    if not code:
        return None, "Empty <SOLUTION> block"

    # Syntax validation via AST
    try:
        ast.parse(code)
    except SyntaxError as e:
        return None, f"Syntax error in extracted code: {e}"

    return code, "Valid Python code extracted"