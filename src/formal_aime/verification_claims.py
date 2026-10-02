"""Distinguish an Action name from a library used by successful Python code."""
import ast


def observed_sympy_calls(turns, failed):
    """Conservatively recognize direct module-level calls after SymPy imports.

    This establishes library use, not mathematical correctness. Comments,
    strings, unevaluated branches/functions, and failed Actions are not proof.
    """
    for turn in turns:
        action = turn.get("action") or {}
        if action.get("name") != "python_exec" or failed(turn):
            continue
        code = (action.get("arguments") or {}).get("code")
        if not isinstance(code, str):
            continue
        try:
            body = ast.parse(code).body
        except (SyntaxError, ValueError):
            continue
        modules, functions = set(), set()
        for statement in body:
            if isinstance(statement, ast.Import):
                modules.update(alias.asname or alias.name for alias in statement.names if alias.name == "sympy")
            elif isinstance(statement, ast.ImportFrom) and statement.module == "sympy":
                functions.update(alias.asname or alias.name for alias in statement.names if alias.name != "*")
            elif isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
                value = statement.value
                if isinstance(value, ast.Call):
                    fn = value.func
                    if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in modules
                            or isinstance(fn, ast.Name) and fn.id in functions):
                        return True
                # A later reassignment must not masquerade as a library call.
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
                for target in targets:
                    for node in ast.walk(target):
                        if isinstance(node, ast.Name):
                            modules.discard(node.id)
                            functions.discard(node.id)
            else:
                # Unknown control flow can redefine bindings. Do not infer use.
                modules.clear()
                functions.clear()
    return False
