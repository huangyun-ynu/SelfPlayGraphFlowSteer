"""Compile source without executing it or writing bytecode; supports Python 3.6+."""

import json
import sys


def main():
    targets = sys.argv[1:]
    checked, failures = [], []
    for target in targets:
        try:
            with open(target, "rb") as handle:
                source = handle.read()
        except OSError as exc:
            failures.append({"path": target, "phase": "read", "message": str(exc)[:2000]})
            continue
        checked.append(target)
        try:
            compile(source, target, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OverflowError) as exc:
            failures.append({"path": target, "phase": "compile", "message": str(exc)[:2000]})
    complete = bool(targets) and checked == targets
    print(json.dumps({"execution_evidence": "syntax_probe_v1", "files_checked": checked,
                      "complete": complete, "failures": failures}))
    return 0 if complete and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
