"""Executed by the repository's Python, not the training interpreter (Python >=3.9)."""
import json
import os
import sys
from pathlib import Path


class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.started = 0
        self.collected = 0
        self.ran = 0
        self.setup_errors = 0

    def write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"started": self.started, "collected": self.collected,
                                         "ran": self.ran, "setup_errors": self.setup_errors}))
        os.replace(str(temporary), str(self.path))

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
        self.write()

    def pytest_runtest_logstart(self, nodeid, location):
        self.started += 1
        self.write()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and not report.skipped:
            self.ran += 1
        if report.when == "setup" and report.failed:
            self.setup_errors += 1
        self.write()


if __name__ == "__main__":
    import pytest

    evidence = Evidence(sys.argv[1])
    evidence.write()
    sys.exit(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider",
                         "--basetemp=" + str(Path(os.environ["TMPDIR"]) / "pytest"), *sys.argv[2:]],
                         plugins=[evidence]))
