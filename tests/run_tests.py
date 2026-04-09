"""
Run all NeuralOps tests.
Usage:
    python tests/run_tests.py
    python tests/run_tests.py -v
"""
import subprocess
import sys


def main():
    args = ["pytest", "tests/test_all.py", "--tb=short"] + sys.argv[1:]
    result = subprocess.run(args, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
