
import sys
import subprocess


def main():
    # Base pytest command
    args = [
        "pytest",
        "tests",           # run all tests inside tests/
        "--tb=short",      # shorter tracebacks
    ]

    user_args = sys.argv[1:]

    # Verbose mode
    if "-v" in user_args:
        args.append("-v")

    # Fail fast (stop on first failure)
    if "--failfast" in user_args:
        args.append("-x")

    # Coverage (optional)
    if "--coverage" in user_args:
        args.extend([
            "--cov=.",
            "--cov-report=term-missing",
        ])

    # Quiet mode (default if not verbose)
    if "-v" not in user_args:
        args.append("-q")

    try:
        result = subprocess.run(args, check=False)
        sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\nTest run interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
