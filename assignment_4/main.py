import subprocess
import sys


def run_script(script_name):

    print(f"\n{'=' * 60}")
    print(f"Running {script_name}...")
    print('=' * 60)

    try:
        result = subprocess.run([sys.executable, script_name],
                                capture_output=True,
                                text=True,
                                check=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

    return True


def main():

    print("=" * 60)
    print("Assignment 4 - Running All Tasks")
    print("=" * 60)

    scripts = [
        'harris.py',
        'align.py',
        'matches.py'
    ]

    for script in scripts:
        if not run_script(script):
            print(f"\nFailed to run {script}. Stopping.")
            return

    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. harris.png  - Harris corner detection (Page 1 of PDF)")
    print("  2. aligned.png - Aligned image (Page 2 of PDF)")
    print("  3. matches.png - Feature matches (Page 3 of PDF)")
    print("\nNext step: Combine these images into a PDF for submission.")


if __name__ == "__main__":
    main()