import subprocess
import sys

def run(script):
    r = subprocess.run(["python", script])
    if r.returncode != 0:
        sys.exit(1)

if __name__ == "__main__":
    run("harris.py")
    run("align.py")
