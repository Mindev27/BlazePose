import os
import sys
import subprocess


def main():
    print("BlazePose Application Launcher")
    print("==============================")

    # List available Python files to run
    python_files = [f for f in os.listdir(".") if f.endswith(".py") and f != "run.py"]

    print("\nAvailable applications:")
    for i, file in enumerate(python_files, 1):
        print(f"{i}. {file}")

    try:
        choice = int(input("\nSelect an application to run (number): "))
        if 1 <= choice <= len(python_files):
            selected_file = python_files[choice - 1]
            print(f"\nRunning {selected_file}...")
            subprocess.run([sys.executable, selected_file])
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
