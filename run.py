import os
import shutil
import subprocess
import sys

def run_command(command):
    """Executes a shell command safely."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

def generate():
    print("[INFO] Step 1/3: Generating synthetic dataset...")
    run_command(f"{sys.executable} scripts/01_generate_dataset.py")

def benchmark():
    print("[INFO] Step 2/3: Running algorithm benchmark...")
    run_command(f"{sys.executable} scripts/02_run_benchmark.py")

def analyze():
    print("[INFO] Step 3/3: Analyzing results and generating report...")
    run_command(f"{sys.executable} scripts/03_analyze_results.py")

def clean():
    print("[INFO] Cleaning generated files and logs...")
    fits_dir = os.path.join("data", "synthetic_test")
    if os.path.exists(fits_dir):
        for f in os.listdir(fits_dir):
            if f.endswith(".fits"):
                os.remove(os.path.join(fits_dir, f))
                
    for file_to_remove in ["results_benchmark_raw_data.csv", "results_benchmark_rankings.txt"]:
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
    print("[INFO] Cleanup completed successfully.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [generate | benchmark | analyze | all | clean]")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    
    if action == "generate":
        generate()
    elif action == "benchmark":
        benchmark()
    elif action == "analyze":
        analyze()
    elif action == "all":
        generate()
        benchmark()
        analyze()
    elif action == "clean":
        clean()
    else:
        print(f"Unknown command: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()