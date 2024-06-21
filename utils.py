import sys

def print2txt(func, path2file):
  # Save func() output to .txt file
  with open(path2file, 'w') as file:
    # Redirect standard output to the file
    original_stdout = sys.stdout
    sys.stdout = file
    func()
    # Restore standard output
    sys.stdout = original_stdout
    pass
