import pandas as pd
import os


def main():
    DATA_PATH =  r'/Users/pankajti/dev/data/kaggle/nasa/CMaps'
    files = os.listdir(DATA_PATH)
    for f in files:
        file_path = os.path.join(DATA_PATH,f)
        if os.path.exists(file_path):
            print(f"file {file_path} exists "  )

if __name__ == '__main__':
    main()