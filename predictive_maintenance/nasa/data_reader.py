import pandas as pd
import os

class BearingData:
    test_number= 1
    sample_number=1
    bearing_number = 1
    data_frame =None
    is_failed = False

channel_map = {
                1: (0, 1),
                2: (2, 3),
                3: (4, 5),
                4: (6, 7)
            }
data_path = r'/Users/pankajti/dev/data/kaggle/nasa/archive (1)/1st_test/1st_test'


def load_first_test_data (bearing_number):
    files_to_be_loaded = -1
    files = os.listdir(data_path)
    signals_from_files = []
    for file in files[files_to_be_loaded:] :
        file_path = os.path.join(data_path,file)
        if os.path.exists(file_path):
            data = pd.read_table(file_path,header=None)
            col_set = channel_map[bearing_number]
            ch1, ch2 = col_set
            # Extract both channels and flatten them
            signal1 = data.iloc[:, ch1]
            #signal2 = data.iloc[:, ch2].values
            print(f"reading for bearing number{bearing_number} file {file}")
            signals_from_files.append(signal1)
        else:
            print(f"file {file_path} is not present")


    all_signals  = pd.concat(signals_from_files)
    return all_signals

import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_signals = load_first_test_data(bearing_number=3)
    all_signals.plot()
    plt.show()
    print(all_signals)

