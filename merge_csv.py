import os

import argparse
import pandas as pd

from garage.experiment.meta_test_helper import MetaTestHelper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("folder")
    args = parser.parse_args()
    folder = args.folder

    itrs = MetaTestHelper._get_tested_itrs(folder)
    merged_file = os.path.join(folder, "meta-test.csv.2")

    files_to_merge = [os.path.join(folder, "meta-test-itr_{}.csv".format(itr))
                          for itr in itrs]

    merged_csv = pd.concat([pd.read_csv(f) for f in files_to_merge],
                            sort=True)
    merged_csv.sort_values(by=['Iteration'])
    merged_csv.to_csv(merged_file, index=False)