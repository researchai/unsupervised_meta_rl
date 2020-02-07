import dowel
from dowel import logger as dowel_logger
from dowel import tabular
import os
import os.path as osp

from garage.experiment import Snapshotter
from garage.experiment import LocalRunner, SnapshotConfig
import garage.torch.utils as tu


log_dir = '/home/lindawong/Downloads'
temp_dir = '/home/lindawong/Desktop/temp'
snapshot_config = SnapshotConfig(snapshot_dir=temp_dir,
                                 snapshot_mode='all',
                                 snapshot_gap=1)

tabular_log_file = osp.join(temp_dir, '3progress.csv')
dowel_logger.add_output(dowel.StdOutput())
dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
runner = LocalRunner(snapshot_config=snapshot_config)
runner.restore(log_dir, from_epoch=330)

tu.set_gpu_mode(True, gpu_id=0)
runner._algo.to()

num_adapt = [1, 2, 4, 8, 16, 32, 64]
num_steps = [3150, 3300, 3600, 4200, 5400, 7800, 12600]

for a, s in zip(num_adapt, num_steps):
    runner._algo.evaluate(330, a, s)

    dowel_logger.log(tabular)
    dowel_logger.dump_output_type(dowel.CsvOutput)

dowel_logger.remove_output_type(dowel.CsvOutput)
dowel_logger.remove_all()
