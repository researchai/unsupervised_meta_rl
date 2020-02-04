"""Functions exposed directly in the garage namespace."""
from collections import defaultdict

from dowel import tabular
import numpy as np

import garage
from garage.misc.tensor_utils import discount_cumsum


def log_multitask_performance(itr, batch, discount, name_map={}):
    traj_by_name = defaultdict(list)
    evaluate_multitask = True
    for trajectory in batch.split():
        try:
            task_name = trajectory.env_infos['task_name'][0]
        except KeyError:
            if name_map != {}:
                task_name = name_map[trajectory.env_infos['task_id'][0]]
            else:
                evaluate_multitask = False
        if evaluate_multitask:
            traj_by_name[task_name].append(trajectory)
    for (task_name, trajectories) in traj_by_name.items():
        log_performance(itr, garage.TrajectoryBatch.concatenate(*trajectories),
                        discount, prefix=task_name)

    return log_performance(itr, batch, discount)

def log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    completion = []
    success = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))
        completion.append(float(trajectory.terminals.any()))
        if 'success' in trajectory.env_infos:
            success.append(trajectory.env_infos['success'].any())

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumTrajs', len(returns))

        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('CompletionRate', np.mean(completion))
        tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns
