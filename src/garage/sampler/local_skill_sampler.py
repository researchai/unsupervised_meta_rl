from garage import SkillTrajectoryBatch
from garage.sampler import SkillWorker
from garage.sampler.local_sampler import LocalSampler


class LocalSkillSampler(LocalSampler):
    def __init__(self, worker_factory, agents, envs):
        super().__init__(worker_factory, agents, envs)

    # @classmethod
    # def from_worker_factory(cls, worker_factory, agents, envs):

    # def _update_workers(self, agent_update, env_update):

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        self._update_workers(agent_update, env_update)
        batches = []
        completed_samples = 0
        while True:
            for worker in self._workers:
                if not isinstance(worker, SkillWorker):
                    raise ValueError('Worker used by Local Skill Sampler class'
                                     ' must be a Skill Worker object, but got '
                                     '{}'.format(worker.dtype))
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                batches.append(batch)
                if completed_samples >= num_samples:
                    return SkillTrajectoryBatch.concatenate(*batches)

    def obtain_exact_trajectories(self,
                                  n_traj_per_worker,
                                  agent_update,
                                  env_update=None):
        self._update_workers(agent_update, env_update)
        batches = []
        for worker in self._workers:
            for _ in range(n_traj_per_worker):
                batch = worker.rollout()
                batches.append(batch)
        return SkillTrajectoryBatch.concatenate(*batches)

    # def shutdown_worker(self)

    # def __getstate__(self)

    # def __setstate__(self, state)



