from garage.tf.behavioral_cloning.dataset import Dataset


def test_dataset():
    # Dataset with a single numpy array
    indices = [4, 6, 8]
    length = 20
    dataset = Dataset(
        "garage/tf/behavioral_cloning/point_data_sequential_single_task.npy")
    obs = dataset.fromIndex(*indices).get()
    assert (obs.shape == (
        len(indices),
        50,
    ))
    obs = dataset.fromIndex(*indices).withLength(length, axis=1).get()
    assert (obs.shape == (
        len(indices),
        length,
    ))
    obs = (dataset.fromIndex(*indices).withLength(length, axis=1).fromIndex(
        "observation", "reward").get())
    assert (obs["observation"].shape == (len(indices), length, 2))
    assert (obs["reward"].shape == (len(indices), length))

    # Dataset with dictionary
    length = 10
    dataset = Dataset("garage/tf/behavioral_cloning/point_data_sequential.npy")
    obs = dataset.fromKey("task_1").get()
    assert (obs.shape == (
        13,
        50,
    ))
    obs = dataset.fromKey("task_1").fromIndex(7).get()
    assert (obs.shape == (50, ))
    obs = dataset.fromKey("task_1").fromIndex(7).withLength(length).get()
    assert (obs.shape == (length, ))
    obs = (dataset.fromKey("task_1").fromIndex(7).withLength(length).fromIndex(
        "action", "reward").get())
    assert (obs["action"].shape == (length, 2))
    assert (obs["reward"].shape == (length, ))
