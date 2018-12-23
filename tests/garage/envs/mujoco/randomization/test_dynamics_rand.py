import io
import unittest

from lxml import etree
import numpy as np

from garage.envs.mujoco.randomization import Distribution
from garage.envs.mujoco.randomization import Method
from garage.envs.mujoco.randomization import randomize
from garage.envs.mujoco.randomization import Variations
from garage.envs.mujoco.swimmer_env import SwimmerEnv


class TestDynamicsRand(unittest.TestCase):
    def setUp(self):
        self._env = SwimmerEnv()
        self._bodyname = "mid"
        self._attribute = "pos"

        # values for UNIFORM Distributions
        self._lower_bound = np.array([0.5, 0.5, 0.5])
        self._upper_bound = np.array([1.5, 1.5, 1.5])
        # values for GAUSSIAN Distributions
        self._mean = self._lower_bound
        self._std_dev = self._lower_bound * 2

        # Properties for mocking model generation
        self._counter = 0
        self._mocks_abs = [[1.0, 1.0, 1.0], [0.9, 0.87, 0.59], [0.5, 0.5, 0.5],
                           [1, 0.5, 1.5], [0.89, 1.25, 1.36]]
        self._mocks_scaled = [[0.25, 0, 0], [0.5, 0, 0], [0.3, 0, 0],
                              [0.7, 0, 0], [0.64, 0, 0]]

        # ensure you have legit bounds for randomization
        assert all(self._lower_bound != 1)
        assert all(self._upper_bound != 1)
        assert all(self._upper_bound > self._lower_bound)

        # Get original value for scaling
        body_id = self._env.sim.model._body_name2id[self._bodyname]
        self._orig_val = np.array(self._env.sim.model.body_pos[body_id])

    def create_randomized_xml(self):
        with io.StringIO() as model_xml:
            self._env.sim.save(model_xml, 'xml')
            model_xml.seek(0)
            root_node = etree.parse(model_xml)

        if self._testMethodName == 'test_absolute_method':
            mock_vals = self._mocks_abs
        elif self._testMethodName == 'test_scaled_method':
            mock_vals = self._mocks_scaled
        else:
            mock_vals = None
            assert ValueError, "Specify mock values for new test method"

        e = root_node.find("//body[@name=\'{0}\']".format(self._bodyname))
        mock_val = str(mock_vals[self._counter]).strip("[]").replace(',', '')
        self._counter += 1
        e.attrib[self._attribute] = mock_val
        return etree.tostring(root_node.getroot(), encoding='unicode')

    def test_absolute_method(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)
        randomized_vals = []

        for i in range(5):
            with unittest.mock.patch(
                    'garage.envs.mujoco.randomization.variation.Variations.'
                    'get_randomized_xml_model', self.create_randomized_xml):
                randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                self._bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)

            # check if within range
            assert all(self._lower_bound <= randomized_val), (
                "Randomised {1} < lower bound {0}".format(
                    self._lower_bound, randomized_val))
            assert all(randomized_val <= self._upper_bound), (
                "Randomised {1} > upper bound {0}".format(
                    self._upper_bound, randomized_val))

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        assert np.std(randomized_vals) > 0, (
            "Std Dev of randomized values "
            "not > 0. Getting the exact "
            "same numbers?\n {0}".format(randomized_vals))

    def test_scaled_method(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.SCALED) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)
        randomized_vals = []

        for i in range(5):
            with unittest.mock.patch(
                    'garage.envs.mujoco.randomization.variation.Variations.'
                    'get_randomized_xml_model', self.create_randomized_xml):
                randomized_env.reset()
            body_id = randomized_env.wrapped_env.sim.model._body_name2id[
                self._bodyname]
            randomized_val = np.array(
                randomized_env.wrapped_env.sim.model.body_pos[body_id])
            randomized_vals.append(randomized_val)

            # check if within range
            assert all(self._orig_val * self._lower_bound <= randomized_val), (
                "Randomised {1} < lower bound {0}".format(
                    self._orig_val * self._lower_bound, randomized_val))
            assert all(randomized_val <= self._orig_val * self._upper_bound), (
                "Randomised {1} > upper bound {0}".format(
                    self._orig_val * self._upper_bound, randomized_val))

        # check that you have actual variation
        randomized_vals = np.array(randomized_vals)
        assert np.std(randomized_vals) > 0, (
            "Std Dev of randomized values "
            "not > 0. Getting the exact "
            "same numbers?\n {0}".format(randomized_vals))

    def test_env_step(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        second_var = "torso"  # Add a second
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(second_var)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        # step() shouldn't error out (so it actually steps the wrapped env)
        for j in range(5):
            randomized_env.step(randomized_env.action_space.sample())

    def test_xml_node_exception(self):
        fake_node = "fake_node"
        variations = Variations()

        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(fake_node)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()
        with self.assertRaises(ValueError) as context:
            randomize(self._env, variations)

        assert fake_node in str(context.exception)

    def test_xml_attrib_exception(self):
        fake_attrib = "fake_attrib"
        variations = Variations()

        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(fake_attrib)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()
        with self.assertRaises(ValueError) as context:
            randomize(self._env, variations)

        assert fake_attrib in str(context.exception)

    def test_exception_gaussian_dist(self):
        variations = Variations()
        with self.assertRaises(ValueError) as context:
            variations.randomize() \
                .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
                .attribute("{0}".format(self._attribute)) \
                .with_method(Method.ABSOLUTE) \
                .sampled_from(Distribution.GAUSSIAN) \
                .with_range(self._lower_bound, self._upper_bound) \
                .add()
        assert "Need to call with_mean_std when sampled from Gaussian" \
               in str(context.exception)

    def test_exception_bad_distribution(self):
        # Use unused distribution Enum value for test
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(3) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        with self.assertRaises(ValueError) as context:
            randomized_env.reset()
        assert "Unknown distribution" in str(context.exception)

    def test_exception_uniform_dist(self):
        variations = Variations()
        with self.assertRaises(ValueError) as context:
            variations.randomize() \
                .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
                .attribute("{0}".format(self._attribute)) \
                .with_method(Method.ABSOLUTE) \
                .sampled_from(Distribution.UNIFORM) \
                .with_mean_std(self._lower_bound, self._upper_bound) \
                .add()
        assert "Need to call with_range when sampled from Uniform" \
               in str(context.exception)

    def test_exception_bad_sampling_method(self):
        # Use unused distribution Enum value for test
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(3) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        randomized_env = randomize(self._env, variations)

        with self.assertRaises(ValueError) as context:
            randomized_env.reset()
        assert "Unknown method" in str(context.exception)

    def test_prop_elem(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        assert variations.get_list()[0].elem is None

    def test_prop_get_list(self):
        variations = Variations()
        assert variations.get_list() == []

    def test_prop_default(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.UNIFORM) \
            .with_range(self._lower_bound, self._upper_bound) \
            .add()

        assert variations.get_list()[0].default is None

    def test_prop_mean_std(self):
        variations = Variations()
        variations.randomize() \
            .at_xpath("//body[@name=\'{0}\']".format(self._bodyname)) \
            .attribute("{0}".format(self._attribute)) \
            .with_method(Method.ABSOLUTE) \
            .sampled_from(Distribution.GAUSSIAN) \
            .with_mean_std(self._mean, self._std_dev) \
            .add()

        assert variations.get_list()[0].mean_std == (self._mean, self._std_dev)

    def test_swimmer(self):
        import time

        env = SwimmerEnv()
        filename = "/tmp/myfile.xml"
        obj_type = 'geom'
        obj_name = 'torso'
        attrib = 'size'
        f = open(filename, "w", encoding="utf-8")
        m = env.sim.model
        body_funcs = {'mass': m.body_mass, 'pos': m.body_pos}
        geom_funcs = {'size': m.geom_size}
        if obj_type == 'body':
            obj_id = env.sim.model._body_name2id[obj_name]  # get ID
            val_to_access = body_funcs[attrib][obj_id]
        elif obj_type == 'geom':
            obj_id = env.sim.model._geom_name2id[obj_name]  # get ID
            val_to_access = geom_funcs[attrib][obj_id]
        else:
            raise NotImplementedError("Choose body or geom for obj_type")

        val = np.array(val_to_access)
        print(f'Original {attrib}', val)  # get original attribute value

        # Update size and store to new model
        geom_funcs[attrib][obj_id] = 1 * val
        env.sim.save(f, 'xml')
        f.close()

        env2 = SwimmerEnv(file_path=filename)
        env2.reset()
        action = np.ones(env2.action_space.shape) * 4000
        if obj_type == 'body':
            obj_id = env.sim.model._body_name2id[obj_name]  # get ID
            val_to_access = body_funcs[attrib][obj_id]
        elif obj_type == 'geom':
            obj_id = env.sim.model._geom_name2id[obj_name]  # get ID
            val_to_access = geom_funcs[attrib][obj_id]
        val = np.array(val_to_access)
        print(f'New {attrib}', val)

        for i in range(500):
            time.sleep(0.01)
            env2.step(action=action)
            env2.render()

    def test_reacher(self):
        import time
        from sawyer.mujoco.reacher_env import ReacherEnv
        from mujoco_py import MjSim, load_model_from_xml
        import lxml.etree as et
        import io
        from sawyer.garage.envs.mujoco.mujoco_env import MODEL_DIR

        env = ReacherEnv(goal_position=[0, 0, 0], control_method='position_control')
        obj_type = 'body'
        obj_name = 'right_l6'
        attrib = 'pos'
        m = env.sim.model
        body_funcs = {'mass': m.body_mass, 'pos': m.body_pos}
        geom_funcs = {'size': m.geom_size}
        if obj_type == 'body':
            obj_id = env.sim.model._body_name2id[obj_name]  # get ID
            val_to_access = body_funcs[attrib][obj_id]
        elif obj_type == 'geom':
            obj_id = env.sim.model._geom_name2id[obj_name]  # get ID
            val_to_access = geom_funcs[attrib][obj_id]
        else:
            raise NotImplementedError(f"obj_type {obj_type} not recognized")

        val = np.array(val_to_access)
        print(f'Original {attrib}', val)  # get original attribute value

        # Update size and store to new model
        body_funcs[attrib][obj_id] = 3 * val

        mem_io = io.StringIO()
        env.sim.save(mem_io, 'xml')
        env.close()

        # Change the meshdir attribute to point to the write directory for STLs
        mem_io.seek(0)
        tree = et.parse(mem_io)
        compiler = tree.find('compiler')
        compiler.attrib['meshdir'] = MODEL_DIR+'/meshes/'

        env2 = ReacherEnv(goal_position=[0, 0, 0], control_method='position_control')
        env2.model = load_model_from_xml(et.tounicode(tree))
        env2.sim = MjSim(env2.model)
        env2.data = env2.sim.data
        env2.init_qpos = env2.sim.data.qpos
        env2.init_qvel = env2.sim.data.qvel
        env2.init_qacc = env2.sim.data.qacc
        env2.init_ctrl = env2.sim.data.ctrl
        env2.reset()

        for i in range(500):
            time.sleep(0.01)
            if i % 20 == 0:
                print('env2', i)
            env2.render()

        m = env2.sim.model
        body_funcs = {'mass': m.body_mass, 'pos': m.body_pos}
        geom_funcs = {'size': m.geom_size}
        if obj_type == 'body':
            obj_id = env2.sim.model._body_name2id[obj_name]  # get ID
            val_to_access = body_funcs[attrib][obj_id]
        elif obj_type == 'geom':
            obj_id = env.sim.model._geom_name2id[obj_name]  # get ID
            val_to_access = geom_funcs[attrib][obj_id]
        else:
            raise NotImplementedError(f"obj_type {obj_type} not recognized")

        val = np.array(val_to_access)
        print(f'New {attrib}', val)

