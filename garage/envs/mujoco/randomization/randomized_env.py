import io
from lxml import etree

from cached_property import cached_property
import gym
from mujoco_py import load_model_from_xml
from mujoco_py import MjSim

from garage.core import Serializable
from garage.envs.mujoco.mujoco_env import MODEL_DIR
from garage.misc.overrides import overrides


class RandomizedEnv(gym.Env, Serializable):
    """
    Wrapper class for the MujocoEnv to perform training using Dynamics Randomization.
    """

    def __init__(self, mujoco_env, variations, *args, **kwargs):
        """
        Set variations with the node in the XML file at file_path.
        """
        Serializable.quick_init(self, locals())
        self._wrapped_env = mujoco_env
        self._variations = variations
        self._wrapped_env_kwargs = kwargs
        temp_buffer = io.StringIO()
        self.wrapped_env.sim.save(temp_buffer, 'xml')

        # Change the meshdir attribute to point to the write directory for STLs
        temp_buffer.seek(0)
        tree = etree.parse(temp_buffer)
        compiler = tree.find('compiler')
        compiler.attrib['meshdir'] = MODEL_DIR+'/meshes/'
        self._xml_buffer = io.StringIO(etree.tounicode(tree))

        self._variations.initialize_variations(self._xml_buffer)

    def reset(self):
        """
        The new model with randomized parameters is requested and the
        corresponding parameters in the MuJoCo environment class are
        set.
        """
        model = load_model_from_xml(
            self._variations.get_randomized_xml_model())
        env_class = self._wrapped_env.__class__
        self._wrapped_env = env_class(**self._wrapped_env_kwargs)
        if self._wrapped_env_kwargs is None:
            print("No keyword args given for wrapped env "
                  "{0}. Using default constructor".format(self._wrapped_env.__class__))
        self._wrapped_env.model = model
        if 'action_space' in self._wrapped_env.__dict__:
            del self._wrapped_env.__dict__['action_space']
        self._wrapped_env.sim = MjSim(self._wrapped_env.model)
        self._wrapped_env.data = self._wrapped_env.sim.data
        self._wrapped_env.init_qpos = self._wrapped_env.sim.data.qpos
        self._wrapped_env.init_qvel = self._wrapped_env.sim.data.qvel
        self._wrapped_env.init_qacc = self._wrapped_env.sim.data.qacc
        self._wrapped_env.init_ctrl = self._wrapped_env.sim.data.ctrl
        return self._wrapped_env.reset()

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def close(self):
        self._wrapped_env.close()

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    @cached_property
    @overrides
    def max_episode_steps(self):
        return self._wrapped_env.spec.max_episode_steps


randomize = RandomizedEnv
