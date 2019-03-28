"""Maml Policy."""
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.policies.base2 import StochasticPolicy2

# This is an alternative variable store for maml.
MAML_VARIABLE_STORE = set()
MAML_TASK_INDEX = 0


class MamlPolicy(StochasticPolicy2):
    """
    MamlPolicy used by Model-Agnostic Meta-Learning.

    This policy take any garage policies that use models and augment
    them with one step adaptation tensors.

    Args:
        wrapped_policy: A garage policy using model.
        n_tasks: Number of task for meta learning.
        adaptation_step_size: The step size for one step adaptation.
        name: Name of this policy.
    """

    def __init__(self,
                 wrapped_policy,
                 n_tasks,
                 adaptation_step_size=0.5,
                 name="MamlPolicy"):

        self.wrapped_policy = wrapped_policy
        self.n_tasks = n_tasks
        self.name = name
        self._initialized = False
        self._adaptation_step_size = adaptation_step_size
        self._adapted_param_store = dict()
        self._all_model_outputs = list()

        self._create_update_opts()

        super().__init__(wrapped_policy._env_spec)

    def initialize(self, gradient_var, inputs, adaptation_inputs):
        """
        Initialize the MAML Policy.

        This funtion will create all the one step adapted parameters
        and rebuild a model using these adapted parameters for each
        task. A MamlPolicy can only be initialized once.

        Args:
            gradient_var: Gradient variables for one step adaptation.
            inputs: Input tensors for rebuilding the models.
            adaptation_inputs: Input placeholder of adaptation data.
        Returns:
            all_model_outputs: The outputs of rebuilt models.
            update_opts: Adapted parameter tensors.
            update_opts_input: Input for calculating adapted parameters.
        """

        assert not self._initialized, \
            "The MAML policy is initialized and can be initialized once."

        global MAML_VARIABLE_STORE, MAML_TASK_INDEX
        update_opts = []
        with tf.name_scope(self.name):
            # One step adaptation
            with tf.name_scope("Adaptation"):
                for i in range(self.n_tasks):
                    params = self.wrapped_policy.get_params()
                    gradient_i = gradient_var[i]

                    for p, g in zip(params, gradient_i):
                        adapted_param = p - self._adaptation_step_size * g
                        name = "maml_policy/{}/{}".format(i, p.name)
                        self._adapted_param_store[name] = adapted_param
                        if i == 0:
                            update_opts.append(adapted_param)

            def maml_get_variable(name, shape=None, **kwargs):
                scope = tf.get_variable_scope()
                idx = 0
                if MAML_TASK_INDEX >= self.n_tasks:
                    raise ValueError("Invalid TASK_INDEX.")
                fullname = "maml_policy/{}/{}/{}:{}".format(
                    MAML_TASK_INDEX, scope.name, name, idx)
                while fullname in MAML_VARIABLE_STORE:
                    idx += 1
                    fullname = "maml_policy/{}/{}/{}:{}".format(
                        MAML_TASK_INDEX, scope.name, name, idx)
                MAML_VARIABLE_STORE.add(fullname)
                return self._adapted_param_store[fullname]

            # build the model with these parameters
            model = self.wrapped_policy.model

            # overload the whole tf.get_variable function
            # this allows us to use an operation as a variable
            from tensorflow.python.ops import variable_scope
            original_get_variable = variable_scope.get_variable
            variable_scope.get_variable = maml_get_variable

            for i in range(self.n_tasks):
                input_for_adapted = inputs[i]
                model_infos = model.build(
                    input_for_adapted, name="task{}".format(i))
                MAML_TASK_INDEX += 1
                self._all_model_outputs.append(model_infos)

            self._initialized = True

        # Use the original get_variable
        variable_scope.get_variable = original_get_variable
        update_opts_input = inputs[0]

        # Build get_action_with_adaptation_data.
        # Here, just use the last set of params.
        input_ph = tf.placeholder(
            dtype=inputs[0].dtype,
            shape=inputs[0].shape,)
        MAML_TASK_INDEX -= 1
        model_outputs = model.build(
            input_ph, name='get_action_with_adaptation_data')
        MAML_TASK_INDEX += 1

        assert MAML_TASK_INDEX == self.n_tasks

        outputs = [model_outputs[0], model_outputs[1], model_outputs[2]]
        self._f_get_actions_with_adaptation = tensor_utils.compile_function(
            tensor_utils.flatten_inputs([input_ph, adaptation_inputs[-1]]),
            outputs,
            log_name='get_action_with_adaptation_data',)

        return self._all_model_outputs, update_opts, update_opts_input

    @property
    def recurrent(self):
        """Indicate whether the policy is recurrent."""
        return self.wrapped_policy.recurrent

    def get_action(self, observation):
        """Get an action from the policy."""
        return self.wrapped_policy.get_action(observation)

    def get_actions(self, observations):
        """Get actions from the policy."""
        return self.wrapped_policy.get_actions(observations)

    def get_actions_with_adaptation_data(self, observations, adaptation_data):
        observations = self.wrapped_policy.observation_space.flatten_n(observations)
        data = tensor_utils.flatten_inputs([observations, adaptation_data])
        actions, means, std_param = self._f_get_actions_with_adaptation(*data)
        return actions, dict(mean=means, log_std=std_param)

    def _create_update_opts(self):
        """Create assign operations for updating parameters."""
        params = self.get_params()
        with tf.name_scope("{}/UpdateWrappedPolicy".format(self.name)):
            self.update_opts = []
            self.adapated_placeholders = []
            for p in params:
                ph = tf.placeholder(
                    dtype=p.dtype,
                    shape=p.shape,
                )
                self.adapated_placeholders.append(ph)
                self.update_opts.append(tf.assign(p, ph))

    def update_params(self, params):
        """Update parameters for the wrapped_policy"""
        feed_dict = dict(zip(self.adapated_placeholders, params))
        sess = tf.get_default_session()
        sess.run(self.update_opts, feed_dict=feed_dict)

    def get_params(self, trainable=True):
        """Get the trainable variables."""
        return self.wrapped_policy.get_params(trainable)

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = self.__dict__.copy()
        del new_dict['_f_get_actions_with_adaptation']
        del new_dict['_adapted_param_store']
        del new_dict['_all_model_outputs']
        del new_dict['adapated_placeholders']
        del new_dict['update_opts']
        del new_dict['_initialized']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__.update(state)
        self._initialize()
