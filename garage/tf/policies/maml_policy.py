import tensorflow as tf

from garage.core import Serializable
from garage.tf.policies import Policy

# This is an alternative variable store for maml.
MAML_VARIABLE_STORE = set()


class MamlPolicy(Policy, Serializable):

    """
    The MamlPolicy that aguments other policies.

    This policy take arbitary policy and augment them with
    one step adaptation tensors. At this point, the maml
    does not contain its own model (which does not fit the
    garage model design). This should be fixed in future.
    """

    def __init__(self, wrapped_policy, n_tasks, adaptation_step_size=1e-2, name="MamlPolicy"):

        self.wrapped_policy = wrapped_policy
        self.n_tasks = n_tasks
        self.name = name
        self._initialized = False
        self._adaptation_step_size = adaptation_step_size
        self._adapted_param_store = dict()

        self._create_update_opts()

        super().__init__(wrapped_policy._env_spec)
        Serializable.quick_init(self, locals())

    def initialize(self, gradient_var, inputs=None):

        assert not self._initialized, "The MAML policy is initialized and can be initialized once."

        print("Creating maml variables now...")
        global MAML_VARIABLE_STORE

        update_opts = []

        # One step adaptation
        for i in range(self.n_tasks):
            params = self.wrapped_policy.get_params_internal()
            gradient_i = gradient_var[i]

            for p, g in zip(params, gradient_i):
                adapted_param = p - self._adaptation_step_size * g
                name = "maml_policy/{}/{}".format(i, p.name)
                self._adapted_param_store[name] = adapted_param
                print("Created: {} with:\n\t{}\n\t{}\n".format(name, p.name, g.name))

                if i == 0:
                    update_opts.append(adapted_param)

        print("Done with creating variables\n\n\n")

        def maml_get_variable(name, shape=None, **kwargs):
            scope = tf.get_variable_scope()
            idx = 0
            fullname = "{}/{}:{}".format(scope.name, name, idx)
            while fullname in MAML_VARIABLE_STORE:
                idx += 1
                fullname = "{}/{}:{}".format(scope.name, name, idx)
            MAML_VARIABLE_STORE.add(fullname)
            print("Retrieved: {}".format(fullname))
            return self._adapted_param_store[fullname]

        # build the model with these parameters
        model = self.wrapped_policy.model

        # overload the whole tf.get_variable function
        # this allows us to use an operation as a variable 
        from tensorflow.python.ops import variable_scope
        original_get_variable = variable_scope.get_variable
        variable_scope.get_variable = maml_get_variable

        print("Rebuilding the graphs:")
        print(self._adapted_param_store)
        all_model_infos = list()
        for i in range(self.n_tasks):
            input_for_adapted = inputs[i] if inputs else None
            with tf.variable_scope("maml_policy/{}".format(i)):
                model_infos = model.build_model(inputs=input_for_adapted)
                all_model_infos.append(model_infos)

        self._initialized = True

        # Use the original get_variable
        variable_scope.get_variable = original_get_variable

        update_opts_input = inputs[0] if inputs else all_model_infos[0][0]
        return all_model_infos, update_opts, update_opts_input
    
    @property
    def recurrent(self):
        return False

    def get_action(self, observation):
        return self.wrapped_policy.get_action(observation)

    def get_actions(self, observations):
        return self.wrapped_policy.get_actions(observations)

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params_internal(**tags)

    def _create_update_opts(self):
        params = self.get_params_internal()

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
        feed_dict = dict(zip(self.adapated_placeholders, params))
        sess = tf.get_default_session()
        sess.run(self.update_opts, feed_dict=feed_dict)
