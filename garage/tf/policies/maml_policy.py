import tensorflow as tf

# This is an alternative variable store for maml.
MAML_VARIABLE_STORE = set()


class MamlPolicy:

    def __init__(self, wrapped_policy, n_tasks,):

        self._wrapped_policy = wrapped_policy
        self._n_tasks = n_tasks

        self._initialized = False
        self._adapted_param_store = dict()

    def initialize(self, gradient_var, inputs=None):

        print("Creating maml variables now...")

        global MAML_VARIABLE_STORE

        # One step adaptation
        for i in range(self._n_tasks):
            params = self._wrapped_policy.get_params_internal()
            gradient_i = gradient_var[i]

            for p, g in zip(params, gradient_i):
                adapted_param = p - g
                name = "maml/{}/{}".format(i, p.name)
                self._adapted_param_store[name] = adapted_param
                print("Created: {} with:\n\t{}\n\t{}\n".format(name, p.name, g.name))

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

        # build the model with these paramters
        model = self._wrapped_policy.model

        # overload the whole tf.get_variable function
        # this allows us to use an operation as a variable 
        from tensorflow.python.ops import variable_scope
        original_get_variable = variable_scope.get_variable
        variable_scope.get_variable = maml_get_variable

        print("Rebuilding the graphs:")
        print(self._adapted_param_store)
        all_model_infos = list()
        for i in range(self._n_tasks):
            with tf.variable_scope("maml/{}".format(i)):
                model_infos = model.build_model()
                all_model_infos.append(model_infos)

        self._initialized = True

        # Use the original get_variable
        variable_scope.get_variable = original_get_variable

        return all_model_infos


