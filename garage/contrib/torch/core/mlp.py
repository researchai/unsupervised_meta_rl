import torch
import torch.nn as nn

def mlp(input_var,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity=torch.tanh,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_normal_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False):
    """
    MLP model.

    Args:
        input_var: Input torch.Tensor to the MLP.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        name: variable scope of the mlp.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        layer_normalization: Bool for using layer normalization or not.

    Return:
        The output torch.Tensor of the MLP
    """
    x = input_var
    last_size = x.shape[1]
    for size in hidden_sizes:
        layer = nn.Linear(last_size, size)
        hidden_w_init(layer.weight)
        hidden_b_init(layer.bias)
        x = hidden_nonlinearity(layer(x))
        if layer_normalization:
            x = nn.LayerNorm(size)(x)
        last_size = size

    layer = nn.Linear(last_size, output_dim)
    output_w_init(layer.weight)
    output_b_init(layer.bias)
    x = layer(x)
    if output_nonlinearity:
        x = output_nonlinearity(x)

    return x
