import code

from torch.autograd import Function


def snap_to_initial_tokenizer(is_tokenizer_parameter, initial_params, current_params):
    return [init_params if is_tokenizer else curr_params for is_tokenizer, init_params, curr_params in
     zip(is_tokenizer_parameter, initial_params, current_params)]

def snap_to_initial_model(is_tokenizer_parameter, initial_params, current_params):
    return [init_params if not is_tokenizer else curr_params for is_tokenizer, init_params, curr_params in
     zip(is_tokenizer_parameter, initial_params, current_params)]

def stop_tokenizer_grad(is_tokenizer_parameter, params):
    return [StopGrad.apply(param)[0] if is_tokenizer else param for is_tokenizer, param in zip(is_tokenizer_parameter, params)]

def extract_tokenizer(is_tokenizer_parameter, params):
    return [param for is_tokenizer, param in zip(is_tokenizer_parameter, params) if is_tokenizer]

def extract_model(is_tokenizer_parameter, params):
    return [param for is_tokenizer, param in zip(is_tokenizer_parameter, params) if not is_tokenizer]

class StopGrad(Function):

    @staticmethod
    def forward(ctx, *args):
        return tuple(arg.detach() for arg in args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return tuple([None]) * len(grad_outputs)