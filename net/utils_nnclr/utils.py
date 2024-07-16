import torch.distributed as dist
import torch

# from https://github.com/vturrisi/solo-learn/blob/main/solo/utils/misc.py#L180
# had the problem that backward was not called for some reason
# noinspection PyAbstractClass
class AllGatherGradAutograd(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        # without the tuple call here, the gradient is not propagated for some reason
        # (therefore the backward is then not called)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients, op=dist.ReduceOp.SUM)
        grad_out = all_gradients[dist.get_rank()]
        return grad_out
    

def get_device_and_bfloat16supported():
    # gloo cpu -> okay
    # gloo cuda -> okay (although https://pytorch.org/docs/stable/distributed.html says it isn't supported)
    # nccl cpu -> fail (but gloo anyway recommended for cpu multiprocessing)
    # nccl cuda -> okay
    # bfloat16 cpu -> fail
    if not is_distributed():
        return torch.device("cpu"), True
    if dist.get_backend() == "nccl":
        return torch.device("cuda"), True
    if dist.get_backend() == "gloo":
        return torch.device("cpu"), False
    raise NotImplementedError

def _prepare_tensor(x):
    """
    prepare for distributed communication
    - wrap primitive types into tensors
    - push tensor onto supported device
    """
    device, bfloat16_supported = get_device_and_bfloat16supported()
    # I think this doesn't work in some configuration not sure in which though
    # note in which configuration and convert back to bool after gather
    if isinstance(x, bool):
        # x = torch.tensor(x, dtype=torch.float32, device=device)
        # og_device = torch.device("cpu")
        raise RuntimeError
    if isinstance(x, (float, int, list, tuple)):
        x = torch.tensor(x, device=device)
        og_device = torch.device("cpu")
    else:
        og_device = x.device
    if x.dtype == torch.bfloat16 and not bfloat16_supported:
        x = x.type(torch.float32)
    return x.to(device), og_device


def _all_gather_nondistributed(x, og_device):
    if x.ndim == 0:
        # distributed gather adds a dimension to scalars
        x = x.unsqueeze(0)
    return x.to(og_device)

def _all_gather_grad(x, all_gather_fn):
    x, og_device = _prepare_tensor(x)
    if is_distributed():
        result = all_gather_fn(x)
        if result[0].ndim == 0:
            # scalars can't be concatenated
            result = [r.unsqueeze(0) for r in result]
        return torch.concat(result).to(og_device)
    return _all_gather_nondistributed(x, og_device)


def all_gather_grad(x):
    return _all_gather_grad(x, AllGatherGradAutograd.apply)

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0
