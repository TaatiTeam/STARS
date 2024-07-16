import torch
import torch.nn.functional as F

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def kld(inputs, targets):
        inputs = F.log_softmax(inputs / 0.1, dim=1)
        targets = F.softmax(targets / 0.04, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')