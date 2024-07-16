import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
try:
    from net.generic import concat_all_gather
    from net.utils_nnclr.utils import get_rank, all_gather_grad
except ModuleNotFoundError:
    from generic import concat_all_gather
    from utils_nnclr.utils import get_rank, all_gather_grad


class SkeletonNNCLR(nn.Module):
    """ Referring to the NNCLR, https://arxiv.org/abs/2104.14548 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, num_class=60, queue_size=32768,
                 momentum=0.999, Temperature=0.07, topk=0, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_student = base_encoder(num_class=num_class, clr_pretrain=False, **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.topk = topk

            self.encoder_student = base_encoder(num_class=feature_dim, clr_pretrain=True,
                                                 **kwargs)

            # create the queues
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_id", -torch.ones(queue_size, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, queue, queue_ptr, queue_id, keys, ids):
        keys = concat_all_gather(keys)
        ids = concat_all_gather(ids)
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        remaining_size = ptr + batch_size - self.K
        if remaining_size <= 0:
            queue[:, ptr:ptr + batch_size] = keys.T
            queue_id[ptr:ptr + batch_size] = ids
            queue_ptr[0] = (ptr + batch_size) % self.K
        else:
            queue[:, ptr:self.K] = keys.T[:, 0:self.K - ptr]
            queue[:, 0:remaining_size] = keys.T[:, self.K - ptr:]
            queue_id[ptr:self.K] = ids[0:self.K - ptr]
            queue_id[0:remaining_size] = ids[self.k - ptr:]
            queue_ptr[0] = remaining_size

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward_encoder(self, sequence):
        predict, project = self.encoder_student(sequence)
        predict = F.normalize(predict, dim=1)
        project = F.normalize(project, dim=1)
        return predict, project

    @torch.no_grad()
    def get_queue_similarity_matrix(self, projected_feature, queue, queue_id, ids):
        similarity_matrix = projected_feature @ queue
        # check if queue contains embeddings of the same sample of the previous epoch
        is_own_id = queue_id[None, :] == ids[:, None]
        # set similarity to self to -1
        similarity_matrix[is_own_id] = -1.
        return similarity_matrix 
    
    @torch.no_grad()
    def find_nn(self, projected_feature, queue, queue_id, ids, topk=0):
        similarity_matrix = self.get_queue_similarity_matrix(projected_feature, queue, queue_id, ids)
        if topk == 0:
            idx = similarity_matrix.max(dim=1)[1]
        else:
            n = similarity_matrix.shape[0]
            candidate_idx = similarity_matrix.topk(topk, dim=1)[1]
            dice = torch.randint(size=(n,), high=topk)
            idx = candidate_idx[torch.arange(n), dice]
        nearest_neighbor = queue[:, idx]
        return nearest_neighbor
    
    @staticmethod
    def nnclr_loss_fn(predicted, nn, temperature):
        rank = get_rank()
        predicted = all_gather_grad(predicted)
        logits = nn.T @ predicted.T / temperature
        n = nn.size(1)

        labels = torch.arange(n * rank, n * (rank + 1), device=predicted.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, seq_q, seq_k=None, ids=None):
        """
        Input:
            seq_q: a batch of query sequences
            seq_k: a batch of key sequences
        """
        if not self.pretrain:
            return self.encoder_student(seq_q)

        predict0, project0 = self.forward_encoder(seq_q)
        seq_k, idx_k_unshuffle = self._batch_shuffle_ddp(seq_k)
        predict1, project1 = self.forward_encoder(seq_k)
        predict1 = self._batch_unshuffle_ddp(predict1, idx_k_unshuffle)
        project1 = self._batch_unshuffle_ddp(project1, idx_k_unshuffle)

        nn0 = self.find_nn(project0, self.queue, self.queue_id, ids, self.topk)
        nn1 = self.find_nn(project1, self.queue, self.queue_id, ids, self.topk)

        loss0 = self.nnclr_loss_fn(predict0, nn1, self.T)
        loss1 = self.nnclr_loss_fn(predict1, nn0, self.T)

        self._dequeue_and_enqueue(self.queue, self.queue_ptr, self.queue_id, project0, ids)

        loss = (loss0 + loss1) / 2
        return loss
        