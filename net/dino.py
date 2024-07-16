import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class   
from net.utils_dino.head import DINOHead 
    
class SkeletonDINO(nn.Module):
    """ Referring to the code of DINO, https://arxiv.org/abs/2104.14294 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, dino_head=True, 
                 num_class=60, n_globals=2, **kwargs):
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_student = base_encoder(num_class=num_class, **kwargs)
        else:
            self.n_globals = n_globals
            self.encoder_student = base_encoder(num_class=feature_dim, **kwargs)
            self.encoder_teacher = base_encoder(num_class=feature_dim, **kwargs)

            if dino_head:  # hack: brute-force replacement
                assert kwargs['cls_token'] is False
                self.encoder_student.head = DINOHead(kwargs['dim_feat'], feature_dim)
                self.encoder_teacher.head = DINOHead(kwargs['dim_feat'], feature_dim)

            for param_student, param_teacher in zip(self.encoder_student.parameters(),
                                                    self.encoder_teacher.parameters()):
                param_teacher.data.copy_(param_student.data)    # initialize
                param_teacher.requires_grad = False       # not update by gradient


    @torch.no_grad()
    def _momentum_update_teacher_encoder(self, m):
        """
        Momentum update of the teacher encoder
        """
        for param_student, param_teacher in zip(self.encoder_student.parameters(),
                                                self.encoder_teacher.parameters()):
            param_teacher.data = param_teacher.data * m + param_student.detach().data * (1. - m)

    @staticmethod
    def forward_encoder(x, encoder):
        """
        Forwards multi-trimmed input x to the given encoder.
        Args:
            x (list): List of torch tensors each with shape (B, C, T, V, M)
            encoder (torch.nn.Module): Either student or teacher
        """
        temporal_dimension = 2  # (B, C, T, V, M)
        idx_trims = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[temporal_dimension] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, out_encoder = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_trims:
            _out = encoder(torch.cat(x[start_idx: end_idx]))
            out_encoder = torch.cat((out_encoder, _out))
            start_idx = end_idx

        out_encoder = F.normalize(out_encoder, dim=1)  # (B, NUM_CLASS)
        return out_encoder

    def forward(self, x, m, view='joint'):
        """
        Input:
            x: List of tensors with different temporal sizes (Global and Local)
        """
        if not self.pretrain:
            assert not isinstance(x, list), "Input should be list only in pretraining stage"
            return self.encoder_student(x)
        
        if not isinstance(x, list):
            x = [x]
        out_student = self.forward_encoder(x, self.encoder_student)

        with torch.no_grad():
            self._momentum_update_teacher_encoder(m)
            out_teacher = self.forward_encoder(x[:self.n_globals], self.encoder_teacher)  # Only first two views are global

        return out_student, out_teacher
        