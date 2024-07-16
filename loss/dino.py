import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 



class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_locals, n_globals, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, centering_type='centering',
                 batch_size=16, center_momentum=0.9, loss_type="cross-entropy"):
        super().__init__()
        assert loss_type in ['cross-entropy', 'cosine-similarity']
        assert centering_type in ['centering', 'sinkhorn_knopp', 'none']
        self.loss_type = loss_type
        if loss_type == 'cosine-similarity':
            self.cosine_loss = nn.CosineEmbeddingLoss()
            self.target = torch.ones(batch_size).cuda() # We want them to be similar

        self.student_temp = student_temp
        self.centering_type = centering_type
        self.center_momentum = center_momentum
        self.n_locals = n_locals
        self.n_globals = n_globals
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_locals + self.n_globals)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        if self.centering_type == "centering":
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        elif self.centering_type == "sinkhorn_knopp":
            Q = torch.exp(teacher_output / temp).t()
            B = Q.shape[1]  # sample numbers in our batch
            K = Q.shape[0]  # prototype numbers
            Q /= torch.sum(Q)

            for sinkhorn_iteration in range(3):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B
            Q *= B  # the columns must sum to 1 so that Q is an assignment
            teacher_out = Q.t()
        else:
            teacher_out = F.softmax((teacher_output) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_globals)

        total_loss, n_loss_terms = 0, 0
        total_kl, total_entropy = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                else:
                    if self.loss_type == "cross-entropy":
                        loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) 
                        entropy = torch.sum(-q * torch.log(q + 1e-10), dim=-1)
                        kl = loss - entropy
                    else:
                        loss = self.cosine_loss(q, student_out[v], self.target)
                        entropy = torch.sum(-q * torch.log(q + 1e-10), dim=-1)
                        kl = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) - entropy
                    total_entropy += entropy.mean()
                    total_kl += kl.mean()
                    total_loss += loss.mean()
                    n_loss_terms += 1

        total_loss = total_loss / n_loss_terms
        total_kl /= n_loss_terms
        total_entropy /= n_loss_terms
        if self.centering_type == 'centering':
            self.update_center(teacher_output)
        return total_loss, total_kl, total_entropy

    @torch.no_grad() 
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)