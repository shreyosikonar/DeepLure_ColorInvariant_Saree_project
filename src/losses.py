import torch
import torch.nn.functional as F

def supervised_contrastive_loss(z, labels, temperature=0.07):
    z=F.normalize(z,dim=1)
    sim=torch.matmul(z,z.T)/temperature
    labels=labels.view(-1,1)
    mask=torch.eq(labels,labels.T).float().to(z.device)
    logits_mask=torch.ones_like(mask)-torch.eye(mask.size(0),device=z.device)
    mask=mask*logits_mask
    sim=sim-sim.max(dim=1,keepdim=True).values.detach()
    exp=torch.exp(sim)*logits_mask
    log_prob=sim-torch.log(exp.sum(dim=1,keepdim=True)+1e-12)
    mean_log=(mask*log_prob).sum(1)/(mask.sum(1)+1e-12)
    return -mean_log.mean()
