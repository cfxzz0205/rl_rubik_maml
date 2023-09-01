import numpy as np
import learn2learn as l2l
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F
def smooth(x,y):
    x = np.array(x)
    y = np.array(y)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth ,y_smooth

def lossfn(policy_out_t, value_out_t, weights_t, y_policy_t, y_value_t,config):
    value_out_t = value_out_t.squeeze(-1)
    value_loss_t = (value_out_t - y_value_t)**2
    value_loss_raw_t = value_loss_t.mean()
    if config.weight_samples:
        value_loss_t *= weights_t
    value_loss_t = value_loss_t.mean()
    policy_loss_t = F.cross_entropy(policy_out_t, y_policy_t, reduction='none')
    policy_loss_raw_t = policy_loss_t.mean()
    if config.weight_samples:
        policy_loss_t *= weights_t
    policy_loss_t = policy_loss_t.mean()
    loss_raw_t = policy_loss_raw_t + value_loss_raw_t
    loss_t = value_loss_t + policy_loss_t
    return loss_t