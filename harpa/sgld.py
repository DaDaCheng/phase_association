import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer
import numpy as np

class SGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement pSGLD
    The RMSprop preconditioning code is mostly from pytorch rmsprop implementation.
    """

    def __init__(self, params, lr=1e-3, noise=1e-6, alpha=0.99, eps=1e-8, centered=False, addnoise=True):
        defaults = dict(lr=lr, noise=noise, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(SGLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None, noise=None, add_noise = False):
        """
        Performs a single optimization step.
        """
        loss = None
        
        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            if noise:
                group['noise'] = noise
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data,device=p.device)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data,device=p.device)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                lr_t = group['lr'] * np.power((1 - 1e-5), state['step'] - 1)
                noise_t = group['noise'] * np.power((1 - 5e-5), state['step'] - 1)
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size,device=p.device),
                        torch.ones(size,device=p.device).div_(lr_t).div_(avg).sqrt()
                    )
                    p.data.add_(-lr_t,
                                d_p.div_(avg) + np.sqrt(2) * noise_t * langevin_noise.sample())
                else:
                    #p.data.add_(-group['lr'], d_p.div_(avg))
                    p.data.addcdiv_(-lr_t, d_p, avg)

        return loss
