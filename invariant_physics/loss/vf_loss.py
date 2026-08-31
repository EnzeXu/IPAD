from math import sqrt, pi
from typing import Callable

import numpy as np
from numpy import ndarray, newaxis

from scipy.integrate import trapz, simps


class VF_Loss:
    '''
        Create a `VF_loss` object with specific settings. Now
        only `fourier` is supported.
        
        Args:
        - `func_num`: the number of sine functions you would like to use. 
            40~50 is recommend.
        - `func_type`: Now only `fourier` is supported.
        - `integ_method`: numerical integration method. It should be chosen from `simps` 
            and `trapz`.
            Recommendation: `simps` > `trapz`
        - `reduction`: sum or mean.
        '''
    def __init__(
        self, 
        func_num: int,  
        func_type: str = 'fourier',
        integ_method: str = 'simps',
        reduction: str = 'mean', 
    ) -> None:
        super().__init__()
        self.func_num, self.func_type = func_num, func_type
        self.integ_method = integ_method
        self.reduction = reduction
        
    def fourier_basis(self, tspan: ndarray):
        '''
        Generate sine functions.
        
        Mathematical Expressions: 
            ``g_s(t) = sqrt(2/T) sin (s*pi*t / T)``
            
        Attention: `tspan` must be `0, dt, 2dt, ..., T`.
        
        Return shape: `(func_num, len(tspan))`
        '''
        T = tspan[-1]
        s = np.arange(1, self.func_num + 1, 1.0).reshape((-1, 1))
        funcs_input = pi * s @ tspan.reshape((1, -1)) / T
        gs = sqrt(2 / T) * np.sin(funcs_input)
        dgs = pi * s / T * sqrt(2 / T) * np.cos(funcs_input)
        return gs, dgs # (func_num, len(tspan))
        
    def __call__(self, f: ndarray, X: ndarray, tspan: ndarray) -> ndarray:
        '''
        Shape of inputs:
        - `X.shape`: `(len(tspan), traj_num)`
        - `f.shape`: `(len(tspan), traj_num)`
        - `tspan`: `0, dt, 2dt, ..., T`
        '''
        return self.integ_regular(f, X, tspan)
        # tspan.shape: (len(tspan)[, traj_num])
        # if tspan.axis() == 1: return self.integ_regular(ode_func, X, tspan)
        # elif tspan.axis() == 2: return self.integ_irregular(ode_func, X, tspan)
        
    def integ_regular(self, f: ndarray, X: ndarray, tspan: ndarray) -> ndarray:
        '''
        Shape of inputs:
        - `X.shape`: `(len(tspan), traj_num)`
        - `f.shape`: `(len(tspan), traj_num)`
        - `tspan`: `0, dt, 2dt, ..., T`
        '''
        if self.func_type == 'fourier': g, dg = self.fourier_basis(tspan)
        else: raise NotImplementedError 
        # (func_num, len(tspan), traj_num)
        integrand = f * g[..., newaxis] + X * dg[..., newaxis]
        
        if self.integ_method == 'simps':
            loss_raw: ndarray = (simps(integrand, tspan, axis = 1)**2).sum(axis = 0)
        elif self.integ_method == 'trapz':
            loss_raw: ndarray = (trapz(integrand, tspan, axis = 1)**2).sum(axis = 0)
        else: raise NotImplementedError 
            
        if self.reduction == 'sum': return loss_raw.sum()
        elif self.reduction == 'mean': return loss_raw.mean()
        else: raise ValueError('Reduction mode is not set correctly!')
        
        
        
        
        
        
        
        
        
    # def integ_irregular(self, ode_func: Callable, X: ndarray, tspan: ndarray) -> ndarray:
    #     if self.func_type == 'fourier': g, dg = self.batch_fourier_basis(tspan)
    #     else: raise NotImplementedError # TODO: Other funcs
    #     f = np.stack([ode_func(tspan[i], X[i]) for i in range(len(tspan))])
    #     integrand1 = f * g.unsqueeze(-1)
    #     integrand2 = X * dg.unsqueeze(-1)
    #     integrand = integrand1 + integrand2
    #     if self.integ_method == 'trapezoid':
    #         loss_raw = (np.trapezoid(integrand, tspan.unsqueeze(0).unsqueeze(-1), axis = 1)**2).sum(axis = 1)
    #     elif self.integ_method == 'simpson':
    #         loss_raw = (irregular_simpson(integrand, tspan, axis = 1)**2).sum(axis = 0)
    #     if self.reduction == 'sum': return loss_raw.sum()
    #     elif self.reduction == 'mean': return loss_raw.mean()
    #     else: raise ValueError('Reduction mode is not set correctly!')
    
    # Attention: If you would like use this, please contact with me
    # def batch_fourier_basis(self, sampled_tspan: ndarray):
    #     T = sampled_tspan[-1]
    #     s = np.arange(1, self.func_num + 1, 1.0).reshape((-1, 1, 1))
    #     func_input = np.pi * s * sampled_tspan / T
    #     coeff = np.sqrt(2 / T)
    #     gs = coeff * np.sin(func_input)
    #     dgs = np.pi * s / T * coeff * np.cos(func_input)
    #     return gs, dgs