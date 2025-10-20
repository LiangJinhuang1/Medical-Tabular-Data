"""TabM : orginal from https://github.com/yandex-research/tabm/blob/main/tabm.py"""

import chunk
import dis
from msvcrt import kbhit
import torch 
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

import typing
from typing import List,Any, Literal, Optional, Union
import warnings

#==========================================================
#Initializations
#==========================================================

def _init_rsqrt_uniform_(tensor:Tensor, d:int)->Tensor:
    assert d > 0
    d_rsqrt = d ** -0.5
    return nn.init.uniform_(tensor,-d_rsqrt,d_rsqrt)


@torch.inference_mode()
def _init_random_signs_(tensor:Tenors) -> Tensor:
    return tensor.bernoulli_(0.5).mul(2).add(-1) # [-1,1]

ScalingInitialization = Literal["random-signs","normal"]
ScalingInitialization = Literal[ScalingInitialization,"ones"]

def init_scaling(x:Tensor,distribution:ScalingInitialization, chunks:Optional[list[int]]=None)->Tensor:
    if distribution == 'ones':
        if chunks is not None:
            raise ValueError(
                f'When {distribution=}, chunks must be None'
            )
        init_fn=nn.init.ones_
    elif distribution == "normal":
        init_fn = nn.init.normal_
    elif distribution == "random-signs":
        init_fn = _init_random_signs_
    else:
        raise ValueError(f'Unknown distribution : {distribution=}')
    
    if chunks is not None:
        return init_fn(x)
    else:
        if x.ndim < 1:
            raise ValueError(
                'When chunks is not None, the input Tensor must have at least 1 dimension,'
                f'however: {x.shape[-1]=} != {sum(chunks)=}'
            )
        if sum(chunks) != x.shape[-1]:
            raise ValueError(
                'The tensor shape and chunks are incompatible:'
                f'{x.shape[-1]=} != {sum(chunks)=}'
            )
        # all values with one chunk are initialized to the same (random) value for generalization
        with torch.inference_mode():
            chunk_start = 0
            for chunk_size in chunks:
                x[..., chunk_start: chunk_start + chunk_size] = init_fn(
                    torch.empty(*x.shape[:1],1)
                )
        return x


#==========================================================
#Basic modules
#==========================================================

class _OneHotEncoding(nn.Module):
    """One-hot encoding for categorical features.

    **Shape**

    - Input:''(*, n_cat_features =len(cardinalities))'' 
      where * is an arbitrary bactch size and n_cat_features is the number of categorical features
    - Output''(*, sum(cardinalities))''

    """
    def __init__(self, cardinalities:List[int])->None:
        if not cardinalities:
            raise ValueError(f"cardinalities must be a non-empty list, however : {cardinalities}")
            for i, cardinality in enumerate(cardinalities):
                if cardinality <= 0:
                    raise ValueError(
                        'cardinalities must be a list of positve intergers,'
                        f'however:{cardinalities=}'
                    )
        super().__init__()
        self._cardinalities = cardinalities

    def get_output_shape(self)->torch.Size:
        return torch.Size((sum(self._cardinalities),))

    def forward(self,x:Tensor)->Tensor:
        """
        x: the categoral features. Data type should be torch.long.
           The i-th feature must take values in ''range(0,cardinalities[i])'',
           where cardinalities is the list passed to the constructor.
        """
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)
        return torch.cat([
            nn.functional.one_hot(x[...,i],cardinality)
            for i, cardinality in enumerate(self._cardinalities)
        ],dim=-1)


class ElementwiseAffine(nn.Module):
    """Elementwise affine transformation.
    
    **Shape**
    -Input: ''(*,*shape)'',
           where * is an arbitrary batch size 
    - Output: ''(*,*shape)'',
    """
    bias: Optional[Tensor]

    def __init__(self, shape: tuple[int,...],
    *,
    bias: bool,
    scaling_init: scalinginitialization,
    scaling_init_chunk: Optional[int]=None,
    dtype:Optional[torch.dtype]=None,
    device: Optional[torch.device]=None)->None:
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight = Parameter(torch.empty(shape,**factory_kwargs))
        self.register_parameter(
            'bias',Parameter(torch.empty(shape,**factory_kwargs)) if bias else None
        )
        self._weight_init = scaling_init
        self._weight_init_chunk = scaling_init_chunk
        self.reset_parameters()

    def reset_paramters(self)->None:
        if self._weight_init == 'random-signs':
            _init_random_signs_(self.weight)
        elif self._weight_init == 'normal':
            nn.init.normal_(self.weight)
        elif self._weight_init == 'ones':
            nn.init.ones_(self.weight)
        else:
            raise ValueError(
                f'Invalid scaling initialization: {self._weight_init}',
            )
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x:Tensor)->Tensor:
        expected_shape = self.weight.shape
        expected_sahpe_ndim = len(expected_shape)
        if x.ndim < expected_sahpe_ndim:
            raise ValueError(
                f'The input must have {expected_sahpe_ndim} dimensions, but got {x.ndim}',
            )
        if x.shape[-expected_sahpe_ndim:] != expected_shape:
            raise ValueError(
                f'The input must have shape {expected_shape}, but got {x.shape}', 
            )
        return (x*self.weight if self.bias is None else torch.addcmul(self.bias, self.weight,x))


#==========================================================
#Ensemble
#==========================================================

def ensemble_view(x:Tensor, k:int, training:bool)->Tensor:
    if x.ndim == 2:
        x=x.unsequeeze(-2).expand(-1,k,-1)
    elif x.ndim == 3:
        if x.shape[-2] != k:
            raise ValueError(
                f'The penultimate dimension of the input must be {k}, but got {x.shape[-2]}')
        if not training:
            warnings.warn(
                'When not training, the input should have the shape (batcj,k),'
                f'however:{x.shape=}')
    else:
        raise ValueError(
            f'The input must have 2 or 3 dimensions, but got {x.ndim}',
        )


class EnsembleView(nn.Module):
    """
    Turn a tensor to a valid ensemble input
    2-dimensional tensor to 3-dimensional tensor.
    k is the ensemble size
    For ensembles of MLP-like models.
    Shoudl be placed right before the first ensemble module.

    **Shape**
    -Input : ''(batch_size, d)'' or ''(bacth_size,k,d)''
    -Output : ''(bach_size,k,d)''
    """
    def __init__(self,*, k:int)->None:
        super().__init__()
        self._k = k

    @property
    def k(self)->int:
        return self._k 
    
    def forward(self,x:Tensor)->Tensor:
        return ensemble_view(x,self.k,self.training)



class LinearEnsemble(nn.Module):
    """
    An ensemble of k linear layers applied in parallel to k inputs. 

    **Shape**
    -Input : ''(batch_size, k, in_features)''
    -Output : ''(batch_size, k, out_features)''

    """
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features:int,
        out_feattures:int,
        bias:bool,
        *,
        k:int,
        dtype:Optional[torch.dtype]=None,
        device:Optional[torch.device]=None)->None:
        """
        k: number of linear layes
        """

        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight =Parameter(torch.empty((k,out_feattures,in_features),**factory_kwargs))
        self.register_parameter(
            'bias',
            Parameter(torch.empty((k,out_feattures),**factory_kwargs)) if bias else None
        )
        self.reset_parameters()
    
    @classmethod
    def from_linear(cls, module:nn.Linear,**kwargs)->'LinearEnsemble':
        """Instance from its non-ensemble version"""
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            **kwargs
        )
    
    @property
    def in_features(self)->int:
        return self.weight.shape[-2]
    
    @property
    def out_features(self)->int:
        return self.weight.shape[-1]

    @property
    def k(self)->int:
        return self.weight.shape[0]

    def reset_parameters(self)->None:
        d = self.in_features
        _init_rsqrt_uniform_(self.weight,d)
        if self.bias is not None:
            _init_rsqrt_uniform_(self.bias,d)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        if x.shape[-2] != self.k:
            raise ValueError(
                f'The penultimate input dimensione must equal to {self.k},'
                f'but got {x.shape[-2]}'
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'The last input dimensione must equal to {self.in_features},'
                f'but got {x.shape-1}'
            )
        x = x.transpose(0,1)
        x = x @ self.weight
        x = x.transpose(0,1)
        if self.bias is not None:
            x = x + self.bias
        return x


class LinearBactchEnsemble(nn.Module):
    """
    Parameter-efficiente ensemble of linear layers.
    Particularity here is the R and S matrices, which are the adapters.

    +*Shape**
    -Input:''(batch_size,k, in_features)''
    -Outpiut: ''(bactch_size,k, out_features)''
    """
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool=True,
        *,
        k: int,
        scaling_init:Union[
            ScalingInitialization, tuple[ScalingInitialization,ScalingInitialization]
        ],
        first_scaling_init_chunk: Optional[list[int]]=None,
        dtype:Optional[torch.dtype]=None,
        device: Optional[Union[str,torch.dtype]]=None
    )->None:
        if in_features <= 0:
            raise ValueError(f'in_features must be positive, but got {in_features}')
        if out_features <= 0:
            raise ValueError(f'in_features must be positive, but got {out_features}')
        if k <= 0:
            raise ValueError(f'in_features must be positive, but got {k}')
        
        super().__init__()
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight = Parameter(
            torch.empty(out_features,in_features,**factory_kwargs)
        )
        self.r = Parameter(
            torch.empty(k,in_features,**factory_kwargs)
        )
        self.s = Parameter(
            torch.empty(k,out_features,**factory_kwargs)
        )
        self.register_parameter(
            'bias',
            Parameter(
                torch.empty(k,out_features,**factory_kwargs)
            )
        )
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        if isinstance(scaling_init,tuple):
            self._first_scaling_init =scaling_init[0]
            self._second_scaling_init = scaling_init[1]
        else:
            self._first_scaling_init = scaling_init
            self._second_scaling_init = scaling_init
        self._first_scaling_init_chunk = first_scaling_init_chunk

        self.reset_parameters()

    @lastmethod
    def from_linear(cls, module:nn.Linear,**kwargs)->'LinearBatchEnsemble':
        """Instance from its non-ensemble version"""
        kwargs.setdefault('dtype', module.weight.dtype)
        kwargs.setdefault('device', module.weight.device)
        return cls(
            module.in_features, module.out_features, module.bias is not None, **kwargs
        )
    
    def reset_parameters(self)->None:
        _init_rsqrt_uniform_(self.weight, self.in_features)
        init_scaling(self.r, self._first_scaling_init, self._first_scaling_init_chunk)
        init_scaling(self.s, self._second_scaling_init, None)
        if self.bias is not None:
            bias_init = torch.empty(
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device
            )
            bias_init = _init_rsqrt_uniform_(bias_init, self.out_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x:Tensor)->Tensor:
        if x.ndim !=3:
            raise ValueError(
                f'The input must have 3 dimensions, but got {x.ndim}'
            )
        x = x * self.r
        x = x @ self.weight.T
        x = x * self.s
        if self.bias is not None:
            x = x + self.bias
        return x