# Inherit from Function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class c_EELU(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, input, pa, pb, odd, eps, training):

        neg = (input < 0)

        pa = pa.clamp(min=0)
        pb = pb.clamp(min=0)

        if training:  # 학습
            if odd:
                k = torch.cuda.FloatTensor([1.])

                output = torch.where(neg, pa * (torch.expm1(pb * input)), input)

                ctx.save_for_backward(input, pa, pb, k)
                ctx.odd = odd
            else:
                sigma = np.random.uniform(0, eps)
                k = torch.cuda.FloatTensor(input.shape).normal_(mean=1, std=sigma).clamp(0, 2)
                # k = torch.cuda.FloatTensor(input.shape).uniform_(1-eps, 1+eps)

                ctx.save_for_backward(input, pa, pb, k)
                ctx.odd = odd

                output = torch.where(neg, pa * (torch.expm1(pb * input)), input * k)


        else:  # 테스트
            # k = torch.cuda.FloatTensor(input.shape).uniform_(1, 1)
            output = torch.where(neg, pa * (torch.expm1(pb * input)), input)

        output = torch.where(torch.isnan(output), torch.cuda.FloatTensor([0.]), output)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, pa, pb, k = ctx.saved_variables
        odd = ctx.odd

        neg = (input < 0)

        if odd:
            if pa.size(0) == 1:
                grad_pa = torch.sum(
                    torch.where(neg, grad_output * (torch.expm1(pb * input)), torch.cuda.FloatTensor([0.]))).view(-1)
                grad_pb = torch.sum(
                    torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, torch.cuda.FloatTensor([0.]))).view(-1)
            else:
                grad_pa = torch.sum(
                    torch.where(neg, grad_output * (torch.expm1(pb * input)), torch.cuda.FloatTensor([0.])),
                    dim=(0, 2, 3)).view(-1, 1, 1)
                grad_pb = torch.sum(
                    torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, torch.cuda.FloatTensor([0.])),
                    dim=(0, 2, 3)).view(-1, 1, 1)

            grad_pa = torch.where(torch.isnan(grad_pa), torch.cuda.FloatTensor([0.]), grad_pa)
            grad_pb = torch.where(torch.isnan(grad_pb), torch.cuda.FloatTensor([0.]), grad_pb)

            grad_input = torch.where(neg, grad_output * pa * torch.exp(pb * input) * pb, grad_output)

        else:
            if pa.size(0) == 1:
                grad_pa = torch.cuda.FloatTensor(1).fill_(0)
                grad_pb = torch.cuda.FloatTensor(1).fill_(0)
            else:
                grad_pa = torch.cuda.FloatTensor(pa.size(0), 1, 1).fill_(0)
                grad_pb = torch.cuda.FloatTensor(pb.size(0), 1, 1).fill_(0)

            grad_input = torch.where(neg, grad_output * pa * torch.exp(pb * input) * pb, k * grad_output)

        grad_input = torch.where(torch.isnan(grad_input), torch.cuda.FloatTensor([0.]), grad_input)

        return grad_input, grad_pa, grad_pb, None, None, None


class EELU(torch.nn.Module):
    """
    Linear neural network module based on the operation defined above.
    """

    def __init__(self, num_parameters=1, pa_init=0.25, pb_init=1, eps=1.0):
        super(EELU, self).__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.odd = True
        self.pa_init = pa_init
        self.pb_init = pb_init
        
        if self.num_parameters == 1:
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters).fill_(pa_init))
            self.pb = nn.Parameter(torch.cuda.FloatTensor(num_parameters).fill_(pb_init))
        else:
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters, 1, 1).fill_(pa_init))
            self.pb = nn.Parameter(torch.cuda.FloatTensor(num_parameters, 1, 1).fill_(pb_init))
        # torch.nn.Module.__init__(self)
        # self.register_parameter('k', None)

    def forward(self, input):
        if self.odd:
            self.odd = False
        else:
            self.odd = True

        return c_EELU.apply(input, self.pa, self.pb, self.odd, self.eps, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'num_parameters=' + str(self.num_parameters * 2) + ', eps=' + str(
            self.eps) + ', pa = ' + str(self.pa_init) + ', pb = ' + str(self.pb_init) + ')'

# =============================================================================
# EMPELU
# =============================================================================

class c_EMPELU_back(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, input, pa, pb, eps, training):
        
        neg = (input<0)
        
        if training:
            sigma = np.random.uniform(0, 1)
            k = torch.cuda.FloatTensor(input.shape).normal_(mean=1, std=sigma).clamp(1-eps, 1+eps)
            #k = torch.cuda.FloatTensor(input.shape).exponential_(lambd=eps) + 1
        else:
            #k = torch.cuda.FloatTensor(input.shape).uniform_(1, 1)
            k =  torch.cuda.FloatTensor(input.shape).fill_(1.)
        
        ctx.save_for_backward(input, pa, pb, k)

        output = torch.where(neg, pa*(torch.exp(pb*input) - 1), input * k)
                        
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, pa, pb, k = ctx.saved_variables
        
        neg = (input<0)
                
        if pa.size(0) == 1: 
            grad_pa = torch.sum(torch.where(neg, grad_output * (torch.exp(pb * input) - 1), torch.cuda.FloatTensor([0.]))).view(1,1,1,1)
            grad_pb = torch.sum(torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, torch.cuda.FloatTensor([0.]))).view(1,1,1,1)
        else:
            grad_pa = torch.sum(torch.where(neg, grad_output * (torch.exp(pb * input) - 1), torch.cuda.FloatTensor([0.])), -1).sum(-1).sum(-1).sum(0).view(-1,1,1,1)
            grad_pb = torch.sum(torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, torch.cuda.FloatTensor([0.])), -1).sum(-1).sum(-1).sum(0).view(-1,1,1,1)
        
        grad_input = torch.where(neg, grad_output * pa * torch.exp(pb*input) * pb, k * grad_output)
        
        return grad_input, grad_pa, grad_pb, None, None
    
    
class EMPELU_back(torch.nn.Module):
    """
    Linear neural network module based on the operation defined above.
    """
    def __init__(self, num_parameters=1, pa_init=0, pb_init=0, eps=0.9):
        super(EMPELU_back, self).__init__()
        self.num_parameters = num_parameters
        self.eps = eps
                
        self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters,1,1,1).fill_(pa_init))
        self.pb = nn.Parameter(torch.cuda.FloatTensor(num_parameters,1,1,1).fill_(pb_init))
        #torch.nn.Module.__init__(self)
        #self.register_parameter('k', None)
  
    def forward(self, input):
        #if self.k is None:
        #    self.k = nn.Parameter(torch.FloatTensor(1).fill_(1))        
        return c_EMPELU.apply(input, self.pa, self.pb, self.eps, self.training)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'num_parameters=' + str(self.num_parameters * 2) + ')'
    
    
# =============================================================================
# MPELU
# =============================================================================
class c_MPELU(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, input, pa, pb):
        
        neg = (input<0)
        
        
        ctx.save_for_backward(input, pa, pb)
        
            
#            pa = torch.where(pa<0.1, torch.cuda.FloatTensor([0.1]), pa)
#            pb = torch.where(pb<0.1, torch.cuda.FloatTensor([0.1]), pb)
        
        output = torch.where(neg, pa*(torch.exp(pb*input) - 1), input)
                        
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, pa, pb = ctx.saved_variables
        
        neg = (input<0)
        
        zero_mask = torch.cuda.FloatTensor(input.shape).fill_(0)
                
        if pa.size(0) == 1: 
            grad_pa = torch.sum(torch.where(neg, grad_output * (torch.exp(pb * input) - 1), torch.cuda.FloatTensor([0.]))).view(-1)
            grad_pb = torch.sum(torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, torch.cuda.FloatTensor([0.]))).view(-1)
        else:                    
            grad_pa = torch.where(neg, grad_output * (torch.exp(pb * input) - 1), zero_mask)
            grad_pa = grad_pa.sum((0,2,3)).view(-1,1,1)

            grad_pb = torch.where(neg, grad_output * pa * torch.exp(pb * input) * input, zero_mask)
            grad_pb = grad_pb.sum((0,2,3)).view(-1,1,1)
            
        grad_input = torch.where(neg, grad_output * pa * torch.exp(pb*input) * pb, grad_output)
        
        return grad_input, grad_pa, grad_pb, None, None
    
    
class MPELU(torch.nn.Module):
    """
    Linear neural network module based on the operation defined above.
    """
    def __init__(self, num_parameters=1, pa_init=1, pb_init=1):
        super(MPELU, self).__init__()
        self.num_parameters = num_parameters
        
        if self.num_parameters == 1:
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters).fill_(pa_init))
            self.pb = nn.Parameter(torch.cuda.FloatTensor(num_parameters).fill_(pb_init))
        else:
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters,1,1).fill_(pa_init))
            self.pb = nn.Parameter(torch.cuda.FloatTensor(num_parameters,1,1).fill_(pb_init))
        #torch.nn.Module.__init__(self)
        #self.register_parameter('k', None)
  
    def forward(self, input):
        #if self.k is None:
        #    self.k = nn.Parameter(torch.FloatTensor(1).fill_(1))
        return c_MPELU.apply(input, self.pa, self.pb)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'num_parameters=' + str(self.num_parameters * 2) + ')'
    
    
# =============================================================================
# EReLU
# =============================================================================
class c_EReLU(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, input, eps, training):
        
        neg = (input<0)
                
        if training:
            k = torch.cuda.FloatTensor(input.shape).uniform_(1-eps, 1+eps)
        else:
            k = torch.cuda.FloatTensor(input.shape).uniform_(1, 1)
            
        ctx.save_for_backward(input, k)
        
        output = torch.where(neg, torch.cuda.FloatTensor([0]), k*input)
                        
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, k = ctx.saved_variables
        
        neg = (input<0)
        
        grad_input = torch.where(neg, torch.cuda.FloatTensor([0]), k * grad_output)
        
        return grad_input, None, None
    
    
class EReLU(torch.nn.Module):
    """
    Linear neural network module based on the operation defined above.
    """
    def __init__(self, eps=0.4):
        super(EReLU, self).__init__()

        self.eps = eps
        #torch.nn.Module.__init__(self)
        #self.register_parameter('k', None)
  
    def forward(self, input):
        #if self.k is None:
        #    self.k = nn.Parameter(torch.FloatTensor(1).fill_(1))
        return c_EReLU.apply(input, self.eps, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
                
# =============================================================================
# EPReLU
# =============================================================================
class c_EPReLU(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    # bias is an optional argument
    @staticmethod
    def forward(ctx, input, pa, eps, odd, training):

        neg = (input < 0)

        if training:
            if odd :
                k = torch.cuda.FloatTensor(input.shape).fill_(1.)
            else:
                k = torch.cuda.FloatTensor(input.shape).uniform_(1 - eps, 1 + eps)
        else:
            # k = torch.cuda.FloatTensor(input.shape).uniform_(1, 1)
            k = torch.cuda.FloatTensor(input.shape).fill_(1.)

        ctx.save_for_backward(input, pa, k)
        ctx.odd = odd

        output = torch.where(neg, pa * input, k * input)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, pa, k = ctx.saved_variables
        odd = ctx.odd

        neg = (input < 0)

        if odd:
            if pa.size(0) == 1:
                grad_pa = torch.sum(torch.where(neg, grad_output * input, torch.cuda.FloatTensor([0.]))).view(-1)
            else:
                grad_pa = torch.sum(torch.where(neg, grad_output * input, torch.cuda.FloatTensor([0.])),dim=(0, 2, 3)).view(-1, 1, 1)
        else:
            if pa.size(0) == 1:
                grad_pa = torch.cuda.FloatTensor(1).fill_(0)
            else:
                grad_pa = torch.cuda.FloatTensor(pa.size(0), 1, 1).fill_(0)

        grad_input = torch.where(neg, pa * grad_output, k * grad_output)
        
        return grad_input, grad_pa, None, None, None

class EPReLU(torch.nn.Module):
    """
    Linear neural network module based on the operation defined above.
    """

    def __init__(self, num_parameters=1, pa_init=0.1, eps=0.4):
        super(EPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.eps = eps
        self.odd = True
        self.pa_init = pa_init
        
        if self.num_parameters == 1:
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters).fill_(pa_init))
        else:        
            self.pa = nn.Parameter(torch.cuda.FloatTensor(num_parameters, 1, 1).fill_(pa_init))
        # torch.nn.Module.__init__(self)
        # self.register_parameter('k', None)

    def forward(self, input):
        if self.odd:
            self.odd = False
        else:
            self.odd = True
        return c_EPReLU.apply(input, self.pa, self.eps, self.odd, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'num_parameters=' + str(self.num_parameters) + ', eps=' + str(
            self.eps) + ', pa = ' + str(self.pa_init) + ')'
                
                
class Swish(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


    def __repr__(self):
        return self.__class__.__name__ + '(inplace=' + str(self.inplace) + ')'