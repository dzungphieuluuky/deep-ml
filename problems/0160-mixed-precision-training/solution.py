import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        # Initialize loss scaling factor
        self.loss_scaling = loss_scale
    
    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, return scaled loss as float32
        weights_fp16 = np.array(weights, dtype=np.float16)
        inputs_fp16 = np.array(inputs, dtype=np.float16)
        targets_fp16 = np.array(targets, dtype=np.float16)
        pred = inputs_fp16 @ weights_fp16
        loss = np.mean((pred - targets_fp16)**2) * self.loss_scaling
        return float(loss)
    
    def backward(self, gradients):
        # Unscale gradients and check for overflow, return as float32
        gradients_fp32 = np.array(gradients, dtype=np.float32)
        gradients_fp32 = np.where(np.isnan(gradients_fp32), 0.0, gradients_fp32)
        gradients_fp32 = np.where(np.isinf(gradients_fp32), 0.0, gradients_fp32)
        return gradients_fp32 / self.loss_scaling
