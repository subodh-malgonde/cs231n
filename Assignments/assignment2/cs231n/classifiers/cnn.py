import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import affine_batchnorm_relu_forward, affine_batchnorm_relu_backward


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - batchnorm(?) - relu - 2x2 max pool - affine - batchnorm(?) - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               use_batchnorm=False, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.random.normal(0, weight_scale, (num_filters,))


    # Dimensions of output layer of convolution
    # H_ = 1 + (H-filter_size + 2*pad)/stride
    # W_ = 1 + (W-filter_size + 2*pad)/stride
    # pad and stride is chosen such that H_ = H and W_ = W
    # 2x2 max pooling layer causes the output height to reduce by half
    H_ = H/2
    W_ = W/2

    # Dimensions of the input to the first hidden layer
    N1 = num_filters*H_*W_

    self.params['W2'] = np.random.normal(0, weight_scale, (N1, hidden_dim))
    self.params['b2'] = np.random.normal(0, weight_scale, (hidden_dim,))

    if self.use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters, )
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(hidden_dim,)
      self.params['beta2'] = np.zeros(hidden_dim,)
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.random.normal(0, weight_scale, (num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    mode = 'test' if y is None else 'train'

    if self.use_batchnorm:
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
      a, cache1 = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param, gamma1, beta1, self.bn_params[0])
      a1, cache2 = affine_batchnorm_relu_forward(a, W2, b2, gamma2, beta2, self.bn_params[1])
    else:
      a, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      a1, cache2 = affine_relu_forward(a, W2, b2)

    scores, cache3 = affine_forward(a1, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    dout, grads['W3'], grads['b3'] = affine_backward(dout, cache3)

    if self.use_batchnorm:
      dout, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_batchnorm_relu_backward(dout, cache2)
      dx, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dout, cache1)
    else:
      dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache2)
      dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache1)

    #regularization component of loss
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

    #regularization component of gradient
    grads['W1'] += self.reg*W1
    grads['W2'] += self.reg*W2
    grads['W3'] += self.reg*W3


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class FourlayerLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  (conv - batchnorm - relu - 2x2 max pool)x2 - affine - batchnorm - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 16], filter_size=[7, 5],
               use_batchnorm=False, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters[0], C, filter_size[0], filter_size[0]))
    self.params['b1'] = np.random.normal(0, weight_scale, (num_filters[0],))

    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters[1], num_filters[0], filter_size[1], filter_size[1]))
    self.params['b2'] = np.random.normal(0, weight_scale, (num_filters[1],))

    # Dimensions of output of the second convolution layer
    # H_ = 1 + (H-filter_size + 2*pad)/stride
    # W_ = 1 + (W-filter_size + 2*pad)/stride
    # pad and stride is chosen such that H_ = H and W_ = W
    # 2x2 max pooling layer causes the output height to reduce by half
    H_ = H / 4
    W_ = W / 4

    # Output of the first convolutional layer has dimensions
    N1 = num_filters[1] * H_ * W_

    self.params['W3'] = np.random.normal(0, weight_scale, (N1, hidden_dim))
    self.params['b3'] = np.random.normal(0, weight_scale, (hidden_dim,))

    if self.use_batchnorm:
      self.params['gamma1'] = np.ones(num_filters, )
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(hidden_dim, )
      self.params['beta2'] = np.zeros(hidden_dim, )
      self.params['gamma3'] = np.ones(hidden_dim, )
      self.params['beta3'] = np.zeros(hidden_dim, )
      self.bn_params = [{'mode': 'train'} for i in xrange(3)]

    self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b4'] = np.random.normal(0, weight_scale, (num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    mode = 'test' if y is None else 'train'

    if self.use_batchnorm:
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      gamma3, beta3 = self.params['gamma3'], self.params['beta3']
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # pass conv_param to the forward pass for the convolutional layer
    filter_sizes = [W1.shape[2], W2.shape[2]]
    conv_params = [
      {'stride': 1, 'pad': (filter_sizes[0] - 1) / 2},
      {'stride': 1, 'pad': (filter_sizes[1] - 1) / 2}
    ]

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
      a, cache1 = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_params[0], pool_param, gamma1, beta1, self.bn_params[0])
      a1, cache2 = conv_batchnorm_relu_pool_forward(a, W2, b2, conv_params[1], pool_param, gamma2, beta2, self.bn_params[1])
      a2, cache3 = affine_batchnorm_relu_forward(a1, W3, b3, gamma3, beta3, self.bn_params[2])
    else:
      a, cache1 = conv_relu_pool_forward(X, W1, b1, conv_params[0], pool_param)
      a1, cache2 = conv_relu_pool_forward(a, W2, b2, conv_params[1], pool_param)
      a2, cache3 = affine_relu_forward(a1, W3, b3)

    scores, cache4 = affine_forward(a2, W4, b4)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    dout, grads['W4'], grads['b4'] = affine_backward(dout, cache4)

    if self.use_batchnorm:
      dout, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = affine_batchnorm_relu_backward(dout, cache3)
      dout, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_batchnorm_relu_pool_backward(dout, cache2)
      dx, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dout, cache1)
    else:
      dout, grads['W3'], grads['b3'] = affine_relu_backward(dout, cache3)
      dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, cache2)
      dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache1)

    # regularization component of loss
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3) + np.sum(W4 * W4))

    # regularization component of gradient
    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3
    grads['W4'] += self.reg * W4

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a1, spatial_batchnorm_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, spatial_batchnorm_cache, relu_cache, pool_cache)
  return out, cache


def conv_batchnorm_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, spatial_batchnorm_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da1 = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da1, spatial_batchnorm_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta