import tensorflow as tf

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

class EspcnNet:
    def __init__(self, filters_size, channels, ratio):
        self.filters_size = filters_size
        self.channels = channels
        self.ratio = ratio
        self.variables = self._create_variables()

    def _create_variables(self):
        var = dict()
        var['filters'] = list()
        # the input layer
        var['filters'].append(
            create_variable('filter',
                            [self.filters_size[0],
                             self.filters_size[0],
                             1,
                             self.channels[0]]))
        # the hidden layers
        for idx in range(1, len(self.filters_size) - 1):
            var['filters'].append(
                create_variable('filter', 
                                [self.filters_size[idx],
                                 self.filters_size[idx],
                                 self.channels[idx - 1],
                                 self.channels[idx]]))
        # the output layer
        var['filters'].append(
            create_variable('filter',
                            [self.filters_size[-1],
                             self.filters_size[-1],
                             self.channels[-1],
                             self.ratio**2]))
        var['biases'] = list()
        for channel in self.channels:
            var['biases'].append(create_bias_variable('bias', [channel]))
        var['biases'].append(create_bias_variable('bias', [self.ratio**2]))
        return var

    def _preprocess(self, input_data):
        # cast to float32 and normalize the data
        data = list()
        for ele in input_data:
            ele = tf.cast(ele, tf.float32)
            ele = ele / 255.0
            data.append(ele)
        return data

    def _create_network(self, input_batch):
        '''The default structure of the network is:

        input (3 channels) ---> 5 * 5 conv (64 channels) ---> 3 * 3 conv (32 channels) ---> 3 * 3 conv (27 channels)

        Where `conv` is 2d convolutions with a non-linear activation (tanh) at the output.
        '''

        current_layer = input_batch

        for idx in range(len(self.filters_size)):
            conv = tf.nn.conv2d(current_layer, self.variables['filters'][idx], [1, 1, 1, 1], padding='VALID')
            with_bias = tf.nn.bias_add(conv, self.variables['biases'][idx])
            if idx == len(self.filters_size) - 1:
                current_layer = with_bias
            else:
                current_layer = tf.nn.tanh(with_bias)
        return current_layer

    def loss(self, input_data):
        input_data = self._preprocess(input_data)
        output = self._create_network(input_data[0][:,:,:,0:1])
        residual = output - input_data[1][:,:,:,0:9]
        loss = tf.square(residual)
        reduced_loss = tf.reduce_mean(loss)
        tf.scalar_summary('loss', reduced_loss)
        return reduced_loss

    def generate(self, lr_image):
        lr_image = self._preprocess([lr_image])[0]
        sr_image = self._create_network(lr_image)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)
        return sr_image


