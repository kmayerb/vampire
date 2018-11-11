from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class EmbedViaMatrix(Layer):
    """
    This layer defines a (learned) matrix M such that given matrix input X the
    output is XM. The number of columns of M is embedding_dim, and the number
    of rows is set so that X and M can be multiplied.

    If the rows of the input give the coordinates of a series of objects, we
    can think of this layer as giving an embedding of each of the encoded
    objects in a embedding_dim-dimensional space.
    """

    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super(EmbedViaMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # The first component of input_shape is the batch size (see https://keras.io/layers/core/#dense).
        self.kernel = self.add_weight(
            name='kernel', shape=(input_shape[2], self.embedding_dim), initializer='uniform', trainable=True)
        super(EmbedViaMatrix, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)


class RightTensordot1(Layer):
    """
    Given a numpy tensor Y, this layer tensordots input on the right with Y
    over a single coordinate.

    That is, if the input is X, this performs sum_i X_{...,i} Y_{i,...}.
    """

    def __init__(self, right_np_tensor, **kwargs):
        self.right_tf_tensor = tf.convert_to_tensor(right_np_tensor, dtype=tf.float32)
        super(RightTensordot1, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RightTensordot1, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # Tensordotting with axes=1 sums over the last index of the first argument
        # and the first index of the second argument.
        return tf.tensordot(x, self.right_tf_tensor, axes=1)

    def compute_output_shape(self, input_shape):
        # Make sure that the last dimension of the first argument matches the
        # first dimension of the second argument.
        assert input_shape[-1] == self.right_tf_tensor.shape[0]
        return tuple(input_shape[:-1] + tuple(self.right_tf_tensor.shape[1:]))