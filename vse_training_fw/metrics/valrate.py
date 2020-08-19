import tensorflow as tf
import numpy as np


class ValidationRate(tf.keras.metrics.Metric):
    """The implementation of VAL and FAR metrics for triplets.
    
    See:
        [FaceNet: A Unified Embedding for Face Recognition and Clustering]
        (https://arxiv.org/abs/1503.03832)
    """
    def __init__(self, name="val", treshold=0.02, false_accepts=False):
        super(ValidationRate, self).__init__(name=name)
        self.accept_rate = self.add_weight(name="val", initializer="zeros")
        self.treshold = treshold
        self.step = self.add_weight(name="step", initializer="zeros")
        self.false_accepts = false_accepts


    def l2d(self, x1, x2):
        return tf.math.sqrt(tf.math.squared_difference(x1, x2))[0]


    def update_state(self, y_true, y_pred, sample_weight=None):
        vectors = y_pred
        labels = y_true
        batch_size = tf.shape(labels)[0]

        labels = tf.reshape(labels, [batch_size, 1])
        
        same_mask = tf.math.equal(labels, tf.transpose(labels))

        if self.false_accepts:
            diff_mask = tf.math.logical_not(same_mask)
            mask = diff_mask
        else:
            mask = same_mask

        pairs = tf.constant(0)
        accepts = tf.constant(0)

        for i in range(batch_size):
            pair_vector = vectors[i]

            row_mask = mask[i][i+1:]
            row_mask = tf.pad(row_mask, [[i+1, 0]])

            masked_vectors = tf.boolean_mask(vectors, row_mask)
            masked_vectors_size = tf.shape(masked_vectors)[0]

            pairs = tf.add(pairs, masked_vectors_size)

            for j in range(masked_vectors_size):
                d = self.l2d(pair_vector, masked_vectors[j])
                accepts = tf.cond(
                    d <= self.treshold, 
                    lambda: tf.add(accepts, 1), 
                    lambda: accepts
                )

        accept_rate = tf.math.divide(accepts, pairs)
        accept_rate = tf.dtypes.cast(accept_rate, tf.float32)
        
        self.step.assign_add(1)
        
        self.accept_rate.assign_add(accept_rate)


    def result(self):
        return tf.math.divide(self.accept_rate, self.step)


    def reset_state(self):
        self.accept_rate.assign(0)
        self.step = tf.constant(0, tf.float32)
