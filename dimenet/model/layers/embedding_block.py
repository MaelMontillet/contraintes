import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..initializers import GlorotOrthogonal


"""ATOM_FEATURES = {
    1:  [0.500, 0.000, 0.000, 0.000, 0.000, 0.550, 0.124, 0.544, 0.189, 0.125, 0.000, 0.000],   # H
    2:  [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.112, 0.983, 0.000, 0.250, 0.000, 1.000],   # He
    6:  [1.000, 1.000, 0.333, 0.000, 0.000, 0.637, 0.308, 0.450, 0.315, 0.500, 0.167, 0.765],   # C
    7:  [1.000, 1.000, 0.500, 0.000, 0.000, 0.760, 0.300, 0.581, 0.000, 0.625, 0.167, 0.824],   # N
    8:  [1.000, 1.000, 0.667, 0.000, 0.000, 0.860, 0.292, 0.545, 0.365, 0.750, 0.167, 0.882],   # O
    11: [1.000, 1.000, 1.000, 0.500, 0.000, 0.233, 0.616, 0.206, 0.137, 0.125, 0.333, 0.000],  # Na
    16: [1.000, 1.000, 1.000, 1.000, 0.667, 0.645, 0.412, 0.414, 0.519, 0.750, 0.333, 0.882],  # S
    17: [1.000, 1.000, 1.000, 1.000, 0.833, 0.790, 0.396, 0.519, 0.903, 0.875, 0.333, 0.941],  # Cl
}"""

 
class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_init = GlorotOrthogonal()

        # ============================
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))

        
        self.embeddings = self.add_weight(name="embeddings", shape=(18, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)
        
        """# =================
        # Attempt with physical properties instead of embedings
        # Original : Atom embeddings: We go up to Pu (94). Use 95 dimensions because of 0-based indexing
        # Modified to go to Cl (17)
        initial_embed_size = 64
        initial = emb_init(shape=(18, initial_embed_size), dtype=tf.float32).numpy()

        for z, feat in ATOM_FEATURES.items():
            initial[z, :12] = feat

        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(18, initial_embed_size),
            initializer=tf.constant_initializer(initial),
            trainable=True,
            dtype=tf.float32
        )
        #================== """ # Doesn't work

        self.dense_rbf = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                      kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.emb_size, activation=activation, use_bias=True,
                                  kernel_initializer=self.weight_init)

    def call(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)

        Z_i = tf.gather(Z, idnb_i)
        Z_j = tf.gather(Z, idnb_j)

        x_i = tf.gather(self.embeddings, Z_i)
        x_j = tf.gather(self.embeddings, Z_j)

        x = tf.concat([x_i, x_j, rbf], axis=-1)
        x = self.dense(x)
        return x
