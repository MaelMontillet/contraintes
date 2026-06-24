from collections import OrderedDict
import numpy as np
import tensorflow as tf
from .data_container import index_keys

 
class DataProvider:
    def __init__(self, data_container, ntrain, train=True, batch_size=1,
                 seed=None):
        
        self.data_container = data_container
        self._ndata = len(data_container)

        # ======================================================
        # Splitting adapted to our dataset
        if train:
            # Optionally limit training set to ntrain samples (subset of train_idx)
            full_train_idx = self.data_container.train_idx
            if ntrain is not None and ntrain < len(full_train_idx):
                train_idx = full_train_idx[:ntrain]
            else:
                train_idx = full_train_idx
            self.nsamples = {'train': len(train_idx), 'val': len(self.data_container.val_idx),
                              'test': 0}
        else:
            self.nsamples = {'train': 0, 'val': 0, 'test': self._ndata}
        # ==========================================

        self.batch_size = batch_size

        self._random_state = self.data_container._random_state

        # Store indices of training, validation and test data
        # ======================================================
        # Splitting adapted to our dataset

        if train:
            self.idx = {'train': train_idx,
                        'val': self.data_container.val_idx,
                        'test': None}
        else:
            self.idx = {'train': None,
                        'val': None,
                        'test': np.arange(self._ndata)}
        # ==========================================

        # Weighted sampling: probability ∝ |LR residual| so hard examples
        # appear more frequently. Falls back to uniform if weights unavailable.
        if train and hasattr(self.data_container, 'train_sample_weights'):
            raw_w = self.data_container.train_sample_weights  # shape: (len(full_train_idx),)
            if ntrain is not None and ntrain < len(full_train_idx):
                raw_w = raw_w[:ntrain]
            w = raw_w + raw_w.mean()  # add mean so no sample drops to near-zero weight
            self._train_weights = w / w.sum()
        else:
            self._train_weights = None

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()
        self.dtypes_input['Z'] = tf.int32
        self.dtypes_input['R'] = tf.float32

        for key in index_keys:
            self.dtypes_input[key] = tf.int32
        self.dtype_target = tf.float32
        # Shapes of dataset values
        self.shapes_input = {}
        self.shapes_input['Z'] = [None]
        self.shapes_input['R'] = [None, 3]
        for key in index_keys:
            self.shapes_input[key] = [None]
        
        # ==========================================
        # add id for retrival in test prediction
        self.dtypes_input['id'] = tf.int32
        self.shapes_input['id'] = [None]

        # add N to compute statistics on the number of atoms
        self.dtypes_input['N'] = tf.int32
        self.shapes_input['N'] = [None]
        # we only predict energy
        self.shape_target = [None, 1] 
        # ==========================================

    def shuffle_train(self):
        """Shuffle the training data (used when weighted sampling is off)."""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return indices for the next batch.

        Training uses weighted sampling (with replacement) when weights are
        available, so every batch is an independent draw and epochs are
        virtual (nsamples['train'] steps = one pass through the data by count).
        Validation/test use sequential iteration as before.
        """
        if split == 'train' and self._train_weights is not None:
            # Weighted sampling with replacement: harder molecules appear more often
            return self._random_state.choice(
                self.idx['train'], size=self.batch_size, replace=True,
                p=self._train_weights)

        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch (uniform mode only)
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]
        return self.idx[split][start:end]

    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['targets'], dtype=tf.float32))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = tf.constant(batch['targets'], dtype=tf.float32)
            return (inputs, np.expand_dims(targets, axis=-1)) # expand_dim added

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""
        def generator():
            while True:
                idx = self.get_batch_idx(split)
                yield self.idx_to_data(idx)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=(dict(self.dtypes_input), self.dtype_target),
                output_shapes=(self.shapes_input, self.shape_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=tf.int32,
                output_shapes=[None])

    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        dtypes_flattened = list(self.dtypes_input.values())
        dtypes_flattened.append(self.dtype_target)

        inputs_targets = tf.py_function(lambda idx: self.idx_to_data(idx.numpy(), return_flattened=True),
                                        inp=[idx], Tout=dtypes_flattened)

        inputs = {}
        for i, key in enumerate(self.dtypes_input.keys()):
            inputs[key] = inputs_targets[i]
            inputs[key].set_shape(self.shapes_input[key])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return (inputs, targets)
