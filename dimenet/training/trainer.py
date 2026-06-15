import tensorflow as tf
import tensorflow_addons as tfa
from .schedules import LinearWarmupExponentialDecay



class WarmupCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom scheduler : Linear warmup, then cosine decay with restarts."""

    def __init__(
        self,
        initial_learning_rate: float,
        first_decay_steps: int,
        warmup_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.warmup_steps = warmup_steps

        self._cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        # Linear ramp from 0 → initial_learning_rate
        warmup_lr = self.initial_learning_rate * (step / warmup_steps)

        # Cosine restarts (offset so cycle starts after warmup)
        cosine_lr = self._cosine(step - warmup_steps)

        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

    def get_config(self):
        config = self._cosine.get_config()
        config.update({"warmup_steps": self.warmup_steps})
        return config




class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        #===================================================================================
        # Changed scheduling for more explorations
        if warmup_steps is not None:
            self.learning_rate = WarmupCosineDecayRestarts(learning_rate, decay_steps, warmup_steps, alpha=decay_rate)
        else:
            raise ValueError()

        # Changed Adam to AdamW
        #opt = tfa.optimizers.AdamW(learning_rate=self.learning_rate, amsgrad=True, weight_decay=1e-5)
        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        #===================================================================================
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_weights)

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self.model.trainable_weights)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
            mean_mae = tf.reduce_mean(mae)
            loss = mean_mae
        self.update_weights(loss, tape)

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)

        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae
        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return preds, inputs["id"]
