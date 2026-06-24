#!/usr/bin/env python3
"""
5-fold cross-validation training for DimeNet++.

Run from the dimenet/ directory:
    python train_kfold.py                         # uses config_kfold.yaml
    python train_kfold.py --config my_config.yaml

Each fold is saved under:
    <logdir>/<timestamp>_<uid>_<comment>/fold_<k>/
        best/ckpt          ← best checkpoint (by val MAE)
        best/best_loss.npz ← best val metrics
        best/scaler.pkl    ← fitted LREnergyScaler for this fold
        logs/              ← step checkpoints + TF summaries

After all folds finish, the script prints the cross-validated MAE and saves
out-of-fold predictions to:
    <logdir>/<timestamp>_<uid>_<comment>/oof_predictions.csv
"""

import os
import sys
import pickle
import string
import random
import logging
import argparse
import yaml
import numpy as np
import tensorflow as tf
from math import ceil
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from custom.data_container import DataContainer, get_atom_count
from custom.data_provider import DataProvider

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    _USE_STRATIFIED = True
except ImportError:
    from sklearn.model_selection import KFold
    _USE_STRATIFIED = False
    logging.warning("iterstrat not found — falling back to sklearn KFold (no atom-type stratification)")


# ── helpers ───────────────────────────────────────────────────────────────────

def _uid(size=8):
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


def _build_model(cfg):
    return DimeNetPP(
        emb_size=cfg['emb_size'],
        out_emb_size=cfg['out_emb_size'],
        int_emb_size=cfg['int_emb_size'],
        basis_emb_size=cfg['basis_emb_size'],
        num_blocks=cfg['num_blocks'],
        num_spherical=cfg['num_spherical'],
        num_radial=cfg['num_radial'],
        cutoff=cfg['cutoff'],
        envelope_exponent=cfg['envelope_exponent'],
        num_before_skip=cfg['num_before_skip'],
        num_after_skip=cfg['num_after_skip'],
        num_dense_output=cfg['num_dense_output'],
        num_targets=1,
        activation=swish,
        extensive=cfg['extensive'],
        output_init=cfg['output_init'],
        dropout_rate=cfg['dropout_rate'],
    )


# ── per-fold training ─────────────────────────────────────────────────────────

def train_fold(fold_idx, fold_train_idx, fold_val_idx, cfg, fold_dir, data_root):
    """Train one fold. Returns best val MAE and out-of-fold predictions."""

    log = logging.getLogger(f"fold{fold_idx}")
    log.info(f"Fold {fold_idx}: {len(fold_train_idx)} train / {len(fold_val_idx)} val")

    # ── directories ──
    best_dir = os.path.join(fold_dir, 'best')
    log_dir  = os.path.join(fold_dir, 'logs')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    best_ckpt_file = os.path.join(best_dir, 'ckpt')
    best_loss_file = os.path.join(best_dir, 'best_loss.npz')
    scaler_file    = os.path.join(best_dir, 'scaler.pkl')

    # ── data ──
    dc = DataContainer(
        data_root, cfg['cutoff'],
        train=True,
        scale_target=cfg['target_scaling'],
        seed=cfg['data_seed'],
        fold_train_idx=fold_train_idx,
        fold_val_idx=fold_val_idx,
    )

    # Save the scaler so predict_ensemble.py can inverse-transform test preds
    if dc.scaler is not None:
        with open(scaler_file, 'wb') as f:
            pickle.dump(dc.scaler, f)

    batch_size = cfg['batch_size']
    dp = DataProvider(dc, ntrain=None, train=True, batch_size=batch_size)

    num_train     = dp.nsamples['train']
    steps_per_epoch = ceil(num_train / batch_size)
    num_steps     = ceil(cfg['epochs'] * num_train / batch_size)
    warmup_steps  = int(cfg['warmup_prop'] * num_steps)
    decay_steps   = int(cfg['decay_prop']  * num_steps)

    eval_interval = cfg.get('evaluation_interval') or steps_per_epoch
    save_interval = cfg.get('save_interval')       or steps_per_epoch
    patience      = cfg.get('patience_epochs', 60) * steps_per_epoch

    train_ds = dp.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
    val_ds   = dp.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
    train_iter = iter(train_ds)
    val_iter   = iter(val_ds)

    # ── model + trainer ──
    model = _build_model(cfg)
    trainer = Trainer(
        model,
        learning_rate=cfg['learning_rate'],
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_rate=cfg['decay_rate'],
        ema_decay=cfg['ema_decay'],
        max_grad_norm=cfg['max_grad_norm'],
    )

    train_metrics = Metrics('train', ['energy'])
    val_metrics   = Metrics('val',   ['energy'])

    # ── checkpoint manager ──
    ckpt     = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
    manager  = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3)

    summary_writer = tf.summary.create_file_writer(log_dir)

    # ── best-loss tracker ──
    metrics_best = {k: np.inf for k in val_metrics.result()}
    metrics_best['step'] = 0
    np.savez(best_loss_file, **metrics_best)

    steps_since_improvement = 0

    # ── training loop ──
    with summary_writer.as_default():
        for step in range(1, num_steps + 1):
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            trainer.train_on_batch(train_iter, train_metrics)

            if step % save_interval == 0:
                manager.save()

            if step % eval_interval == 0:
                trainer.save_variable_backups()
                trainer.load_averaged_variables()

                n_val_batches = ceil(dp.nsamples['val'] / batch_size)
                for _ in range(n_val_batches):
                    trainer.test_on_batch(val_iter, val_metrics)

                val_result = val_metrics.result()
                current_mae = val_result['mean_mae_val']

                if current_mae < metrics_best['mean_mae_val']:
                    metrics_best.update(val_result)
                    metrics_best['step'] = step
                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += eval_interval

                epoch = step // steps_per_epoch
                log.info(
                    f"  step {step}/{num_steps} (epoch {epoch}) | "
                    f"train={train_metrics.loss:.5f} | "
                    f"val={current_mae:.5f} | "
                    f"best={metrics_best['mean_mae_val']:.5f}"
                )

                val_metrics.write()
                val_metrics.reset_states()
                train_metrics.write()
                train_metrics.reset_states()

                trainer.restore_variable_backups()

                if cfg.get('early_stop') and steps_since_improvement >= patience:
                    log.info(f"  Early stop at step {step} (no improvement for {patience} steps)")
                    break

    # ── collect out-of-fold predictions ──
    log.info("Computing out-of-fold predictions …")
    model.load_weights(best_ckpt_file)
    trainer.load_averaged_variables()

    val_preds, val_ids, val_true = [], [], []
    n_val_batches = ceil(dp.nsamples['val'] / batch_size)
    for _ in range(n_val_batches):
        inputs, targets = next(val_iter)
        preds = model(inputs, training=False)
        N = inputs['N'].numpy()
        Z = inputs['Z'].numpy()
        if dc.scaler is not None:
            atom_count = get_atom_count(Z, N)
            preds  = dc.scaler.inverse_transform(atom_count, preds)
            targets_np = dc.scaler.inverse_transform(atom_count, targets.numpy())
        else:
            preds      = preds.numpy()
            targets_np = targets.numpy()
        val_preds.extend(preds.squeeze())
        val_true.extend(targets_np.squeeze())
        val_ids.extend(inputs['id'].numpy())

    val_preds = np.array(val_preds[:dp.nsamples['val']])
    val_true  = np.array(val_true [:dp.nsamples['val']])
    val_ids   = np.array(val_ids  [:dp.nsamples['val']])
    oof_mae   = np.mean(np.abs(val_preds - val_true))
    log.info(f"  OOF MAE (unscaled) = {oof_mae:.5f}")

    best_scaled_mae = float(metrics_best['mean_mae_val'])
    return best_scaled_mae, oof_mae, val_ids, val_preds, val_true


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_kfold.yaml')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('main')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root  = "/home/n7student/Documents/2A/S2/contraintes/projet_kaggle/upload data_modia-phyml-26"
    n_folds    = cfg.get('n_folds', 5)
    data_seed  = cfg.get('data_seed', 42)
    comment    = cfg.get('comment', 'kfold')

    run_id  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uid()}_{comment}"
    run_dir = os.path.join(cfg['logdir'], run_id)
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    # ── compute fold splits on the full training set ──
    # Load just the N and Z arrays (no model) to get atom-type presence
    from custom.data_container import DataContainer as DC
    ids_all, N_all, Z_all, _ = DC.parse_dataset(data_root, 'train')

    from custom.data_container import get_atom_count
    atom_count_all    = get_atom_count(Z_all, N_all)
    atom_presence_all = atom_count_all > 0

    n_total = len(N_all)
    rng = np.random.RandomState(data_seed)

    if _USE_STRATIFIED:
        kf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_seed)
        splits = list(kf.split(np.zeros(n_total), atom_presence_all))
        log.info(f"Using MultilabelStratifiedKFold (atom-type balanced)")
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=data_seed)
        splits = list(kf.split(np.zeros(n_total)))
        log.info(f"Using KFold (no stratification)")

    # ── train each fold ──
    all_oof_ids, all_oof_preds, all_oof_true = [], [], []
    fold_maes = []

    for fold_k, (train_idx, val_idx) in enumerate(splits):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_k + 1}/{n_folds}")
        log.info(f"{'='*60}")

        fold_dir = os.path.join(run_dir, f"fold_{fold_k}")

        tf.keras.backend.clear_session()  # free GPU memory between folds

        best_mae, oof_mae, oof_ids, oof_preds, oof_true = train_fold(
            fold_k, train_idx, val_idx, cfg, fold_dir, data_root
        )

        fold_maes.append(oof_mae)
        all_oof_ids.extend(oof_ids)
        all_oof_preds.extend(oof_preds)
        all_oof_true.extend(oof_true)

        log.info(f"Fold {fold_k+1} finished | best_scaled_val={best_mae:.5f} | oof_mae={oof_mae:.5f}")

    # ── cross-validated results ──
    log.info(f"\n{'='*60}")
    log.info("CROSS-VALIDATION RESULTS")
    log.info(f"{'='*60}")
    for k, mae in enumerate(fold_maes):
        log.info(f"  Fold {k+1}: OOF MAE = {mae:.5f}")
    cv_mae = np.mean(fold_maes)
    cv_std = np.std(fold_maes)
    log.info(f"  CV MAE = {cv_mae:.5f} ± {cv_std:.5f}")

    # Recompute global OOF MAE (avoids averaging averages across different fold sizes)
    all_oof_preds = np.array(all_oof_preds)
    all_oof_true  = np.array(all_oof_true)
    global_oof_mae = np.mean(np.abs(all_oof_preds - all_oof_true))
    log.info(f"  Global OOF MAE = {global_oof_mae:.5f}")

    # ── save OOF predictions ──
    import pandas as pd
    oof_df = pd.DataFrame({
        'id':    all_oof_ids,
        'true':  all_oof_true,
        'pred':  all_oof_preds,
        'error': np.abs(all_oof_preds - all_oof_true),
        'fold':  [k for k, (_, val_idx) in enumerate(splits) for _ in val_idx],
    })
    oof_path = os.path.join(run_dir, 'oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)
    log.info(f"OOF predictions saved to {oof_path}")
    log.info(f"Run dir: {run_dir}")

    # Print the run_dir so predict_ensemble.py knows where to find fold checkpoints
    print(f"\nRUN_DIR={run_dir}")


if __name__ == '__main__':
    main()
