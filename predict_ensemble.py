#!/usr/bin/env python3
"""
Ensemble prediction from a K-fold training run.

Usage:
    python predict_ensemble.py --run_dir ../checkpoints/<run_id>_kfold5
    python predict_ensemble.py --run_dir ... --config config_kfold.yaml

Loads each fold's best checkpoint and scaler, runs inference on the test set,
then saves two submission files:
  ../submission/ensemble_<run_id>_uniform.csv   ← simple average
  ../submission/ensemble_<run_id>_weighted.csv  ← weighted by 1/val_MAE
"""

import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from custom.data_container import DataContainer, get_atom_count
from custom.data_provider import DataProvider


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
        dropout_rate=0.0,   # no dropout at inference time
    )


def predict_fold(fold_dir, cfg, dc_test, dp_test, data_root):
    """
    Load fold checkpoint + scaler, run inference on the test DataContainer,
    return (predictions array of shape [n_test], val_mae).
    """
    best_dir   = os.path.join(fold_dir, 'best')
    ckpt_file  = os.path.join(best_dir, 'ckpt')
    scaler_file = os.path.join(best_dir, 'scaler.pkl')
    loss_file   = os.path.join(best_dir, 'best_loss.npz')

    # Load scaler (may be None if scale_target=False)
    scaler = None
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

    # Val MAE for weighting
    val_mae = np.inf
    if os.path.exists(loss_file):
        d = np.load(loss_file)
        val_mae = float(d.get('mean_mae_val', np.inf))

    tf.keras.backend.clear_session()
    model = _build_model(cfg)

    # Build the model by running one forward pass
    batch_size = cfg['batch_size']
    test_iter  = iter(dp_test.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE))
    inputs, _  = next(test_iter)
    _ = model(inputs, training=False)  # build

    model.load_weights(ckpt_file)

    # Collect predictions over the full test set
    n_test     = dp_test.nsamples['test']
    n_batches  = int(np.ceil(n_test / batch_size))
    preds_list = []
    ids_list   = []

    # Reset iterator
    test_iter = iter(dp_test.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE))
    for _ in range(n_batches):
        inputs, _ = next(test_iter)
        preds = model(inputs, training=False)
        N = inputs['N'].numpy()
        Z = inputs['Z'].numpy()
        if scaler is not None:
            atom_count = get_atom_count(Z, N)
            preds = scaler.inverse_transform(atom_count, preds)
        else:
            preds = preds.numpy()
        preds_list.extend(preds.squeeze())
        ids_list.extend(inputs['id'].numpy())

    preds_arr = np.array(preds_list[:n_test])
    ids_arr   = np.array(ids_list[:n_test])
    return ids_arr, preds_arr, val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Path to the kfold run directory (contains fold_0/ fold_1/ …)')
    parser.add_argument('--config', default='config_kfold.yaml')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of folds (auto-detected from run_dir if omitted)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = "/home/n7student/Documents/2A/S2/contraintes/projet_kaggle/upload data_modia-phyml-26"
    run_dir   = args.run_dir

    # Auto-detect number of folds
    if args.n_folds is not None:
        n_folds = args.n_folds
    else:
        n_folds = sum(1 for d in os.listdir(run_dir)
                      if d.startswith('fold_') and os.path.isdir(os.path.join(run_dir, d)))
    print(f"Ensembling {n_folds} folds from {run_dir}")

    # Load test DataContainer once (no scaler needed here — each fold's scaler handles it)
    dc_test = DataContainer(data_root, cfg['cutoff'], train=False)
    dp_test = DataProvider(dc_test, ntrain=None, train=False, batch_size=cfg['batch_size'])

    # Collect predictions from each fold
    all_preds = []
    all_ids   = None
    val_maes  = []

    for k in range(n_folds):
        fold_dir = os.path.join(run_dir, f'fold_{k}')
        print(f"  Fold {k+1}/{n_folds} …", end=' ', flush=True)
        ids, preds, val_mae = predict_fold(fold_dir, cfg, dc_test, dp_test, data_root)
        print(f"val_MAE={val_mae:.5f}")

        # Sort by ID so all folds align
        order    = np.argsort(ids)
        ids      = ids[order]
        preds    = preds[order]

        if all_ids is None:
            all_ids = ids
        else:
            assert np.array_equal(all_ids, ids), "Test molecule order differs between folds!"

        all_preds.append(preds)
        val_maes.append(val_mae)

    all_preds = np.stack(all_preds, axis=0)   # shape: (n_folds, n_test)

    # ── Uniform average ──
    preds_uniform = all_preds.mean(axis=0)

    # ── Weighted average (weight ∝ 1/val_MAE, better model gets more weight) ──
    finite_maes = [m for m in val_maes if np.isfinite(m)]
    if len(finite_maes) == n_folds:
        weights     = np.array([1.0 / m for m in val_maes])
        weights    /= weights.sum()
        preds_weighted = (all_preds * weights[:, None]).sum(axis=0)
        print(f"\nWeights: {[f'{w:.3f}' for w in weights]}")
    else:
        preds_weighted = preds_uniform
        print("\nSome val_MAEs missing — using uniform weights for both outputs")

    # ── Save submissions ──
    run_name = os.path.basename(run_dir.rstrip('/'))
    sub_dir  = os.path.join(HERE, '..', 'submission')
    os.makedirs(sub_dir, exist_ok=True)

    for tag, preds in [('uniform', preds_uniform), ('weighted', preds_weighted)]:
        path = os.path.join(sub_dir, f"ensemble_{run_name}_{tag}.csv")
        df   = pd.DataFrame({'id': all_ids, 'energy': preds})
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    print(f"\nPrediction summary (uniform ensemble):")
    print(f"  mean={preds_uniform.mean():.3f}  std={preds_uniform.std():.3f}")
    print(f"  min={preds_uniform.min():.3f}  max={preds_uniform.max():.3f}")


if __name__ == '__main__':
    main()
