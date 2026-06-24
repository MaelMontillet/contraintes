# ===========================================
# Student code: adapt dataloading to the project dataset

import os
import csv
from custom.target_scaling import LREnergyScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# Periodic-table mapping from element symbol to atomic number (Z).
# Only elements actually present in the dataset.
ELEMENT_TO_Z = {
    'H': 1,  'C': 6,
    'N': 7,  'O': 8,
    'S': 16, 'Cl': 17
}
ELEMENT_LIST = [1, 6, 7, 8, 16, 17]

Z_TO_COL = {element: i for i, element in enumerate(ELEMENT_LIST)}


# ---------------------------------------------------------------------------
# Extended-XYZ parser
# ---------------------------------------------------------------------------

def parse_extxyz(filepath):
    """Parse a single-frame extended XYZ file.

    Expected layout
    ---------------
    <N>                                 <- number of atoms (int)
    Properties=species:S:1:pos:R:3 ...  <- comment / key=value line (ignored)
    <symbol>  <x>  <y>  <z>            <- one line per atom (N lines)

    Returns
    -------
    N  : int
    Z  : np.ndarray, shape (N,), dtype int32    -- atomic numbers
    R  : np.ndarray, shape (N, 3), dtype float32 -- Cartesian coords (Angstrom)
    """
    with open(filepath, 'r') as fh:
        lines = [l.rstrip('\n') for l in fh if l.strip()]  # drop blank lines

    n_atoms = int(lines[0].strip())
    atom_lines = lines[2: 2 + n_atoms]

    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"{filepath}: header says {n_atoms} atoms but only "
            f"{len(atom_lines)} atom lines found."
        )

    Z_list, R_list = [], []
    for line in atom_lines:
        parts = line.split()
        symbol = parts[0]
        if symbol not in ELEMENT_TO_Z:
            raise ValueError(
                f"Unknown element '{symbol}' in {filepath}. "
                "Add it to ELEMENT_TO_Z."
            )
        Z_list.append(ELEMENT_TO_Z[symbol])
        R_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return (
        n_atoms,
        np.array(Z_list, dtype=np.int32),
        np.array(R_list, dtype=np.float32),
    )


def get_atom_count(Z, N):
    n_sample = len(N)
    n_distinct_atoms = len(Z_TO_COL)
    atom_count = np.zeros((n_sample, n_distinct_atoms))
    start = 0
    for i, n in enumerate(N):
        for z in Z[start:start + n]:
            atom_count[i, Z_TO_COL[z]] += 1
        start += n
    return atom_count


# ==========================================================================================================
# Code from DimeNet


import numpy as np
import scipy.sparse as sp

index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj",
              "id_reduce_ji", "id3dnb_i", "id3dnb_j", "id3dnb_k"]


class DataContainer:
    def __init__(self, data_root, cutoff, train=True, scale_target=False,
                 seed=42, val_size=0.1,
                 fold_train_idx=None, fold_val_idx=None):
        """
        Parameters
        ----------
        fold_train_idx / fold_val_idx : array-like, optional
            Pre-computed integer indices into the training dataset.
            When provided, the internal splitting logic is bypassed so that
            K-fold CV can inject its own splits while the scaler is still
            fitted exclusively on each fold's training portion.
        """
        self._random_state = np.random.RandomState(seed=seed)

        if scale_target or train:
            ids_train, N_train, Z_train, R_train = DataContainer.parse_dataset(data_root, 'train')
            energies_csv = os.path.join(data_root, 'energies/train.csv')
            energy_by_id = {}
            with open(energies_csv, newline='') as energy_file:
                reader = csv.DictReader(energy_file)
                for row in reader:
                    mol_id = int(row['id'])
                    energy_by_id[mol_id] = row["energy"]
                train_targets = np.array([energy_by_id[id] for id in ids_train], dtype=np.float32)

        if scale_target:
            whole_train_atom_count = get_atom_count(Z_train, N_train)
            whole_train_presence = whole_train_atom_count > 0

            if fold_train_idx is not None:
                # K-fold CV: caller supplies the split
                self.train_idx = np.asarray(fold_train_idx)
                self.val_idx   = np.asarray(fold_val_idx)
            else:
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=val_size, random_state=self._random_state)
                self.train_idx, self.val_idx = next(
                    msss.split(np.zeros(len(N_train)), whole_train_presence))

            train_only_atom_count = whole_train_atom_count[self.train_idx]
            train_only_targets    = train_targets[self.train_idx]
            self.scaler = LREnergyScaler(True)
            self.scaler.fit(train_only_atom_count, train_only_targets)
            # |LR residual| per sample — used by DataProvider for weighted sampling
            raw_residuals = train_only_targets - self.scaler.lr.predict(train_only_atom_count)
            self.train_sample_weights = np.abs(raw_residuals).astype(np.float32)

        else:
            if train:
                if fold_train_idx is not None:
                    self.train_idx = np.asarray(fold_train_idx)
                    self.val_idx   = np.asarray(fold_val_idx)
                else:
                    from math import ceil
                    n_val = ceil(len(N_train) * val_size)
                    all_idx = np.arange(len(N_train))
                    all_idx = self._random_state.permutation(all_idx)
                    self.train_idx = all_idx[n_val:]
                    self.val_idx   = all_idx[:n_val]

        if train:
            self.id      = ids_train
            self.Z       = Z_train
            self.N       = N_train
            self.R       = R_train
            self.targets = train_targets
        else:
            self.id, self.N, self.Z, self.R = DataContainer.parse_dataset(data_root, "test")
            self.targets = np.zeros(len(self.id), dtype=np.float32)

        self.cutoff   = cutoff
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        assert self.R is not None

        if scale_target:
            if train:
                self.targets = self.scaler.transform(whole_train_atom_count, self.targets)
        else:
            self.scaler = None

    @staticmethod
    def parse_dataset(data_root, dataset_name):
        ids_list, N_list, Z_list, R_list = [], [], [], []

        split_dir = os.path.join(os.path.join(data_root, 'atoms'), dataset_name)
        for molecule_file in sorted(os.listdir(split_dir)):
            if not molecule_file.endswith('.xyz'):
                continue

            mol_id = int(os.path.splitext(molecule_file)[0][3:])
            ids_list.append(mol_id)

            n, Z, R = parse_extxyz(os.path.join(split_dir, molecule_file))
            N_list.append(n)
            Z_list.append(Z)
            R_list.append(R)

        ids = np.array(ids_list, dtype=np.int32)
        N   = np.array(N_list,   dtype=np.int32)
        Z   = np.concatenate(Z_list).astype(np.int32)
        R   = np.concatenate(R_list).astype(np.float32)
        return ids, N, Z, R

    # ==============================================
    # Code from DimeNet
    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))])

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {}
        data['targets'] = self.targets[idx]
        data['id']      = self.id[idx]
        data['N']       = self.N[idx]
        data['batch_seg'] = np.repeat(np.arange(len(idx), dtype=np.int32), data['N'])
        adj_matrices = []

        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)

        nend = 0
        for k, i in enumerate(idx):
            n = data['N'][k]
            nstart = nend
            nend   = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))
            adj_matrices[-1] -= sp.eye(n, dtype=np.bool_)

        adj_matrix = self._bmat_fast(adj_matrices)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        data['idnb_i'] = edgeid_to_target
        data['idnb_j'] = edgeid_to_source

        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i  = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j  = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k  = adj_matrix[edgeid_to_source].nonzero()[1]

        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
        data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
        data['id3dnb_k'] = id3ynb_k[id3_y_to_d]

        data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        return data
