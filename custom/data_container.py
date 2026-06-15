# ===========================================
# Student code: adapt dataloading to the project dataset
# Commentary / robustness with Claude

import os
import csv

# Periodic-table mapping from element symbol to atomic number (Z).
# Extend as needed for your dataset.
ELEMENT_TO_Z = {
    'H': 1,  'He': 2,   'C': 6,
    'N': 7,  'O': 8,    'Na': 11, 
    'S': 16, 'Cl':17
}


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
    # lines[1] is the comment / Properties line -> skip it
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


# ==========================================================================================================
# Code from DimeNet


import numpy as np
import scipy.sparse as sp

index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj",
              "id_reduce_ji", "id3dnb_i", "id3dnb_j", "id3dnb_k"]

 
class DataContainer:
    # =============================================
    # init modified to match our problem
    def __init__(self, data_root, cutoff, train=True):
        if train:
            dataset_name = "train"
        else:
            dataset_name = "test"


        ids_list, N_list, Z_list, R_list = [], [], [], []

        split_dir = os.path.join(os.path.join(data_root, 'atoms'), dataset_name)
        for molecule_file in os.listdir(split_dir):
            if not molecule_file.endswith('.xyz'):
                continue

            # splittext[0] get name without extension
            # [3:] --> remove the "id_" art the start of the name
            mol_id = int(os.path.splitext(molecule_file)[0][3:])
            ids_list.append(mol_id)

            n, Z, R = parse_extxyz(os.path.join(split_dir, molecule_file))
            N_list.append(n)
            Z_list.append(Z)
            R_list.append(R)
        self.id = np.array(ids_list, dtype=np.int32)           
        self.N = np.array(N_list,   dtype=np.int32)           
        self.Z = np.concatenate(Z_list).astype(np.int32)      
        self.R = np.concatenate(R_list).astype(np.float32) 
        print(self.Z.shape)
        
        if train:
            # load training target
            energies_csv = os.path.join(data_root, 'energies/train.csv')
            energy_by_id = {}  # mol_id (int) -> {col: float}
            with open(energies_csv, newline='') as energy_file:
                reader = csv.DictReader(energy_file)
                for row in reader:
                    mol_id = int(row['id'])
                    energy_by_id[mol_id] = row["energy"]
        
            self.targets = np.array([energy_by_id[id] for id in self.id], dtype=np.float32)
        else:
            self.targets = np.array([0 for _ in self.id], dtype=np.float32)
        
        self.cutoff = cutoff
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        assert self.R is not None
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
        #return self.targets.shape[0]
        return len(self.targets)

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {}
        data['targets'] = self.targets[idx]
        data['id'] = self.id[idx]
        data['N'] = self.N[idx]
        data['batch_seg'] = np.repeat(np.arange(len(idx), dtype=np.int32), data['N'])
        adj_matrices = []

        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)

        nend = 0
        for k, i in enumerate(idx):
            n = data['N'][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))
            adj_matrices[-1] -= sp.eye(n, dtype=np.bool_)
        
        # Entry x,y is edge x<-y (!)
        adj_matrix = self._bmat_fast(adj_matrices)
        # Entry x,y is edgeid x<-y (!)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target (i) and source (j) nodes of edges
        data['idnb_i'] = edgeid_to_target
        data['idnb_j'] = edgeid_to_source

        # Indices of triplets k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Indices of triplets that are not i->j->i
        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
        data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
        data['id3dnb_k'] = id3ynb_k[id3_y_to_d]

        # Edge indices for interactions
        # j->i => k->j
        data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # j->i => k->j => j->i
        data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        return data