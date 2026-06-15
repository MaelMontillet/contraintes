import numpy as np

class AtomWiseEnergyScaler:
    def __init__(self, possible_element):
        self.ref_energy = {}  # per-element reference
        self.residual_mean = 0.0
        self.residual_std  = 1.0
    
    def fit(self, Z, N, possible_elements):
        
        # Compute per-element average energy
        
        
        for Z_list, E in zip(atomic_numbers_list, y_train):
            for Z in Z_list:
                if Z not in element_energies:
                    element_energies[Z] = []
                element_energies[Z].append(E / len(Z_list))  # per-atom
        
        self.ref_energy = {
            Z: np.mean(energies) 
            for Z, energies in element_energies.items()
        }
        
        # Compute residuals
        residuals = []
        for Z_list, E in zip(atomic_numbers_list, y_train):
            ref_E = sum(self.ref_energy.get(Z, 0) for Z in Z_list)
            residual = E - ref_E
            residuals.append(residual)
        
        self.residual_mean = np.mean(residuals)
        self.residual_std  = np.std(residuals)
    
    def transform(self, atomic_numbers_list, y):
        residuals = []
        for Z_list, E in zip(atomic_numbers_list, y):
            ref_E = sum(self.ref_energy.get(Z, 0) for Z in Z_list)
            residual = E - ref_E
            residuals.append((residual - self.residual_mean) / self.residual_std)
        return np.array(residuals)
    
    def inverse_transform(self, atomic_numbers_list, y_norm):
        predictions = []
        for Z_list, y_n in zip(atomic_numbers_list, y_norm):
            ref_E = sum(self.ref_energy.get(Z, 0) for Z in Z_list)
            E = ref_E + y_n * self.residual_std + self.residual_mean
            predictions.append(E)
        return np.array(predictions)