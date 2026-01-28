import numpy as np

# constants used to get growth rate to order unity
hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
GeV = 1e9 * eV
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3
ndens_to_invsec = np.sqrt(2.0)*GF/hbar
print("ndens_to_invsec")
print(ndens_to_invsec)