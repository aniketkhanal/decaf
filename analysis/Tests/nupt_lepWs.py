import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import random

leps = ak.zip(
    {
        "x": [37.2, 58.8, 33.5, -146, 43.6, 20.6], 
        "y": [22, -165, 59.2, -75.4, 36.1, 32.7],  
        "z": [-169, -128, -14.9, -15.3, 0.174, -36],     
        "t": [174, 217, 69.7, 165, 56.6, 52.8]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

nu = ak.Array({
    "px": [0.715, -24.4, -59.9, -114, 2.98, 158], 
    "py": [-12.2, 0.714, 56.5, -6.01, -27.3, 4.63],     
    "phi" : [-1.51, 3.11, 2.39, -3.09, -1.46, 0.0293]})  

#these are gen Ws
qq = ak.zip(
    {
        "x": [15.6, 52.2, -16.1, -168, 95.4, 139], 
        "y": [48.2, -157, 138, -63.3, 80.8, 33.2],  
        "z": [-495, -132, -22.1, 11.5, 51.3, -294],     
        "t": [504, 226, 163, 198, 156, 337]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

def nu_pt(eta_mWs, l, nu, W):
    mW = 80.36
    eta, mWs = eta_mWs
    mH = 125.35
    scale_factor = mW / W.mass
        
    # Scale the momentum components
    W_px = W.px * scale_factor
    W_py = W.py * scale_factor
    W_pz = W.pz * scale_factor
    W_energy = W.energy * scale_factor

    nu_t = -((mWs**2 + mW**2 - mH**2 - 2*(l.px*W_px + l.py*W_py + l.pz*W_pz))/2 - l.energy*W_energy)/(np.cosh(eta)*W_energy + np.cos(nu.phi)*W_px + np.sin(nu.phi)*W_py + np.sinh(eta)* W_pz)
    w = (-np.exp(-(nu.px - nu_t*np.cos(nu.phi))**2/10000) * np.exp(-(nu.py - nu_t*np.sin(nu.phi))**2/10000))
    return w, nu_t

def run_minimization(l, nu, W):
    nu_t_reco = []
    initial_guess = [0, random.randint(5,60)]
    bounds = [(-6, 6), (5, 60)]
    final_nu_t = None
    
    def objective_function(eta_mWs):
        nonlocal final_nu_t
        w, nu_t = nu_pt(eta_mWs, l, nu, W)
        final_nu_t = nu_t
        return w

    result = dual_annealing(objective_function, bounds=bounds)
    return result.x, final_nu_t
                

args = [ (l, met, W) for l, met, W in zip(leps, nu, qq)]
results = [run_minimization(l, met, W) for l, met, W in args]

opt = ak.Array([res[0] for res in results])
pt = ak.Array([res[1] for res in results])
eta = opt[:,0]
mWs = opt[:,1]

#create reconstructed neutrino four vector
nu_reco = ak.zip(
    {
        "x": nu.px,
        "y": nu.py,
        "z": np.sqrt(nu.px**2 + nu.py**2) * np.sinh(eta),
        "t": np.sqrt(nu.px**2 + nu.py**2 + (np.sqrt(nu.px**2 + nu.py**2) * np.sinh(eta))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

print(eta, mWs, pt)
print((leps+nu_reco+qq).mass)
