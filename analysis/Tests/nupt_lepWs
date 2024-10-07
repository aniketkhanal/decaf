import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import matplotlib.pyplot as plt

leps = ak.zip(
    {
        "x": [37.2, 58.8, 33.5, 37, 43.6, 20.6], 
        "y": [22, -165, 59.2, 19.9, 36.1, 32.7],  
        "z": [-169, -128, -14.9, 13.8, 0.174, -36],     
        "t": [174, 217, 69.7, 44.2, 56.6, 52.8]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

nu = ak.Array({
    "px": [3.21, -9.11, -39.8, 18.6, -27.4, 159], 
    "py": [-21.8, -6.36, 54.3, -6.63, -29.1, 1.61],     
    "phi" : [-1.42, -2.53, 2.2, -0.342, -2.33, 0.0102]})  

#these are gen Ws
qq = ak.zip(
    {
        "x": [42.2, 8.57, -8.77, 39, 77.3, 69.7], 
        "y": [20.7, -54.2, 40.3, 20.6, 34.7, 19.1],  
        "z": [-330, -19.1, -39.6, -8.04, -6.13, -67.7],     
        "t": [336, 67.9, 64.8, 55.8, 90.5, 103]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

def nu_pt(eta_mWs, l, nu, W):
    mW = 80.36
    
    eta = eta_mWs.eta
    mWs = eta_mWs.mWs
    
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

def optimize(leptons, met, W):
    eta = np.arange(-6,6,0.1) #generate eta space
    mWs = np.arange(0,55,0.1)
    sample_space = ak.cartesian({"eta": eta, "mWs": mWs},axis=0)
    sample_space = ak.Array([sample_space] * len(met))

    #extract weights and inferred pT
    weights, pt = nu_pt(sample_space, leptons, met, W)
    
    #select minimum weights and corresponding eta, pT

    max_weights = weights[ak.argsort(weights, axis=1)]
    max_pt = pt[ak.argsort(weights, axis=1)] 
    max_eta = sample_space[ak.argsort(weights, axis=1)].eta

    return max_weights, max_pt, max_eta

weights, pt, eta = optimize(leps, nu, qq)

#create reconstructed neutrino four vector
nu_reco = ak.zip(
    {
        "x": nu.px,
        "y": nu.py,
        "z": np.sqrt(nu.px**2 + nu.py**2) * np.sinh(eta[:,0]),
        "t": np.sqrt(nu.px**2 + nu.py**2 + (np.sqrt(nu.px**2 + nu.py**2) * np.sinh(eta[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

print((leps+nu_reco+qq).mass)
