import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector

leps_good = ak.zip(
    {
        "x": [37.2, 58.8, 120, 108, 30.4], 
        "y": [22, -165, -162, 14.3, 8.06],  
        "z": [-169, -128, -168, 223, 21.3],     
        "t": [174, 217, 262, 248, 38]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

nu_good = ak.Array({
    "px": [-1.39, -20.8, 42, -49.8, 19.6], 
    "py": [-10.6, 4.66, -7.85, 20.1, 5.81],     
    "phi" : [-1.7, 2.92, -0.185, 2.76, 0.289]})  

#these are gen Ws
qq_good = ak.zip(
    {
        "x": [7.84, 19.2, -62.9, -37.7, -38], 
        "y": [48.7, -30.7, -40, 64.6, -13.8],  
        "z": [-612, 5.94, -190, 208, 46.9],     
        "t": [624, 71.3, 216, 234, 124]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

leps_bad = ak.zip(
    {
        "x": [33.5, -146, 20.6, 43.6, 37], 
        "y": [59.2, -75.4, 32.7, 36.1, 19.9],  
        "z": [-14.9, -15.3, -36, 0.0174, 13.8],     
        "t": [69.7, 165, 52.8, 56.6, 44.2]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

nu_bad = ak.Array({
    "px": [-49.8, -57.1, 141, 11.6, -67.6], 
    "py": [47, 4.5, 2.69, -17.2, 16.9],     
    "phi" : [2.38, 3.06, 0.0191, -0.98, 2.9]})  

#these are gen Ws
qq_bad = ak.zip(
    {
        "x": [-3.13, -10.8 , 62.3, 94.6, -23], 
        "y": [-25.1, -47.1 , -3.46, 77.1, 65],  
        "z": [-111, -38.1 , -88.6, 49.5, 77.1],     
        "t": [161, 105, 143, 155, 132]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

leps_gen = ak.zip(
    {
        "x": [2.86, -42, -0.156, -8.18, -16.6], 
        "y": [-45.3, 15.8, -45.6, -0.777, 1.84 ],  
        "z": [26, -62.1, 7.65, 6.05, 103],     
        "t": [52.3, 76.6, 46.3, 10.2, 104]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

nu_gen = ak.Array({
    "px": [10.9, -1.55, 10.6, -27.2, 2.62 ], 
    "py": [4.58, -10.1, -23.4, 20.3, -6.14 ],     
    "phi" : [0.399, -1.72, -1.14, 2.5, -1.17],
    "pz": [4.62, -7.08, 11.1, -59.4, 37.3]} )  

#these are gen Ws
qq_gen = ak.zip(
    {
        "x": [6.29, -103, -63.7, -102, -59.2], 
        "y": [-55, -21.4, -162, 1.7, -82.1],  
        "z": [12.8, -78.1, -61, -145, 214],     
        "t": [99.6, 154, 197, 194, 250]
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

weights_good, pt_good, eta_good = optimize(leps_good, nu_good, qq_good)
weights_bad, pt_bad, eta_bad = optimize(leps_bad, nu_bad, qq_bad)
weights_gen, pt_gen, eta_gen = optimize(leps_gen, nu_gen, qq_gen)

#create reconstructed neutrino four vector
nu_reco_good = ak.zip(
    {
        "x": nu_good.px,
        "y": nu_good.py,
        "z": np.sqrt(nu_good.px**2 + nu_good.py**2) * np.sinh(eta_good[:,0]),
        "t": np.sqrt(nu_good.px**2 + nu_good.py**2 + (np.sqrt(nu_good.px**2 + nu_good.py**2) * np.sinh(eta_good[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

nu_reco_bad = ak.zip(
    {
        "x": nu_bad.px,
        "y": nu_bad.py,
        "z": np.sqrt(nu_bad.px**2 + nu_bad.py**2) * np.sinh(eta_bad[:,0]),
        "t": np.sqrt(nu_bad.px**2 + nu_bad.py**2 + (np.sqrt(nu_bad.px**2 + nu_bad.py**2) * np.sinh(eta_bad[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

nu_reco_gen = ak.zip(
    {
        "x": nu_gen.px,
        "y": nu_gen.py,
        "z": np.sqrt(nu_gen.px**2 + nu_gen.py**2) * np.sinh(eta_gen[:,0]),
        "t": np.sqrt(nu_gen.px**2 + nu_gen.py**2 + (np.sqrt(nu_gen.px**2 + nu_gen.py**2) * np.sinh(eta_gen[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

print((leps_good+nu_reco_good+qq_good).mass)
print((leps_bad+nu_reco_bad+qq_bad).mass)
print((leps_gen+nu_reco_gen+qq_gen).mass)
