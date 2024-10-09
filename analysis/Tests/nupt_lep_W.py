import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector

leps_good = ak.zip(
    {
        "x": [0.97,  13.1, -8.1, 54, -27.8, 33.5, -40.2, -146, 108, -10.6], 
        "y": [-44.1, -43.7, -37.2, -26.8, -31.2, 59.2, -7.4, -75.4, 14.3, 33.7 ],  
        "z": [4.79, 17.8,  59.8, 50.1, 74.5, -14.9, -38.4, -15.3, 223, -28.5],     
        "t": [44.4, 49,  70.9, 78.4, 85.5, 69.7, 56, 165, 248, 45.4 ]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

met_good = ak.Array({
    "px": [2.47, 128, 60.5, -5.47, 44.4, -29.8, -1.81, -1.06, -5.4, 20.1], 
    "py": [27.4, -37.4, -28.6, -14.4, -67.9, 33.8, -32.4, 3.34, 19.1, 11.6],     
    "phi" : [1.48, -0.285, -0.442, -1.93, -0.991, 2.29,-1.63, 1.88, 1.85, 0.525]})  

leps_bad = ak.zip(
    {
        "x": [49.2, -42.9, -19.2, 92, 12.8,-57, 1.31, 58.5, 130, 120], 
        "y": [138, 16.1, 53.1, -53.9, 85.3, 114, -40.2, -29.9, -6.89, -162],  
        "z": [-4.82, -63.4, 43.8, -151, -67.9, 236, 24, -5.43, 17.1, -168],     
        "t": [146, 78.3, 71.5, 185, 110, 268, 46.8, 65.9, 131, 262]
    },
    with_name="LorentzVector",
    behavior = vector.behavior,
)

met_bad = ak.Array({
    "px": [-29.2, 106, 32.4,-28.1,-60.4,-20.4, 49.4, 1.85, -32.3, -34.5,], 
    "py": [-33.4, 36.6,-52.7,-49.9,0.648,-16, 41.6, 42.9, 2.22, -12.2],     
    "phi" : [-2.29, 0.332, -1.02, -2.08, 3.13, -2.48, 0.7, 1.53, 3.07, -2.8]})  

# calculate neutrino pT and gives weights compareing to MET pT
def nu_pt(eta, l, nu):
    mW = 80.36
    nu_t = -mW**2/(2*(l.px*np.cos(nu.phi)+ l.py*np.sin(nu.phi) + l.pz*np.sinh(eta) - l.energy*np.cosh(eta))) #analytic solution for pT

    w = (-np.exp(-(nu.px - nu_t*np.cos(nu.phi))**2/10000) * np.exp(-(nu.py - nu_t*np.sin(nu.phi))**2/10000)) #scaling by 10000 seems to gives "reasonable" digits for the weight 
        
    return w, nu_t

#manual optimization

def optimize(leptons, met):
    eta = np.arange(-6,6,0.001) #generate eta space
    eta = ak.Array([eta] * len(met))

    #extract weights and inferred pT
    weights, pt = nu_pt(eta, leptons, met)
    
    #select minimum weights and corresponding eta, pT

    max_weights = weights[ak.argsort(weights, axis=1)]
    max_pt = pt[ak.argsort(weights, axis=1)] 
    max_eta = eta[ak.argsort(weights, axis=1)] 

    return max_weights, max_pt, max_eta

weights_good, pt_good, eta_good = optimize(leps_good, met_good)
weights_bad, pt_bad, eta_bad = optimize(leps_bad, met_bad)

#create reconstructed neutrino four vector
nu_good = ak.zip(
    {
        "x": met_good.px,
        "y": met_good.py,
        "z": np.sqrt(met_good.px**2 + met_good.py**2) * np.sinh(eta_good[:,0]),
        "t": np.sqrt(met_good.px**2 + met_good.py**2 + (np.sqrt(met_good.px**2 + met_good.py**2) * np.sinh(eta_good[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

nu_bad = ak.zip(
    {
        "x": met_bad.px,
        "y": met_bad.py,
        "z": np.sqrt(met_bad.px**2 + met_bad.py**2) * np.sinh(eta_bad[:,0]),
        "t": np.sqrt(met_bad.px**2 + met_bad.py**2 + (np.sqrt(met_bad.px**2 + met_bad.py**2) * np.sinh(eta_bad[:,0]))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

#check if we get back W mass
print((nu_bad+leps_bad).mass)
