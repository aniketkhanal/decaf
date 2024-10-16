import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import random
from scipy.optimize import differential_evolution

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

#here are some gen events
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

    result = differential_evolution(objective_function, bounds=bounds)
    return result.x, final_nu_t
                
args_good = [ (l, met, W) for l, met, W in zip(leps_good, nu_good, qq_good)]
args_bad = [ (l, met, W) for l, met, W in zip(leps_bad, nu_bad, qq_bad)]
results_good = [run_minimization(l, met, W) for l, met, W in args_good]
results_bad = [run_minimization(l, met, W) for l, met, W in args_bad]

opt_good = ak.Array([res[0] for res in results_good])
opt_bad = ak.Array([res[0] for res in results_bad])
pt_good = ak.Array([res[1] for res in results_good])
pt_bad = ak.Array([res[1] for res in results_bad])
eta_good = opt_good[:,0]
mWs_good = opt_good[:,1]
eta_bad = opt_bad[:,0]
mWs_bad = opt_bad[:,1]

nu_good_reco = ak.zip(
    {
        "x": nu_good.px,
        "y": nu_good.py,
        "z": np.sqrt(nu_good.px**2 + nu_good.py**2) * np.sinh(eta_good),
        "t": np.sqrt(nu_good.px**2 + nu_good.py**2 + (np.sqrt(nu_good.px**2 + nu_good.py**2) * np.sinh(eta_good))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

nu_bad_reco = ak.zip(
    {
        "x": nu_bad.px,
        "y": nu_bad.py,
        "z": np.sqrt(nu_bad.px**2 + nu_bad.py**2) * np.sinh(eta_bad),
        "t": np.sqrt(nu_bad.px**2 + nu_bad.py**2 + (np.sqrt(nu_bad.px**2 + nu_bad.py**2) * np.sinh(eta_bad))**2 )  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

print('good eta, mWs, pt (optimized):', eta_good, mWs_good, pt_good)
print('bad eta, mWs, pt (optimized):', eta_bad, mWs_bad, pt_bad)
print('good Higgs mass:',(leps_good+nu_good_reco+qq_good).mass)
print('bad Higgs mass:', (leps_bad+nu_bad_reco+qq_bad).mass)

#analytic solution
def nu_pz(l,nu,W):
    m_H = 125.35
    
    A = m_H**2 - W.mass**2 - 2*l.energy*W.energy + 2*(l.px*W.px + l.py*W.py + l.pz*W.pz) + 2*(l.px*nu.px + l.py*nu.py + W.px*nu.px + W.py*nu.py)
    B = A**2/4 - (l.energy + W.energy)**2*(nu.px**2 + nu.py**2)
    C = (l.pz + W.pz)**2 - (l.energy + W.energy)**2
    
    discriminant = A**2*(l.pz+W.pz)**2 - 4*B*C
    sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant),np.nan) # avoiding imaginary solutions

    pz_1 = (-A*(l.pz + W.pz) + sqrt_discriminant)/(2*C)
    pz_2 = (-A*(l.pz + W.pz) - sqrt_discriminant)/(2*C)
    pz =  ak.where(abs(pz_1) < abs(pz_2), pz_1, pz_2)                  
    
    return pz

print('analytic pz good:',nu_pz(leps_good, nu_good, qq_good))
print('analytic_pz_bad:',nu_pz(leps_bad, nu_bad, qq_bad))
print('analytic_pz_gen:',nu_pz(leps_gen, nu_gen, qq_gen))

#checking if our gen analytic solutions give back Higgs mass 
nu_reco_gen = ak.zip(
    {
        "x": nu_gen.px,
        "y": nu_gen.py,
        "z": nu_pz(leps_gen, nu_gen, qq_gen),
        "t": np.sqrt(nu_gen.px**2 + nu_gen.py**2 + nu_pz(leps_gen, nu_gen, qq_gen)**2)  ,
    },
    with_name="LorentzVector",
    behavior=vector.behavior,
)

print('reco Higgs mass (with gen events):' , (leps_gen+nu_reco_gen + qq_gen).mass)
