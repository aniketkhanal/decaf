import numpy as np
import awkward as ak
import vector
from coffea.nanoevents.methods import candidate
from coffea.util import load, save
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
import dask
import dask_awkward as dak
import hist
import hist.dask as hda
from dask.diagnostics import ProgressBar
from dask.diagnostics import ResourceProfiler

vector.register_awkward()

path = "decaf/analysis/data/"

def make_output():
    return {
        'sumw': 0.,
        "met": (
            hda.Hist.new
            .StrCat([], name="dataset", growth = True)     
            .Var([-10, 0, 55, 1000], name="sr_hadw")   
            .Var([-10, 0, 55, 1000], name="sr_hadws")
            .Var([-10, 0, 55, 1000], name="br_tt")
            .Reg(50, 0, 300, name="met", label = 'pT [GeV]')               
            .Weight()                               
        ),
        "chi_hadW": (
            hda.Hist.new
            .StrCat([], name="dataset", growth = True)
            .Var([-10, 0, 55, 1000], name="sr_hadw")
            .Var([-10, 0, 55, 1000], name="sr_hadws")
            .Var([-10, 0, 55, 1000], name="br_tt")
            .Reg(50, 0, 5, name="chi_hadW", label=r'$\chi^2$')
            .Weight()
        ),
        "chi_hadWs": (
            hda.Hist.new
            .StrCat([], name="dataset", growth = True)
            .Var([-10, 0, 55, 1000], name="sr_hadw")
            .Var([-10, 0, 55, 1000], name="sr_hadws")
            .Var([-10, 0, 55, 1000], name="br_tt")
            .Reg(50, 0, 5, name="chi_hadWs", label=r'$\chi^2$')
            .Weight()
        ),
        "chi_tt": (
            hda.Hist.new
            .StrCat([], name="dataset", growth = True)
            .Var([-10, 0, 55, 1000], name="sr_hadw")
            .Var([-10, 0, 55, 1000], name="sr_hadws")
            .Var([-10, 0, 55, 1000], name="br_tt")
            .Reg(50, 0, 5, name="chi_tt", label=r'$\chi^2$')
            .Weight()
        ),
    }

def selection(events):

    #metadata section
    dataset = events.metadata['dataset']
    shift_name = None #placeholder until we add systematics and corrections. Eventually want shift_name as input to selection

    samples = {
            'msr':('QCD', 'TT', 'SingleMuon', 'TTToSemiLeptonic', 'GluGluToHHTo2B2VLNu2J'),
            'esr':('QCD', 'TT', 'SingleElectron', 'EGamma','TTToSemiLeptonic', 'GluGluToHHTo2B2VLNu2J'),
        }

    singleelectron_triggers = { 
        #Triggers from: https://github.com/rishabhCMS/decaf/blob/new_coffea/analysis/processors/leptonic_new_coffea.py
        #Trigger efficiency SFs from there as well
        '2016postVFP': [
            'Ele27_WPTight_Gsf',
            'Ele105_CaloIdVT_GsfTrkIdT'
        ],
        '2016preVFP': [
            'Ele27_WPTight_Gsf',
            'Ele105_CaloIdVT_GsfTrkIdT'
        ],
        '2017': [
            'Ele35_WPTight_Gsf',
            'Ele115_CaloIdVT_GsfTrkIdT',
            'Photon200'
        ],
        '2018': [
            'Ele32_WPTight_Gsf',
            'Ele115_CaloIdVT_GsfTrkIdT',
            'Photon200'
        ]
    }
    singlemuon_triggers = {
        '2016preVFP': [
            'IsoMu24', 
            'IsoTkMu24'
        ],
        '2016postVFP': [
            'IsoMu24',
            'IsoTkMu24'
        ],
        '2017': [
            'IsoMu27'
        ],
        '2018': [
            'IsoMu24'
        ]
    }

    lumis = { 
        #Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVDatasetsUL2016                                                      
        '2016postVFP': 19.5,
        '2016preVFP': 16.8,
        '2017': 41.48,
        '2018': 59.83
    }

    lumiMasks = {
        '2016postVFP': LumiMask(f"{path}/jsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2016preVFP': LumiMask(f"{path}/jsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        '2017': LumiMask(f"{path}/jsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
        '2018': LumiMask(f"{path}/jsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
    }
    met_filters_names = {
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        '2016postVFP': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                'BadPFMuonDzFilter',
                'eeBadScFilter'
                ],

        '2016preVFP': [
                'goodVertices',
                'globalSuperTightHalo2016Filter',
                'HBHENoiseFilter',
                'HBHENoiseIsoFilter',
                'EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter',
                # 'BadPFMuonDzFilter',
                'eeBadScFilter'
                ],
        
        '2017': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
                ],

        '2018': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                #'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
                ]
    }

    selected_regions = []
    for region, samples_all in samples.items():
        for sample in samples_all:
            if sample not in dataset: continue
            selected_regions.append(region)

    isData = not hasattr(events, "genWeight")
    selection = PackedSelection(dtype="uint64")
    weights = Weights(None, storeIndividual=True)

    output = make_output()
    if shift_name is None and not isData:
        output['sumw'] = ak.sum(events.genWeight)

    year = '2018' #placeholder, want this as an input while running the processor eventually
    systematics = False
    skipJER = True

    corrections = load(f'{path}/corrections.coffea')
    ids         = load(f'{path}/ids.coffea')
    common      = load(f'{path}/common.coffea')

    get_ele_loose_id_sf      = corrections['get_ele_loose_id_sf']
    get_ele_tight_id_sf      = corrections['get_ele_tight_id_sf']
    get_ele_trig_weight      = corrections['get_ele_trig_weight']
    get_ele_reco_sf_below20  = corrections['get_ele_reco_sf_below20']
    get_ele_reco_sf_above20  = corrections['get_ele_reco_sf_above20']
    get_mu_loose_id_sf       = corrections['get_mu_loose_id_sf']
    get_mu_tight_id_sf       = corrections['get_mu_tight_id_sf']
    get_mu_loose_iso_sf      = corrections['get_mu_loose_iso_sf']
    get_mu_tight_iso_sf      = corrections['get_mu_tight_iso_sf']
    get_mu_trig_weight       = corrections['get_mu_trig_weight']
    get_met_xy_correction    = corrections['get_met_xy_correction']
    get_pu_weight            = corrections['get_pu_weight']    
    get_nlo_ewk_weight       = corrections['get_nlo_ewk_weight']    
    get_nnlo_nlo_weight      = corrections['get_nnlo_nlo_weight']
    get_btag_weight          = corrections['get_btag_weight']
    get_ttbar_weight         = corrections['get_ttbar_weight']
    
    isLooseElectron = ids['isLooseElectron'] 
    isTightElectron = ids['isTightElectron'] 
    isLooseMuon     = ids['isLooseMuon']     
    isTightMuon     = ids['isTightMuon']     
    isLooseTau      = ids['isLooseTau']      
    isLoosePhoton   = ids['isLoosePhoton']   
    isGoodAK4       = ids['isGoodAK4']       
    isSoftAK4       = ids['isSoftAK4']
    isHEMJet        = ids['isHEMJet']  
            
    
    deepflavWPs = common['btagWPs']['deepflav'][year]
    deepcsvWPs =  common['btagWPs']['deepcsv'][year]

    ###
    #Initialize global quantities (MET ecc.)
    ###

    npv = events.PV.npvsGood 
    run = events.run
    #calomet = events.CaloMET
    met = events.MET
    met['pt'] , met['phi'] = get_met_xy_correction(year, npv, run, met.pt, met.phi, isData)

    ###
    #Initialize physics objects
    ###

    mu = events.Muon
        
    mu['isloose'] = isLooseMuon(mu)
    mu['id_sf'] = ak.where(
        mu.isloose, 
        get_mu_loose_id_sf(year, abs(mu.eta), mu.pt), 
        ak.ones_like(mu.pt)
    )
    mu['iso_sf'] = ak.where(
        mu.isloose, 
        get_mu_loose_iso_sf(year, abs(mu.eta), mu.pt), 
        ak.ones_like(mu.pt)
    )
    mu['istight'] = isTightMuon(mu)
    mu['id_sf'] = ak.where(
        mu.istight, 
        get_mu_tight_id_sf(year, abs(mu.eta), mu.pt), 
        mu.id_sf
    )
    mu['iso_sf'] = ak.where(
        mu.istight, 
        get_mu_tight_iso_sf(year, abs(mu.eta), mu.pt), 
        mu.iso_sf
    )
    mu_loose=mu[mu.isloose]
    mu_tight=mu[mu.istight]
    mu_ntot = ak.num(mu, axis=1) 
    mu_nloose = ak.num(mu_loose, axis=1)
    mu_ntight = ak.num(mu_tight, axis=1)
    leading_mu = ak.firsts(mu_tight)

    e = events.Electron
    e['isclean'] = ak.all(e.metric_table(mu_loose) > 0.3, axis=2)
    e['reco_sf'] = ak.where(
        (e.pt<20),
        get_ele_reco_sf_below20(year, e.eta+e.deltaEtaSC, e.pt), 
        get_ele_reco_sf_above20(year, e.eta+e.deltaEtaSC, e.pt)
    )
    e['isloose'] = isLooseElectron(e)
    e['id_sf'] = ak.where(
        e.isloose,
        get_ele_loose_id_sf(year, e.eta+e.deltaEtaSC, e.pt),
        ak.ones_like(e.pt)
    )
    e['istight'] = isTightElectron(e)
    e['id_sf'] = ak.where(
        e.istight,
        get_ele_tight_id_sf(year, e.eta+e.deltaEtaSC, e.pt),
        e.id_sf
    )
    e_clean = e[e.isclean]
    e_loose = e_clean[e_clean.isloose]
    e_tight = e_clean[e_clean.istight]
    e_ntot = ak.num(e, axis=1)
    e_nloose = ak.num(e_loose, axis=1)
    e_ntight = ak.num(e_tight, axis=1)
    leading_e = ak.firsts(e_tight)

    tau = events.Tau
    tau['isclean']=(
        ak.all(tau.metric_table(mu_loose) > 0.4, axis=2) 
        & ak.all(tau.metric_table(e_loose) > 0.4, axis=2)
    )
    
    tau['isloose']=isLooseTau(tau)
    tau_clean=tau[tau.isclean]
    tau_loose=tau_clean[tau_clean.isloose]
    tau_ntot=ak.num(tau, axis=1)
    tau_nloose=ak.num(tau_loose, axis=1)

    pho = events.Photon
    pho['isclean']=(
        ak.all(pho.metric_table(mu_loose) > 0.5, axis=2)
        & ak.all(pho.metric_table(e_loose) > 0.5, axis=2)
        & ak.all(pho.metric_table(tau_loose) > 0.5, axis=2)
    )
    pho['isloose']=isLoosePhoton(pho)
    pho_clean=pho[pho.isclean]
    pho_loose=pho_clean[pho_clean.isloose]
    pho_ntot=ak.num(pho, axis=1)
    pho_nloose=ak.num(pho_loose, axis=1)
    
    met = events.MET

    j = events.Jet
    j['isclean'] = (
        ak.all(j.metric_table(mu_loose) > 0.4, axis=2)
        & ak.all(j.metric_table(e_loose) > 0.4, axis=2)
        & ak.all(j.metric_table(tau_loose) > 0.4, axis=2)
        & ak.all(j.metric_table(pho_loose) > 0.4, axis=2)
    )
    j['isgood'] = isGoodAK4(j, year)
    j['issoft'] = isSoftAK4(j, year)
    j['isHEM'] = isHEMJet(j)
    j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose']) # deep flavour 
    j['isdflvM'] = (j.btagDeepFlavB>deepflavWPs['medium'])
    j['isdflvT'] = (j.btagDeepFlavB>deepflavWPs['tight'])
    j_clean = j[j.isclean]
    j_good = j_clean[j_clean.isgood]
    j_soft = j_clean[j_clean.issoft]
    j_dflvL = j_good[j_good.isdflvL]
    j_dflvM = j_good[j_good.isdflvM]
    j_dflvT = j_good[j_good.isdflvT]
    j_HEM = j[j.isHEM]
    j_nsoft=ak.num(j_soft, axis=1)
    j_ndflvL=ak.num(j_dflvL, axis=1)
    j_ndflvM=ak.num(j_dflvM, axis=1)
    j_ndflvT=ak.num(j_dflvT, axis=1)
    j_nHEM = ak.num(j_HEM, axis=1)
    leading_j = ak.firsts(j_clean)

    j_candidates = j_soft[ak.argsort(j_soft.particleNetAK4_QvsG, axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    j_candidates = j_candidates[:, :5] #consider only the first 5
    j_candidates = j_candidates[ak.argsort(j_candidates.particleNetAK4_B, axis=1, ascending=False)]#particleNetAK4_B btagPNetB
    
    valid_jets = ak.num(j_candidates) >= 4
    j_candidates = ak.mask(j_candidates,valid_jets) # proceed only if we have at least 2 b-jets and 2 non b-jets

    jb_candidates = ak.pad_none(j_candidates[:,:2], 2,axis=1) # two b-jets
    j_candidates = j_candidates[:,2:] #3 non b-jets
    j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)] #pt sort the jets
 
    jj_i = ak.argcombinations(j_candidates,2,fields=["j1","j2"]) #take dijet combinations
    jj_i = jj_i[j_candidates[jj_i.j1].deltaeta(j_candidates[jj_i.j2])<2.0]
    jj_i = jj_i[(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).mass<120.0] #dijet cuts
    jj_tt_mask =  ak.pad_none(j_candidates[jj_i.j2].pt>20.0, 3, axis=1) # select subleading pt > 20 for TTbar
    
    qq = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2], 3, axis=1)
    mbb = (jb_candidates[:,0]+jb_candidates[:,1]).mass
        
    #ttbar and hadronic W neutrino pz reconstruction
    def nu_pz(l,v):
        m_w = 80.379
        m_l = l.mass            
        A = (l.px*v.pt * np.cos(v.phi)+l.py*v.pt * np.sin(v.phi)) + (m_w**2 - m_l**2)/2
        B = l.energy**2*((v.pt * np.cos(v.phi))**2+(v.pt * np.sin(v.phi))**2)
        C = l.energy**2 - l.pz**2
        discriminant = (2 * A * l.pz)**2 - 4 * (B - A**2) * C
        # avoiding imaginary solutions
        sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant), np.nan)
        pz_1 = (2*A*l.pz + sqrt_discriminant)/(2*C)
        pz_2 = (2*A*l.pz - sqrt_discriminant)/(2*C)
        return ak.where(abs(pz_1) < abs(pz_2), pz_1, pz_2)
        
    # hadronic W* signal reconstruction
    v_e = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz(leading_e, met),
            "t": np.sqrt(met.pt**2 + nu_pz(leading_e, met)**2),
            "charge" : met.pt * 0 #placeholder value so four vector addition is compatible
        },
        with_name="Candidate"
    )

    v_mu = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz(leading_mu, met),
            "t": np.sqrt(met.pt**2+nu_pz(leading_mu, met)**2) ,
            "charge" : met.pt * 0 
        },
        with_name="Candidate"
    )

    v_mu = ak.mask(v_mu, ~np.isnan(v_mu.pz))
    v_e = ak.mask(v_e, ~np.isnan(v_e.pz)) #avoid calculations for imaginary solutions

    # H -> lvqq with electrons and muons
    mevqq = (leading_e + v_e + qq ).mass
    mmuvqq = (leading_mu + v_mu + qq).mass

    l_mu = ~ak.is_none(leading_mu.pt)
    l_e = ~ak.is_none(leading_e.pt)
    muge = leading_mu.pt > leading_e.pt

    mlvqq_hadWs = {
        'esr'  : mevqq,
        'msr'  : mmuvqq
    }
        
    mlvqq_hadWs = ak.where(l_mu & l_e,
                        ak.where(muge, mlvqq_hadWs['msr'], mlvqq_hadWs['esr']),
                        ak.where(l_mu, mlvqq_hadWs['msr'], mlvqq_hadWs['esr'])
                        ) #select leading lepton combination

    def chi_square(data,mean,std):
        x_2 = ak.sum(data**2)
        n = ak.count(data[~ak.is_none(data)])
        chi2 = ((data - mean)/std)**2
        return chi2, mean, std

    #individual chi squares for hadronic W* signal selection
    chi1_hadWs, mean1_hadWs, std1_hadWs = chi_square(mbb,116.02, 45.04) # H -> bb            
    chi2_hadWs, mean2_hadWs, std2_hadWs = chi_square(mlvqq_hadWs, 173.59, 48.67) # H -> lvqq
    chi3_hadWs, mean3_hadWs, std3_hadWs = chi_square(qq.mass,41.77, 14.92) #hadronic W*    

    #total chi square
    chi_sq_hadWs = np.sqrt(chi1_hadWs + chi2_hadWs + chi3_hadWs)
    min_chi_sq_hadWs = ak.argmin(chi_sq_hadWs, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadWs = chi_sq_hadWs[min_chi_sq_hadWs]

    jj_gen_mass = ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass, 3, axis=1) #gen mass of dijet pair
    jj_sel_gen_mass_hadWs =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_hadWs]),-1) #get gen mass of pair selected using chi square

    ## end hadronic W* signal reconstruction

    ## hadronic W signal reconstruction
    def nu_pz_Ws(l,nu,W):
        m_H = 125.35
    
        A = m_H**2 - W.mass**2 - l.mass**2 - 2*l.energy*W.energy + 2*(l.px*W.px + l.py*W.py + l.pz*W.pz) + 2*(l.px*nu.pt * np.cos(nu.phi) + l.py*nu.pt * np.sin(nu.phi) + W.px*nu.pt * np.cos(nu.phi) + W.py*nu.pt * np.sin(nu.phi))
        B = A**2/4 - (l.energy + W.energy)**2*((nu.pt * np.cos(nu.phi))**2 + (nu.pt * np.sin(nu.phi))**2)
        C = (l.pz + W.pz)**2 - (l.energy + W.energy)**2
    
        discriminant = A**2*(l.pz+W.pz)**2 - 4*B*C
        sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant),np.nan) # avoiding imaginary solutions
    
        pz_1 = (-A*(l.pz + W.pz) + sqrt_discriminant)/(2*C)
        pz_2 = (-A*(l.pz + W.pz) - sqrt_discriminant)/(2*C)
        pz =  ak.where(abs(pz_1) < abs(pz_2), pz_1, pz_2)                  
    
        return pz

    v_e_hadW = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz_Ws(leading_e, met, qq),
            "t": np.sqrt(met.pt**2 + nu_pz_Ws(leading_e, met, qq)**2),
            "charge" : met.pt * 0
        },
        with_name="Candidate",
        )

    v_mu_hadW = ak.zip(
        {
            "x": met.pt * np.cos(met.phi),
            "y": met.pt * np.sin(met.phi),
            "z": nu_pz_Ws(leading_mu, met, qq),
            "t": np.sqrt(met.pt**2 + nu_pz_Ws(leading_mu, met, qq)**2)
        },
        with_name="Candidate",
        )
    
    v_mu_hadW = ak.mask(v_mu_hadW, ~np.isnan(v_mu_hadW.pz))
    v_e_hadW = ak.mask(v_e_hadW, ~np.isnan(v_e_hadW.pz)) #avoid calculations for imaginary solutions that are not always skipped
    
    #transverse mass
    mT = {
        'esr'  : np.sqrt(2*leading_e.pt*met.pt*(1-np.cos(met.deltaphi(leading_e)))),
        'msr'  : np.sqrt(2*leading_mu.pt*met.pt*(1-np.cos(met.deltaphi(leading_mu))))
    }


    mT_leading_lep = ak.where(l_mu & l_e,
                        ak.where(muge, mT['msr'], mT['esr']),
                        ak.where(l_mu, mT['msr'], mT['esr'])
                        )


    #individual chi squares for hadronic W signal selection
    chi1_hadW, mean1_hadW, std1_hadW = chi_square(mbb,115.33, 46.29) # H -> bb
    chi2_hadW, mean2_hadW, std2_hadW = chi_square(mT_leading_lep, 58.87, 37.35) #transverse mass             
    chi3_hadW, mean3_hadW, std3_hadW = chi_square(qq.mass,66.89, 10.98) #hadronic W

    chi_sq_hadW = np.sqrt(chi1_hadW + chi2_hadW + chi3_hadW)
    min_chi_sq_hadW= ak.argmin(chi_sq_hadW, axis=1, keepdims = True) #index of the minimum chi square non-bjet pair
    chi_sq_hadW = chi_sq_hadW[min_chi_sq_hadW]

    jj_sel_gen_mass_hadW =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_hadW]),-1) #gen mass of the di jet pair with minimum chi square
    
    ## ttbar reconstruction
    
    #leptonic top with electrons
    mevb1 = (leading_e + v_e + ak.pad_none(jb_candidates,2,axis=1)[:,0]).mass
    mevb2 = (leading_e + v_e + ak.pad_none(jb_candidates,2,axis=1)[:,1]).mass

    #leptonic top with muons
    mmvb1 = (leading_mu + v_mu + ak.pad_none(jb_candidates,2,axis=1)[:,0]).mass
    mmvb2 = (leading_mu + v_mu + ak.pad_none(jb_candidates,2,axis=1)[:,1]).mass

    mlvb1 = ak.where(l_mu & l_e,
                        ak.where(muge, mmvb1, mevb1),
                        ak.where(l_mu, mmvb1, mevb1)
                        ) #leptonic candidate 1

    mlvb2 = ak.where(l_mu & l_e,
                        ak.where(muge, mmvb2, mevb2),
                        ak.where(l_mu, mmvb2, mevb2)
                        ) #leptonic candidate 2  

    mbqq1 = ak.pad_none((ak.pad_none(jb_candidates,2,axis=1)[:,0] + ak.mask(qq, jj_tt_mask)).mass,3,axis=1) #hadronic candidate 1
    mbqq2 = ak.pad_none((ak.pad_none(jb_candidates,2,axis=1)[:,1] + ak.mask(qq, jj_tt_mask)).mass,3,axis=1) #hadronic candidate 2

    def distance(x1,y1,x2,y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)
    
    #ttbar candidates
    tt1 = ak.cartesian({"t1":mlvb1,"t2":mbqq2},axis=1)
    tt2 = ak.cartesian({"t1":mlvb2,"t2":mbqq1},axis=1)
    b_sel = abs(distance(tt1.t1,tt1.t2,172.5,172.5)) <  abs(distance(tt2.t1,tt2.t2,172.5,172.5)) #pick pair closest to ttbar mass

    #conditions to work around None values
    c1 = ~ak.is_none(distance(tt1.t1, tt1.t2,172.5,172.5))
    c2 = ~ak.is_none(distance(tt2.t1, tt2.t2,172.5,172.5))

    #final ttbar candidates
    tt = ak.pad_none(ak.where( c1 & c2, ak.where(b_sel, tt1 , tt2), ak.where(c1, tt1, tt2)),3,axis=1)

    qq_tt = ak.mask(qq, jj_tt_mask) #select leading pT jet > 20 GeV for ttbar
    chi1_tt, mean1_tt, std1_tt = chi_square(tt.t1,194.93 , 47.59 ) #leptonic top
    chi2_tt, mean2_tt, std2_tt = chi_square(tt.t2, 171.55, 44.95 ) #hadronic top
    chi3_tt, mean3_tt, std3_tt = chi_square(qq_tt.mass,73.9, 23.56) #hadronic W
    
    chi_sq_tt = np.sqrt(chi1_tt + chi2_tt + chi3_tt)
    min_chi_sq_tt = ak.argmin(chi_sq_tt, axis=1, keepdims = True) #get index of the minimum chi square 
    chi_sq_tt = chi_sq_tt[min_chi_sq_tt]

    jj_sel_gen_mass_tt =  ak.fill_none(ak.firsts(jj_gen_mass[min_chi_sq_tt]),-1) #gen mass of the di jet pair with minimum chi square

    if not isData:
            
        gen = events.GenPart

        gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        genTops = gen[gen.isTop]
        nlo = dak.ones_like(events.MET.pt, dtype='float')
        if('TT' in dataset): 
            nlo = np.sqrt(get_ttbar_weight(genTops[:,0].pt) * get_ttbar_weight(genTops[:,1].pt))
            
        gen['isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        gen['isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
        
        genWs = gen[gen.isW] 
        genZs = gen[gen.isZ]
        genDYs = gen[gen.isZ&(gen.mass>30)]
        
        nnlo_nlo = {}
        nlo_qcd = dak.ones_like(events.MET.pt, dtype='float')
        nlo_ewk = dak.ones_like(events.MET.pt, dtype='float')
        if('WJets' in dataset): 
            nlo_qcd = get_nlo_qcd_weight['w'](genWs.pt.max())
            nlo_ewk = get_nlo_ewk_weight['w'](genWs.pt.max())
            for systematic in get_nnlo_nlo_weight['w']:
                nnlo_nlo[systematic] = get_nnlo_nlo_weight['w'][systematic](genWs.pt.max())*((ak.num(genWs, axis=1)>0)&(genWs.pt.max()>=100)) + \
                                        (~((ak.num(genWs, axis=1)>0)&(genWs.pt.max()>=100))).astype(np.int)
        elif('DY' in dataset): 
            nlo_qcd = get_nlo_qcd_weight['dy'](genDYs.pt.max())
            nlo_ewk = get_nlo_ewk_weight['dy'](genDYs.pt.max())
            for systematic in get_nnlo_nlo_weight['dy']:
                nnlo_nlo[systematic] = get_nnlo_nlo_weight['dy'][systematic](genDYs.pt.max())*((ak.num(genDYs, axis=1)>0)&(genDYs.pt.max()>=100)) + \
                                        (~((ak.num(genDYs, axis=1)>0)&(genDYs.pt.max()>=100))).astype(np.int)
        elif('ZJets' in dataset): 
            nlo_qcd = get_nlo_qcd_weight['z'](genZs.pt.max())
            nlo_ewk = get_nlo_ewk_weight['z'](genZs.pt.max())
            for systematic in get_nnlo_nlo_weight['z']:
                nnlo_nlo[systematic] = get_nnlo_nlo_weight['z'][systematic](genZs.pt.max())*((ak.num(genZs, axis=1)>0)&(genZs.pt.max()>=100)) + \
                                        (~((ak.num(genZs, axis=1)>0)&(genZs.pt.max()>=100))).astype(np.int)

        ###
        # Calculate PU weight and systematic variations
        ###

        pu = get_pu_weight(year, events.Pileup.nTrueInt)

        ###
        # Trigger efficiency weight
        ###
        
        trig = {
            'esr':   get_ele_trig_weight(year, leading_e.eta+leading_e.deltaEtaSC, leading_e.pt),
            #'msr':   get_mu_trig_weight(year, leading_mu.eta, leading_mu.pt)
            'msr': dak.ones_like(events.MET.pt, dtype='float'),
        }

        ### 
        # Calculating electron and muon ID weights
        ###

        ids ={
            'esr':  leading_e.id_sf,
            'msr':  leading_mu.id_sf
        }
        
        ###
        # Reconstruction weights for electrons
        ###
                                                
        reco = {
            'esr': leading_e.reco_sf, 
            'msr': dak.ones_like(events.MET.pt, dtype='float'),
        }

        ###
        # Isolation weights for muons
        ###

        isolation = {
            'esr': dak.ones_like(events.MET.pt, dtype='float'),
            'msr': leading_mu.iso_sf
        }
        

        ###
        # AK4 b-tagging weights
        ###

        btagSF, \
        btagSFbc_correlatedUp, \
        btagSFbc_correlatedDown, \
        btagSFbc_uncorrelatedUp, \
        btagSFbc_uncorrelatedDown, \
        btagSFlight_correlatedUp, \
        btagSFlight_correlatedDown, \
        btagSFlight_uncorrelatedUp, \
        btagSFlight_uncorrelatedDown  = get_btag_weight('deepflav',year,'medium').btag_weight(
            j_good.pt,
            j_good.eta,
            j_good.hadronFlavour,
            j_good.isdflvM
        )

        if hasattr(events, "L1PreFiringWeight"): 
            weights.add('prefiring', events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)
        weights.add('genw',events.genWeight)
        weights.add('nlo_ewk',nlo_ewk)
        #weights.add('nlo',nlo) 
        if 'cen' in nnlo_nlo:
            #weights.add('nnlo_nlo',nnlo_nlo['cen'])
            weights.add('qcd1',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
            weights.add('qcd2',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
            weights.add('qcd3',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
            weights.add('ew1',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
            weights.add('ew2G',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
            weights.add('ew3G',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
            weights.add('ew2W',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
            weights.add('ew3W',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
            weights.add('ew2Z',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
            weights.add('ew3Z',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
            weights.add('mix',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
            #weights.add('muF',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
            #weights.add('muR',dak.ones_like(events.MET.pt, dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
        weights.add('pileup',pu)
        weights.add('trig', trig[region])
        weights.add('ids', ids[region])
        weights.add('reco', reco[region])
        weights.add('btagSF',btagSF)
        weights.add('btagSFbc_correlated',dak.ones_like(events.MET.pt, dtype='float'), btagSFbc_correlatedUp/btagSF, btagSFbc_correlatedDown/btagSF)
        weights.add('btagSFbc_uncorrelated',dak.ones_like(events.MET.pt, dtype='float'), btagSFbc_uncorrelatedUp/btagSF, btagSFbc_uncorrelatedDown/btagSF)
        weights.add('btagSFlight_correlated',dak.ones_like(events.MET.pt, dtype='float'), btagSFlight_correlatedUp/btagSF, btagSFlight_correlatedDown/btagSF)
        weights.add('btagSFlight_uncorrelated',dak.ones_like(events.MET.pt, dtype='float'), btagSFlight_uncorrelatedUp/btagSF, btagSFlight_uncorrelatedDown/btagSF)
        
    lumimask = dak.ones_like(events.MET.pt, dtype='bool') #using events.MET.pt to get 1d array with len(events)
    if isData:
        lumimask = lumiMasks[year](events.run, events.luminosityBlock)
    selection.add('lumimask', lumimask)

    met_filters =  dak.ones_like(events.MET.pt, dtype='bool')
    #if isData: met_filters = met_filters & events.Flag['eeBadScFilter'] #this filter is recommended for data only
    for flag in met_filters_names[year]:
        met_filters = met_filters & events.Flag[flag]
    selection.add('met_filters',met_filters)

    triggers = dak.zeros_like(events.MET.pt, dtype='bool')
    for trigger_path in singleelectron_triggers[year]:
        if not hasattr(events.HLT, trigger_path): continue
        triggers = triggers | events.HLT[trigger_path]
    selection.add('singleelectron_triggers', triggers)
    
    triggers = dak.zeros_like(events.MET.pt, dtype='bool')
    for trigger_path in singlemuon_triggers[year]:
        if not hasattr(events.HLT, trigger_path): continue
        triggers = triggers | events.HLT[trigger_path]
    selection.add('singlemuon_triggers', triggers)

    noHEMj = dak.ones_like(events.MET.pt, dtype='bool')
    if year=='2018': noHEMj = (j_nHEM==0)
    noHEMmet = dak.ones_like(events.MET.pt, dtype='bool')
    if year=='2018': noHEMmet = (met.pt>470)|(met.phi>-0.62)|(met.phi<-1.62)    
    
    selection.add('isoneE', (e_ntight==1) & (mu_nloose==0) & (pho_nloose==0) & (tau_nloose==0))
    selection.add('isoneM', (mu_ntight==1) & (e_nloose==0) & (pho_nloose==0) & (tau_nloose==0))
    selection.add('njets',  (j_nsoft>2))
    selection.add('nbjets', (j_ndflvM>0))
    selection.add('noHEMj', noHEMj)
    selection.add('noHEMmet', noHEMmet)

    regions = {
        'esr': ['isoneE', 'noHEMj', 'njets', 'nbjets', 'met_filters', 'noHEMmet'],
        'msr': ['isoneM', 'noHEMj', 'njets', 'nbjets', 'met_filters', 'noHEMmet']
        }
    
    def normalize(val):
        return ak.fill_none(val, np.nan)

    variables = {
        'met':                         met.pt,
        'chi_hadW':                    ak.firsts(chi_sq_hadW),
        'chi_hadWs':                   ak.firsts(chi_sq_hadWs),
        'chi_tt':                      ak.firsts(chi_sq_tt),
    }

    for variable in output:
        if variable not in variables:
            continue
        normalized_variable = normalized_variable = {variable: normalize(variables[variable])}
        output[variable].fill(
            dataset = dataset,
            sr_hadw =  jj_sel_gen_mass_hadW,
            sr_hadws = jj_sel_gen_mass_hadWs,
            br_tt =    jj_sel_gen_mass_tt,
            **normalized_variable,
            weight= weights.weight() 
        )

    return output

#show progress bar and resource usage
pbar = ProgressBar()
pbar.register()
rprof = ResourceProfiler()

filename = "root://cmseos.fnal.gov//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_3.root"

dakevents = NanoEventsFactory.from_root(
    {filename: "Events"},
    steps_per_file=10,
    metadata={"dataset": "GluGluToHHTo2B2VLNu2J"},
    schemaclass=NanoAODSchema,
    delayed = True,
).events()

out = selection(dakevents)
computed, = dask.compute(out, scheduler='synchronous', scheduling_mode="depth-first",)
save(computed, 'hists/hists.coffea')
