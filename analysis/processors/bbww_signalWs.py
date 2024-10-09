#!/usr/bin/env python
import logging
import numpy as np
import awkward as ak
import json
import copy
import os
from collections import defaultdict
from coffea import processor
import hist
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.util import load, save
from optparse import OptionParser
from coffea.nanoevents.methods import vector
import gzip
from scipy.optimize import dual_annealing
import random

def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

path = "decaf/analysis/data/" if "srv" in os.getcwd() else "data/"   ### to make it run with coffea4bees

class AnalysisProcessor(processor.ProcessorABC):

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
    
    met_filters = {
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
            
    def __init__(self, year, xsec):

        self._year = year
        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])
        self._xsec = xsec
        self._systematics = False
        self._skipJER = True

        self._samples = {
            'msr':('QCD', 'TT', 'SingleMuon', 'TTToSemiLeptonic', 'GluGluToHHTo2B2VLNu2J'),
            'esr':('QCD', 'TT', 'SingleElectron', 'EGamma','TTToSemiLeptonic', 'GluGluToHHTo2B2VLNu2J'),
        }
        
        self._singleelectron_triggers = { 
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
        self._singlemuon_triggers = {
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


        self._corrections = load(f'{path}/corrections.coffea')
        self._ids         = load(f'{path}/ids.coffea')
        self._common      = load(f'{path}/common.coffea')

    

        ptbins=[15.0,
                20.0,
                25.0,
                30.0, 
                60.0, 
                90.0, 
                120.0, 
                150.0, 
                180.0, 
                210.0, 
                250.0, 
                280.0, 
                310.0, 
                340.0, 
                370.0, 
                400.0, 
                430.0, 
                470.0, 
                510.0, 
                550.0, 
                590.0, 
                640.0, 
                690.0, 
                740.0, 
                790.0, 
                840.0, 
                900.0, 
                960.0, 
                1020.0, 
                1090.0, 
                1160.0, 
                1250.0]

        self.make_output = lambda: {
            'sumw': 0.,
            'met': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.Regular(50,-6,6, name='met', label=r'$\eta$'),
                storage=hist.storage.Weight(),
            ),
            'Ws_good': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
		hist.axis.Regular(100,0,400, name='Ws_good', label='mass [GeV]'),
                storage=hist.storage.Weight(),
                ),
            'Ws_bad': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
	        hist.axis.Regular(100,0,400, name='Ws_bad', label='mass [GeV]'),
		storage=hist.storage.Weight(),
	        ),

        }
            
    def process(self, events):
        isData = not hasattr(events, "genWeight")
        if isData:
            # Nominal JEC are already applied in data
            return self.process_shift(events, None)
        
        jet_factory              = self._corrections['jet_factory']
        met_factory              = self._corrections['met_factory']

        import cachetools
        jec_cache = cachetools.Cache(np.inf)
    
        nojer = "NOJER" if self._skipJER else ""
        if events.metadata['year']: 
            self._year = events.metadata['year'].replace('UL','20').replace("_", "")
            self._lumi = events.metadata['lumi']
        thekey = f"{self._year}mc{nojer}"

        def add_jec_variables(jets, event_rho):
            jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
            jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
            return jets
        
        jets = jet_factory[thekey].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = events.DeepMETResolutionTune

        shifts = [({"Jet": jets,"MET": met}, None)]
        if self._systematics:
            shifts.extend([
                ({"Jet": jets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
                ({"Jet": jets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
                ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
            ])
            if not self._skipJER:
                shifts.extend([
                    ({"Jet": jets.JER.up, "MET": met.JER.up}, "JERUp"),
                    ({"Jet": jets.JER.down, "MET": met.JER.down}, "JERDown"),
                ])
        return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)

    def process_shift(self, events, shift_name):

        dataset = events.metadata['dataset']

        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample not in dataset: continue
                selected_regions.append(region)

        isData = not hasattr(events, "genWeight")
        selection = PackedSelection(dtype="uint64")
        weights = Weights(len(events), storeIndividual=True)
        output = self.make_output()
        if shift_name is None and not isData:
            output['sumw'] = ak.sum(events.genWeight)

        ###
        #Getting corrections, ids from .coffea files
        ###

        get_ele_loose_id_sf      = self._corrections['get_ele_loose_id_sf']
        get_ele_tight_id_sf      = self._corrections['get_ele_tight_id_sf']
        get_ele_trig_weight      = self._corrections['get_ele_trig_weight']
        get_ele_reco_sf_below20  = self._corrections['get_ele_reco_sf_below20']
        get_ele_reco_sf_above20  = self._corrections['get_ele_reco_sf_above20']
        get_mu_loose_id_sf       = self._corrections['get_mu_loose_id_sf']
        get_mu_tight_id_sf       = self._corrections['get_mu_tight_id_sf']
        get_mu_loose_iso_sf      = self._corrections['get_mu_loose_iso_sf']
        get_mu_tight_iso_sf      = self._corrections['get_mu_tight_iso_sf']
        get_mu_trig_weight       = self._corrections['get_mu_trig_weight']
        get_met_xy_correction    = self._corrections['get_met_xy_correction']
        get_pu_weight            = self._corrections['get_pu_weight']    
        get_nlo_ewk_weight       = self._corrections['get_nlo_ewk_weight']    
        get_nnlo_nlo_weight      = self._corrections['get_nnlo_nlo_weight']
        get_btag_weight          = self._corrections['get_btag_weight']
        get_ttbar_weight         = self._corrections['get_ttbar_weight']
        
        isLooseElectron = self._ids['isLooseElectron'] 
        isTightElectron = self._ids['isTightElectron'] 
        isLooseMuon     = self._ids['isLooseMuon']     
        isTightMuon     = self._ids['isTightMuon']     
        isLooseTau      = self._ids['isLooseTau']      
        isLoosePhoton   = self._ids['isLoosePhoton']   
        isGoodAK4       = self._ids['isGoodAK4']       
        isSoftAK4       = self._ids['isSoftAK4']
        isHEMJet        = self._ids['isHEMJet']  
              
        
        deepflavWPs = self._common['btagWPs']['deepflav'][self._year]
        deepcsvWPs = self._common['btagWPs']['deepcsv'][self._year]

        ###
        #Initialize global quantities (MET ecc.)
        ###

        npv = events.PV.npvsGood 
        run = events.run
        #calomet = events.CaloMET
        met = events.MET
        met['pt'] , met['phi'] = get_met_xy_correction(self._year, npv, run, met.pt, met.phi, isData)

        ###
        #Initialize physics objects
        ###

        mu = events.Muon
        
        mu['isloose'] = isLooseMuon(mu)
        mu['id_sf'] = ak.where(
            mu.isloose, 
            get_mu_loose_id_sf(self._year, abs(mu.eta), mu.pt), 
            ak.ones_like(mu.pt)
        )
        mu['iso_sf'] = ak.where(
            mu.isloose, 
            get_mu_loose_iso_sf(self._year, abs(mu.eta), mu.pt), 
            ak.ones_like(mu.pt)
        )
        mu['istight'] = isTightMuon(mu)
        mu['id_sf'] = ak.where(
            mu.istight, 
            get_mu_tight_id_sf(self._year, abs(mu.eta), mu.pt), 
            mu.id_sf
        )
        mu['iso_sf'] = ak.where(
            mu.istight, 
            get_mu_tight_iso_sf(self._year, abs(mu.eta), mu.pt), 
            mu.iso_sf
        )
        mu['T'] = ak.zip(
            {
                "r": mu.pt,
                "phi": mu.phi,
            },
            with_name="PolarTwoVector",
            behavior=vector.behavior,
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
            get_ele_reco_sf_below20(self._year, e.eta+e.deltaEtaSC, e.pt), 
            get_ele_reco_sf_above20(self._year, e.eta+e.deltaEtaSC, e.pt)
        )
        e['isloose'] = isLooseElectron(e)
        e['id_sf'] = ak.where(
            e.isloose,
            get_ele_loose_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt),
            ak.ones_like(e.pt)
        )
        e['istight'] = isTightElectron(e)
        e['id_sf'] = ak.where(
            e.istight,
            get_ele_tight_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt),
            e.id_sf
        )
        e['T'] = ak.zip(
            {
                "r": e.pt,
                "phi": e.phi,
            },
            with_name="PolarTwoVector",
            behavior=vector.behavior,
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

        
        j = events.Jet
        j['isclean'] = (
            ak.all(j.metric_table(mu_loose) > 0.4, axis=2)
            & ak.all(j.metric_table(e_loose) > 0.4, axis=2)
            & ak.all(j.metric_table(tau_loose) > 0.4, axis=2)
            & ak.all(j.metric_table(pho_loose) > 0.4, axis=2)
        )
        j['isgood'] = isGoodAK4(j, self._year)
        j['issoft'] = isSoftAK4(j, self._year)
        j['isHEM'] = isHEMJet(j)
        j['isdflvL'] = (j.btagDeepFlavB>deepflavWPs['loose']) # deep flavour 
        j['isdflvM'] = (j.btagDeepFlavB>deepflavWPs['medium'])
        j['isdflvT'] = (j.btagDeepFlavB>deepflavWPs['tight'])
        j['T'] = ak.zip( 
            {
                "r": j.pt,
                "phi": j.phi,
            },
            with_name="PolarTwoVector",
            behavior=vector.behavior,
        )
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

        ###
        # Calculate derivatives
        ###

        j_candidates = j_soft[ak.argsort(j_soft.particleNetAK4_B, axis=1, ascending=False)] #particleNetAK4_B btagPNetB 
        jb_candidates = j_candidates[:,:2] # two b-jets
        j_candidates = j_candidates[:,2:]
        
        j_candidates = j_candidates[ak.argsort(j_candidates.particleNetAK4_QvsG, axis=1, ascending=False)]#particleNetAK4_QvsG btagPNetQvG
        j_candidates = j_candidates[:,:3] # three leading qvg jets
        j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)]
        
        jj_i = ak.argcombinations(j_candidates,2,fields=["j1","j2"])
        jj_i = jj_i[(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).eta<2.0]
        jj_i = jj_i[(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).mass<120.0]
        
        qq = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2],3, axis=1)

        leading_ls = ak.concatenate([ak.singletons(leading_mu),ak.singletons(leading_e)],axis=1)
        leading_ls = leading_ls[ak.argmax(leading_ls.pt,axis=1,keepdims=True)]
        leading_ls = ak.firsts(leading_ls)
        
        leading_ls = ak.zip(
            {
                "pt": leading_ls.pt,
                "eta": leading_ls.eta,
                "phi": leading_ls.phi,
                "mass": leading_ls.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
        
        def nu_pt(eta_mWs, l, nu, W):
            mW = 80.36
            if (W is None) | (l.phi is None) :
                eta, mWs = eta_mWs
                w = 5000
                nu_t = None
            else:
                eta, mWs = eta_mWs
                mH = 125.35
                scale_factor = mW / W.mass
        
                # Scale the momentum components
                W_px = W.px * scale_factor
                W_py = W.py * scale_factor
                W_pz = W.pz * scale_factor
                W_energy = W.energy * scale_factor

                nu_t = -((mWs**2 + mW**2 - mH**2 - 2*(l.px*W_px + l.py*W_py + l.pz*W_pz))/2 - l.energy*W_energy)/(np.cosh(eta)*W_energy + np.cos(nu.phi)*W_px + np.sin(nu.phi)*W_py + np.sinh(eta)* W_pz)
                w = (-np.exp(-(nu.pt*np.cos(nu.phi) - nu_t*np.cos(nu.phi))**2/10000) * np.exp(-(nu.pt * np.sin(nu.phi) - nu_t*np.sin(nu.phi))**2/10000))
            if w is None:
                w = 5000
            return w, nu_t
                
        def run_minimization(l, nu, W):
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

        args1 = [ (l, nu, W) for l, nu, W in zip(leading_ls, met, qq[:,0])]
        args2 = [ (l, nu, W) for l, nu, W in zip(leading_ls, met, qq[:,1])]
        args3 = [ (l, nu, W) for l, nu, W in zip(leading_ls, met, qq[:,2])]
        results1 = [run_minimization(l, nu, W) for l, nu, W in args1]
        results2 = [run_minimization(l, nu, W) for l, nu, W in args2]
        results3 = [run_minimization(l, nu, W) for l, nu, W in args3]

        opt1 = ak.Array([res[0] for res in results1])
        opt2 = ak.Array([res[0] for res in results2])
        opt3 = ak.Array([res[0] for res in results3])
        eta = ak.concatenate([ak.singletons(opt1[:,0]),ak.singletons(opt2[:,0]), ak.singletons(opt3[:,0])], axis=1)
        mWs = ak.concatenate([ak.singletons(opt1[:,1]),ak.singletons(opt2[:,1]), ak.singletons(opt3[:,1])], axis=1)

        eta = ak.mask(eta, ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass,3,axis=1) >= 55.0)
        mWs = ak.mask(mWs, ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass,3,axis=1) >= 55.0)

        v_reco = ak.zip(
            {
                "x": met.pt * np.cos(met.phi),
                "y": met.pt * np.sin(met.phi),
                "z": met.pt*np.sinh(eta),
                "t": np.sqrt(met.pt**2+(met.pt*np.sinh(eta))**2) ,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        mh = ak.mask((v_reco+leading_ls+qq).mass ,ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass,3,axis=1) >= 55.0)
        print(mh, (v_reco+leading_ls+qq).mass)

        def nu_pz(l,nu,W):
            m_H = 125.35
            A = -(l.px**2+l.py**2+l.pz**2 + 2*(l.px*W.px+l.py*W.py+l.pz*W.pz)+W.px**2+W.py**2+W.pz**2) - 2*l.px*nu.pt * np.cos(nu.phi) -2*l.py*nu.pt * np.sin(nu.phi)-2*W.px*nu.pt * np.cos(nu.phi)- 2*W.py*nu.pt * np.sin(nu.phi) + (-m_H**2 + (l.energy + W.energy)**2)
            B = (nu.pt * np.cos(nu.phi)**2+nu.pt * np.sin(nu.phi)**2)*4*(l.energy+W.energy)**2
            C = 4*(-(l.energy+W.energy)**2 + (l.pz+W.pz)**2)
            discriminant = (4 * A * (l.pz+W.pz))**2 - 4 * (A**2 - B) * C
            # avoiding imaginary solutions
            sqrt_discriminant = ak.where(discriminant >= 0, np.sqrt(discriminant),np.nan)
            pz_1 = (4*A*(l.pz+W.pz) + sqrt_discriminant)/(2*C)
            pz_2 = (4*A*(l.pz+W.pz) - sqrt_discriminant)/(2*C)
            pz =  ak.where(abs(pz_1) > abs(pz_2), pz_1, pz_2)                  
            return pz

        met_pz = nu_pz(leading_ls, met, qq)
        mh_good = ak.mask(mh, ~np.isnan(met_pz))
        mh_bad = ak.mask(mh, np.isnan(met_pz))
        
        ###
        #Calculating weights
        ###

        if not isData:
            
            gen = events.GenPart

            gen['is_nue'] = (abs(gen.pdgId)==12)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['is_numu'] = (abs(gen.pdgId)==14)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            
            
            gen['isTop'] = (abs(gen.pdgId)==6)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            nlo = np.ones(len(events), dtype='float')
            if('TT' in dataset): 
                nlo = np.sqrt(get_ttbar_weight(genTops[:,0].pt) * get_ttbar_weight(genTops[:,1].pt))
                
            gen['isW'] = (abs(gen.pdgId)==24)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            gen['isZ'] = (abs(gen.pdgId)==23)&gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            
            genWs = gen[gen.isW] 
            genZs = gen[gen.isZ]
            genDYs = gen[gen.isZ&(gen.mass>30)]
            
            nnlo_nlo = {}
            nlo_qcd = np.ones(len(events), dtype='float')
            nlo_ewk = np.ones(len(events), dtype='float')
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

            pu = get_pu_weight(self._year, events.Pileup.nTrueInt)

            ###
            # Trigger efficiency weight
            ###
           
            trig = {
                'esr':   get_ele_trig_weight(self._year, leading_e.eta+leading_e.deltaEtaSC, leading_e.pt),
                #'msr':   get_mu_trig_weight(self._year, leading_mu.eta, leading_mu.pt)
                'msr': np.ones(len(events), dtype='float'),
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
                'msr': np.ones(len(events), dtype='float'),
            }

            ###
            # Isolation weights for muons
            ###

            isolation = {
                'esr': np.ones(len(events), dtype='float'),
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
            btagSFlight_uncorrelatedDown  = get_btag_weight('deepflav',self._year,'medium').btag_weight(
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
                weights.add('qcd1',np.ones(len(events), dtype='float'), nnlo_nlo['qcd1up']/nnlo_nlo['cen'], nnlo_nlo['qcd1do']/nnlo_nlo['cen'])
                weights.add('qcd2',np.ones(len(events), dtype='float'), nnlo_nlo['qcd2up']/nnlo_nlo['cen'], nnlo_nlo['qcd2do']/nnlo_nlo['cen'])
                weights.add('qcd3',np.ones(len(events), dtype='float'), nnlo_nlo['qcd3up']/nnlo_nlo['cen'], nnlo_nlo['qcd3do']/nnlo_nlo['cen'])
                weights.add('ew1',np.ones(len(events), dtype='float'), nnlo_nlo['ew1up']/nnlo_nlo['cen'], nnlo_nlo['ew1do']/nnlo_nlo['cen'])
                weights.add('ew2G',np.ones(len(events), dtype='float'), nnlo_nlo['ew2Gup']/nnlo_nlo['cen'], nnlo_nlo['ew2Gdo']/nnlo_nlo['cen'])
                weights.add('ew3G',np.ones(len(events), dtype='float'), nnlo_nlo['ew3Gup']/nnlo_nlo['cen'], nnlo_nlo['ew3Gdo']/nnlo_nlo['cen'])
                weights.add('ew2W',np.ones(len(events), dtype='float'), nnlo_nlo['ew2Wup']/nnlo_nlo['cen'], nnlo_nlo['ew2Wdo']/nnlo_nlo['cen'])
                weights.add('ew3W',np.ones(len(events), dtype='float'), nnlo_nlo['ew3Wup']/nnlo_nlo['cen'], nnlo_nlo['ew3Wdo']/nnlo_nlo['cen'])
                weights.add('ew2Z',np.ones(len(events), dtype='float'), nnlo_nlo['ew2Zup']/nnlo_nlo['cen'], nnlo_nlo['ew2Zdo']/nnlo_nlo['cen'])
                weights.add('ew3Z',np.ones(len(events), dtype='float'), nnlo_nlo['ew3Zup']/nnlo_nlo['cen'], nnlo_nlo['ew3Zdo']/nnlo_nlo['cen'])
                weights.add('mix',np.ones(len(events), dtype='float'), nnlo_nlo['mixup']/nnlo_nlo['cen'], nnlo_nlo['mixdo']/nnlo_nlo['cen'])
                #weights.add('muF',np.ones(len(events), dtype='float'), nnlo_nlo['muFup']/nnlo_nlo['cen'], nnlo_nlo['muFdo']/nnlo_nlo['cen'])
                #weights.add('muR',np.ones(len(events), dtype='float'), nnlo_nlo['muRup']/nnlo_nlo['cen'], nnlo_nlo['muRdo']/nnlo_nlo['cen'])
            weights.add('pileup',pu)
            weights.add('trig', trig[region])
            weights.add('ids', ids[region])
            weights.add('reco', reco[region])
            weights.add('btagSF',btagSF)
            weights.add('btagSFbc_correlated',np.ones(len(events), dtype='float'), btagSFbc_correlatedUp/btagSF, btagSFbc_correlatedDown/btagSF)
            weights.add('btagSFbc_uncorrelated',np.ones(len(events), dtype='float'), btagSFbc_uncorrelatedUp/btagSF, btagSFbc_uncorrelatedDown/btagSF)
            weights.add('btagSFlight_correlated',np.ones(len(events), dtype='float'), btagSFlight_correlatedUp/btagSF, btagSFlight_correlatedDown/btagSF)
            weights.add('btagSFlight_uncorrelated',np.ones(len(events), dtype='float'), btagSFlight_uncorrelatedUp/btagSF, btagSFlight_uncorrelatedDown/btagSF)
            

        
        ###
        # Selections
        ###

        lumimask = np.ones(len(events), dtype='bool')
        if isData:
            lumimask = AnalysisProcessor.lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumimask', lumimask)

        met_filters =  np.ones(len(events), dtype='bool')
        #if isData: met_filters = met_filters & events.Flag['eeBadScFilter'] #this filter is recommended for data only
        for flag in AnalysisProcessor.met_filters[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters',met_filters)


        triggers = np.zeros(len(events), dtype='bool')
        for path in self._singleelectron_triggers[self._year]:
            if not hasattr(events.HLT, path): continue
            triggers = triggers | events.HLT[path]
        selection.add('singleelectron_triggers', triggers)
        
        triggers = np.zeros(len(events), dtype='bool')
        for path in self._singlemuon_triggers[self._year]:
            if not hasattr(events.HLT, path): continue
            triggers = triggers | events.HLT[path]
        selection.add('singlemuon_triggers', triggers)
        

        noHEMj = np.ones(len(events), dtype='bool')
        if self._year=='2018': noHEMj = (j_nHEM==0)
        noHEMmet = np.ones(len(events), dtype='bool')
        if self._year=='2018': noHEMmet = (met.pt>470)|(met.phi>-0.62)|(met.phi<-1.62)    
        

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
        

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.flatten(ak.to_numpy(ak.fill_none(val[cut], np.nan)),axis=None)
                return ar
                
        def fill(region, systematic):
            cut = selection.all(*regions[region])
            sname = 'nominal' if systematic is None else systematic
            if systematic in weights.variations:
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut]
            if systematic is None:
                variables = {
                    'Ws_good':      	   ak.pad_none(mh_good,3,axis=1),
                    'Ws_bad':              ak.pad_none(mh_bad,3,axis=1),
                    'met':         ak.pad_none(ak.firsts(gen[gen.is_nue|gen.is_numu].eta)-eta,3,axis=1)
                }
                
                for variable in output:
                    if variable not in variables:
                        continue
                    normalized_variable = {variable: normalize(variables[variable],cut)}
                    output[variable].fill(
                        region=region,
                        **normalized_variable,
                        #weight=weight,
                    )

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        for region in regions:
            if region not in selected_regions: continue

            ###
            # Adding recoil and minDPhi requirements
            ###

            for systematic in systematics:
                if isData and systematic is not None:
                    continue
                fill(region, systematic)


        scale = 1
        if isinstance(self._xsec, dict):
            if self._xsec[dataset]!= -1: 
                scale = self._lumi*self._xsec[dataset]
        else:
            if self._xsec!= -1: 
                scale = events.metadata['lumi']*events.metadata['xs']

        for key in output:
            if key=='sumw': 
                continue
            output[key] *= scale
                
        return output

    def postprocess(self, accumulator):

        return accumulator

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    (options, args) = parser.parse_args()


    with gzip.open('metadata/'+options.metadata+'.json.gz') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k,v in samplefiles.items()}

    processor_instance=AnalysisProcessor(year=options.year,
                                         xsec=xsec)

    save(processor_instance, 'data/bbww'+options.name+'.processor')
