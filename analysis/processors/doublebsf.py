#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import numpy as np
import math
import awkward
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea.arrays import Initialize
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from uproot_methods import TVector2Array, TLorentzVectorArray

class AnalysisProcessor(processor.ProcessorABC):

    lumis = { #Values from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable                                                      
        '2016': 36.31,
        '2017': 41.53,
        '2018': 59.74
    }

    met_filter_flags = {
        '2016': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter'
             ],
        '2017': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilterV2'
             ],
        '2018': ['goodVertices',
                 'globalSuperTightHalo2016Filter',
                 'HBHENoiseFilter',
                 'HBHENoiseIsoFilter',
                 'EcalDeadCellTriggerPrimitiveFilter',
                 'BadPFMuonFilter',
                 'ecalBadCalibFilterV2'
             ]
    }

    def __init__(self, year, xsec, corrections, ids, common):

        self._year = year
        self._lumi = 1000.*float(AnalysisProcessor.lumis[year])
        self._xsec = xsec

        self._gentype_map = {
            'bb':       0,
            'b':        1,
            'cc' :      2,
            'c':        3,
            'other':    4,
            'hs':       5,
        }

        self._ZHbbvsQCDwp = {
            '2016': 0.53,
            '2017': 0.61,
            '2018': 0.65
        }

        self._btagmu_triggers = {
            '2016': [
                'BTagMu_AK4Jet300_Mu5',
                'BTagMu_AK8Jet300_Mu5',
                'BTagMu_AK4DiJet170_Mu5'
                ],
            '2017': [
                'BTagMu_AK4Jet300_Mu5',
                'BTagMu_AK8Jet300_Mu5',
                'BTagMu_AK4DiJet170_Mu5'
                ],
            '2018': [
                'BTagMu_AK4Jet300_Mu5',
                'BTagMu_AK8Jet300_Mu5',
                'BTagMu_AK4DiJet170_Mu5'
                ]
        }

        self._corrections = corrections
        self._ids = ids
        self._common = common

        self._accumulator = processor.dict_accumulator({
            'sumw': hist.Hist(
                'sumw',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('sumw', 'Weight value', [0.])
            ),
            'svtemplate': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6]),
                hist.Bin('svmass','Secondary Vertices (SV) mass',[-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2]),
                hist.Bin('ZHbbvsQCD','ZHbbvsQCD', [0, self._ZHbbvsQCDwp[self._year], 1])
            ),
            'ZHbbvsQCD': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6]),
                hist.Bin('ZHbbvsQCD','ZHbbvsQCD',15,0,1),
                hist.Bin('tau21','tau21', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
            ),
            'fj1pt': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6]),
                hist.Bin('fj1pt','Leading AK15 Jet SoftDrop Pt',[160.0, 250.0, 280.0, 310.0, 340.0, 370.0, 400.0, 430.0, 470.0, 510.0, 550.0, 590.0, 640.0, 690.0, 740.0, 790.0, 840.0, 900.0, 960.0, 1020.0, 1090.0, 1160.0, 1250.0]),
                hist.Bin('ZHbbvsQCD','ZHbbvsQCD', [0, self._ZHbbvsQCDwp[self._year], 1]),
            ),
            'fjmass': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Bin('gentype', 'Gen Type', [0, 1, 2, 3, 4, 5, 6]),
                hist.Bin('fjmass','AK15 Jet Mass',52,40,300),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, events):

        dataset = events.metadata['dataset']

        isData = 'genWeight' not in events.columns
        selection = processor.PackedSelection()
        hout = self.accumulator.identity()

        ###
        #Getting ids from .coffea files
        ###

        get_msd_weight  = self._corrections['get_msd_weight']
        get_pu_weight   = self._corrections['get_pu_weight'][self._year]  
        isSoftMuon      = self._ids['isSoftMuon']
        isGoodFatJet    = self._ids['isGoodFatJet']
        isHEMJet        = self._ids['isHEMJet']  

        match = self._common['match']

        ###
        #Initialize physics objects
        ###

        mu = events.Muon
        mu['issoft'] = isSoftMuon(mu.pt,mu.eta,mu.pfRelIso04_all,mu.looseId,self._year)
        mu_soft=mu[mu.issoft.astype(np.bool)]

        j = events.Jet
        j['isHEM'] = isHEMJet(j.pt, j.eta, j.phi)
        j_HEM = j[j.isHEM.astype(np.bool)]
        j_nHEM = j_HEM.counts

        fj = events.AK15Puppi
        fj['sd'] = fj.subjets.sum()
        fj['isgood'] = isGoodFatJet(fj.sd.pt, fj.sd.eta, fj.jetId)
        fj['T'] = TVector2Array.from_polar(fj.pt, fj.phi)
        fj['msd_raw'] = (fj.subjets * (1 - fj.subjets.rawFactor)).sum().mass
        fj['msd_corr'] = fj.msd_raw * awkward.JaggedArray.fromoffsets(fj.array.offsets, np.maximum(1e-5, get_msd_weight(fj.sd.pt.flatten(),fj.sd.eta.flatten())))
        probQCD=fj.probQCDbb+fj.probQCDcc+fj.probQCDb+fj.probQCDc+fj.probQCDothers
        probZHbb=fj.probZbb+fj.probHbb
        fj['ZHbbvsQCD'] = probZHbb/(probZHbb+probQCD)
        fj['tau21'] = fj.tau2/fj.tau1
        jetmu = fj.subjets.flatten(axis=1).cross(mu_soft, nested=True)
        #mask = (mu.counts>0) & ((jetmu.i0.delta_r(jetmu.i1) < 0.4) & ((jetmu.i1.pt/jetmu.i0.pt) < 0.7) & (jetmu.i1.pt > 7)).sum() == 1
        #mask = (mu.counts>0) & ((jetmu.i0.delta_r(jetmu.i1) < 0.4).sum() == 1)
        #step1 = fj.subjets.flatten()
        #step2 = awkward.JaggedArray.fromoffsets(step1.offsets, mask.content)
        #step2 = step2.pad(1).fillna(0) ##### Fill None for empty arrays and convert None to False
        #step3 = awkward.JaggedArray.fromoffsets(fj.subjets.offsets, step2)
        #fj['withmu'] = step3.sum() == 2
        jetmu = fj.sd.cross(mu_soft, nested=True)
        fj['withmu'] = (mu_soft.counts>0) & ((jetmu.i0.delta_r(jetmu.i1) < 1.5).sum()>0)
        fj_good = fj[fj.isgood.astype(np.bool)]
        fj_withmu = fj_good[fj_good.withmu.astype(np.bool)]
        fj_ngood = fj_good.counts
        fj_nwithmu = fj_withmu.counts
        

        SV = events.SV
        SV['ismatched'] = match(SV, fj, 1.5)
        
        ###
        # Calculating weights
        ###
        if not isData:

            gen = events.GenPart

            ###
            # Fat-jet dark H->bb matching at decay level
            ###
            Hs = gen[
                (gen.pdgId == 54) &
                gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            ]
            bFromHs = gen[
                (abs(gen.pdgId) == 5) &
                gen.hasFlags(['fromHardProcess', 'isLastCopy']) &
                (gen.distinctParent.pdgId == 54)
            ]
            def bbmatch():
                jetgenb = fj.sd.cross(bFromHs, nested=True)
                bbmatch = ((jetgenb.i0.delta_r(jetgenb.i1) < 1.5).sum()==2) & (bFromHs.counts>0)
                return bbmatch
            def hsmatch():
                jetgenhs = fj.sd.cross(Hs, nested=True)
                hsmatch = ((jetgenhs.i0.delta_r(jetgenhs.i1) < 1.5).sum()==1) & (Hs.counts>0)
                return hsmatch
            fj['isHsbb']  = bbmatch()&hsmatch()
            
            #####
            ###
            # Fat-jet matching to two b's
            ###
            #####

            fj['isbb']  = (fj.nBHadrons > 1)

            #####
            ###
            # Fat-jet matching to two c's
            ###
            #####

            fj['iscc']  = (fj.nCHadrons > 1) & (fj.nBHadrons == 0)

            #####
            ###
            # Fat-jet matching to one b
            ###
            #####

            fj['isb']  = (fj.nBHadrons == 1)


            #####
            ###
            # Fat-jet matching to one c
            ###
            #####

            fj['isc']  = (fj.nCHadrons == 1) & (fj.nBHadrons == 0)

            #####
            ###
            # Light-flavor fat-jet
            ###
            #####

            fj['isl']  = (fj.nCHadrons == 0) & (fj.nBHadrons == 0)
            

            #####
            ###
            # Calculate PU weight and systematic variations
            ###
            #####

            pu = get_pu_weight(events.Pileup.nTrueInt)

        #### ak15 jet selection ####
        leading_fj = fj[fj.sd.pt.argmax()]
        leading_fj = leading_fj[leading_fj.isgood.astype(np.bool)]

        #### SV selection for matched with leading ak15 jet ####
        leading_SV = SV[SV.dxySig.argmax()]
        leading_SV = leading_SV[leading_SV.ismatched.astype(np.bool)]
        SV_mass = SV[SV.ismatched.astype(np.bool)].sum().mass

        ###
        # Selections
        ###

        #### trigger selection ####
        triggers = np.zeros(events.size, dtype=np.bool)
        for path in self._btagmu_triggers[self._year]:
            if path not in events.HLT.columns: continue
            triggers = triggers | events.HLT[path]
        selection.add('btagmu_triggers', triggers)

        #### MET filters ####
        met_filters =  np.ones(events.size, dtype=np.bool)
        if isData:
            met_filters = met_filters & events.Flag['eeBadScFilter'] #this filter is recommended for data only
        for flag in AnalysisProcessor.met_filter_flags[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters',met_filters)

        noHEMj = np.ones(events.size, dtype=np.bool)
        if self._year=='2018': noHEMj = (j_nHEM==0)

        selection.add('noHEMj', noHEMj)
        selection.add('fj_pt', (leading_fj.sd.pt.max() > 250) )
        selection.add('fj_mass', (leading_fj.msd_corr.sum() > 40) ) ## optionally also <130
        selection.add('fj_good', (fj_ngood==1))
        selection.add('fj_withmu', (fj_nwithmu==1))
        #selection.add('fj_tau21', (leading_fj.tau21.sum() < 0.3) )

        isFilled = False
        if isData:
            if not isFilled:
                hout['sumw'].fill(dataset=dataset, sumw=1, weight=1)
                isFilled=True

            cut = selection.all(*selection.names)
            ##### template for bb SF #####
            hout['svtemplate'].fill(dataset=dataset,
                                    gentype=np.zeros(events.size, dtype=np.int),
                                    #svmass=np.log(leading_SV.mass.sum()),
                                    svmass=np.log(SV_mass.sum()),
                                    ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                    weight=np.ones(events.size)*cut)
            hout['ZHbbvsQCD'].fill(dataset=dataset,
                                   gentype=np.zeros(events.size, dtype=np.int),
                                   ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                   tau21=leading_fj.tau21.sum(),
                                   weight=np.ones(events.size)*cut)
            hout['fj1pt'].fill(dataset=dataset,
                                   gentype=np.zeros(events.size, dtype=np.int),
                                   fj1pt=leading_fj.sd.pt.sum(),
                                   ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                   weight=np.ones(events.size)*cut)
            hout['fjmass'].fill(dataset=dataset,
                                   gentype=np.zeros(events.size, dtype=np.int),
                                   fjmass=leading_fj.msd_corr.sum(),
                                   weight=np.ones(events.size)*cut)

        else:
            weights = processor.Weights(len(events))
            if 'L1PreFiringWeight' in events.columns: weights.add('prefiring',events.L1PreFiringWeight.Nom)
            weights.add('genw',events.genWeight)
            weights.add('pileup',pu)

            wgentype = {
                'bb' : (~leading_fj.isHsbb&leading_fj.isbb).sum(),
                'b'  : (~leading_fj.isHsbb&leading_fj.isb).sum(),
                'cc' : (~leading_fj.isHsbb&leading_fj.iscc).sum(),
                'c'  : (~leading_fj.isHsbb&leading_fj.isc).sum(),
                'other' : (~leading_fj.isHsbb&leading_fj.isl).sum(),
                'hs'    : (leading_fj.isHsbb).sum()
            }
            vgentype=np.zeros(events.size, dtype=np.int)
            for gentype in self._gentype_map.keys():
                vgentype += self._gentype_map[gentype]*wgentype[gentype]

            if not isFilled:
                hout['sumw'].fill(dataset=dataset, sumw=1, weight=events.genWeight.sum())
                isFilled=True

            cut = selection.all(*selection.names)
            if 'QCD' in dataset:
                ##### template for bb SF #####
                hout['svtemplate'].fill(dataset=dataset,
                                        gentype=vgentype,
                                        #svmass=np.log(leading_SV.mass.sum()),
                                        svmass=np.log(SV_mass.sum()),
                                        ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                        weight=weights.weight()*cut)
                hout['ZHbbvsQCD'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                       tau21=leading_fj.tau21.sum(),
                                       weight=weights.weight()*cut)
                hout['fj1pt'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       fj1pt=leading_fj.sd.pt.sum(),
                                       ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                       weight=weights.weight()*cut)
                hout['fjmass'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       fjmass=leading_fj.msd_corr.sum(),
                                       weight=weights.weight()*cut)
            else:
                ##### template for bb SF #####
                hout['svtemplate'].fill(dataset=dataset,
                                        gentype=vgentype,
                                        #svmass=np.log(leading_SV.mass.sum()),
                                        svmass=np.log(SV_mass.sum()),
                                        ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                        weight=weights.weight())
                hout['ZHbbvsQCD'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                       tau21=leading_fj.tau21.sum(),
                                       weight=weights.weight())
                hout['fj1pt'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       fj1pt=leading_fj.sd.pt.sum(),
                                       ZHbbvsQCD=leading_fj.ZHbbvsQCD.sum(),
                                       weight=weights.weight())
                hout['fjmass'].fill(dataset=dataset,
                                       gentype=vgentype,
                                       fjmass=leading_fj.msd_corr.sum(),
                                       weight=weights.weight())

        return hout

    def postprocess(self, accumulator):
        scale = {}
        for d in accumulator['sumw'].identifiers('dataset'):
            print('Scaling:',d.name)
            dataset = d.name
            if '--' in dataset: dataset = dataset.split('--')[1]
            print('Cross section:',self._xsec[dataset])
            if self._xsec[dataset]!= -1: scale[d.name] = self._lumi*self._xsec[dataset]
            else: scale[d.name] = 1

        for histname, h in accumulator.items():
            if histname == 'sumw': continue
            if isinstance(h, hist.Hist):
                h.scale(scale, axis='dataset')

        return accumulator

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    (options, args) = parser.parse_args()


    with open('metadata/'+options.metadata+'.json') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k,v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids         = load('data/ids.coffea')
    common      = load('data/common.coffea')

    processor_instance=AnalysisProcessor(year=options.year,
                                         xsec=xsec,
                                         corrections=corrections,
                                         ids=ids,
                                         common=common)

    save(processor_instance, 'data/doublebsf'+options.name+'.processor')
