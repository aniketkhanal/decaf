import json
import dask
import os
import time
from coffea.dataset_tools import preprocess

path = "decaf/analysis/data/"

## uncomment this part if you need to create a input_datasets.json file for preprocessing  ##
## required if adding new datasets ##
fileset = {
    'GluGluToHHTo2B2VLNu2J': {
        "files" : { 'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_3.root': "Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_4.root': "Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_10.root':"Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_11.root':"Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_8.root': "Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_12.root':"Events",
                    'root://cms-xrd-global.cern.ch//store/user/algomez/JMEnano/GluGluToHHTo2B2VLNu2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD_JMENanoAODv9_PrivateProdv1p1/240906_201658/0000/B2G-RunIISummer20UL18NanoAODv9-00923_7.root': "Events",
        },
        "metadata" :  {"dataset": "GluGluToHHTo2B2VLNu2J", "year" : "2018", "lumi" : 59.83 }
    },
    'TTToSemiLeptonic': {
        "files": {  'root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/100000/02C616A0-8765-F349-87AB-B2C32B8DCAC2.root': "Events",
                    'root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/100000/05FF754A-602C-C647-944E-E35D152F5F3E.root': "Events", 
                    'root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/100000/08A35EA4-806F-9C4B-97A4-EA49543128CD.root': "Events"
        },
        "metadata" :  {"dataset": "TTToSemiLeptonic", "year" : "2018", "lumi" : 59.83 }
    }                                                                                                                                   
}

###########################################################

## pre-process the datasets present in input_datasets.json ###
## this should only be done once for any given dataset. The process will later use the output json files ##


@dask.delayed
def process_fileset(fs):
    return preprocess(
        fs,
        align_clusters=False,
        step_size=100000,
        files_per_batch=2,
        skip_bad_files=True,
        save_form=False,
    )[0]

# Process directly
dataset_runnable = dask.compute(
    process_fileset(fileset),
    scheduler='threads',
    resources={"cores": 4},
    prune_depth=2,
    lazy_transfers=True,
)[0]

samples_ready = {}
for dataset_name in dataset_runnable:
    samples_ready[dataset_name] = dataset_runnable[dataset_name]

with open(f'{path}/samples_ready.json', "w") as fout:
    json.dump(samples_ready, fout)

print(f'All samples processed. Output in {path}/samples_ready.json')

