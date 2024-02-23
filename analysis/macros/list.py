#!/usr/bin/env python
import os
from optparse import OptionParser
import subprocess
import json 

parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
(options, args) = parser.parse_args()

globalredirect = "root://xrootd-cms.infn.it/"

campaigns ={}
campaigns['2016'] = 'UL2016'
campaigns['2017'] = 'UL2017'
campaigns['2018'] = 'UL2018'

campaigns_TTTo = {}
campaigns_TTTo['2016'] = 'UL16' #different naming convention for TTTo
campaigns_TTTo['2017'] = 'UL17'
campaigns_TTTo['2018'] = 'UL18'


slist = ['SingleMuon', 'TTTo']

if options.year == '2016' or options.year == '2017':
    slist += ['SingleElectron']
if options.year == '2018':
    slist += ['EGamma']

for sample in slist:
    # Initialize an empty dictionary to hold dataset information for the current sample
    sample_data_structure = {}

    if sample == 'TTTo':
        dataset_query = "dasgoclient --query=\"dataset=/*{0}*/*{1}*JMENano*/*\" ".format(sample, campaigns_TTTo[options.year])

    # Construct the query to get datasets
    else:
        dataset_query = "dasgoclient --query=\"dataset=/{0}/*{1}*JMENano*/*\" ".format(sample, campaigns[options.year])
    
    print('Querying datasets:', dataset_query)
    
    # Execute the query and decode the output
    datasets = subprocess.check_output(dataset_query, shell=True).decode('utf-8').strip().split('\n')
    
    # Loop over each dataset to get the list of files
    for dataset in datasets:
        # Make sure dataset is not empty
        if not dataset: continue
        
        # Construct the query to get files for the dataset
        file_query = "dasgoclient --query=\"file dataset={0}\"".format(dataset)
        print('Querying files in dataset:', file_query)
        
        # Execute the query and decode the output
        files = subprocess.check_output(file_query, shell=True).decode('utf-8').strip().split('\n')
        files = [globalredirect + file for file in files if file]  # Prepend redirect and filter out empty lines
        
        # Add the dataset and its files to the data structure
        sample_data_structure[dataset] = {
            "files": files
        }
    metadata_dir = os.path.abspath('metadata')
    # Write the sample's data structure to a JSON file
    sample_json_filename = os.path.join(metadata_dir, '{0}.json'.format(sample))
    with open(sample_json_filename, 'w') as outfile:
        json.dump(sample_data_structure, outfile, indent=4)

    print("Metadata for {0} written to {1}".format(sample, sample_json_filename))

