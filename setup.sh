#!/usr/bin/env bash

source env.sh

python3 -m pip install --user coffea==0.7.22
python3 -m pip install --user https://github.com/mcremone/rhalphalib/archive/master.zip
python3 -m pip install --user xxhash
python3 -m pip install --user correctionlib[convert]
# progressbar, sliders, etc.
#jupyter nbextension enable --py widgetsnbextension
