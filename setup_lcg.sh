#!/usr/bin/env bash

source env_lcg.sh

pip install --user coffea==0.7.22
pip install --user https://github.com/mcremone/rhalphalib/archive/py3.zip
pip install --user xxhash
# progressbar, sliders, etc.
jupyter nbextension enable --py widgetsnbextension

