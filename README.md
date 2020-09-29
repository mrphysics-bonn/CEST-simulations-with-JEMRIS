# CEST simulations with JEMRIS

This repository contains a Jupyter Notebook with an example code for CEST simulations with the open source framework JEMRIS (www.jemris.org).

- continuous wave saturation
- arbitrary labelling

A CEST experiment with continuous wave saturation can be used from the Python GUI without any additional knowledge of JEMRIS.
The design of arbitrary saturation modules is done with the MATLAB GUI of JEMRIS.


### Installation/Requirements

The Jupyter Notebook can be used with a local Python 3.6 installation containing Jupyter Lab and numpy.

JEMRIS: https://github.com/JEMRIS/jemris.git \
Installation instructions: http://jemris.org/ug_install.html


### Background
Chemical exchange saturation transfer (CEST) experiments are based on the chemical exchange between a dilute molecules and water protons. By off-resonant saturation, solute pools can be measured indirectly. The relaxation of water protons is affected by the spin population of the solute proton pool due to inter- and intra-molecular magnetization transfer (MT) processes mediated by scalar or dipolar spin-spin couplings or chemical exchange.

The theory, that describes the dynamics of the coupled magnetization vectors in spin-exchange experiments, is based on the Bloch-McConnell (BM) equations [1,2].


### References
 1. HM McConnell _Reaction Rates by Nuclear Magnetic Resonance._ 1958. J Chem Phys. 28;430. https://doi.org/10.1063/1.1744152
 2. M Zaiss & P Bachert. _Chemical exchange saturation transfer (CEST) and MR Z-spectroscopy in vivo: a review of theoretical approaches and methods._ 2013. Phys Med Biol. 21;58(22):R221-69. https://doi.org/10.1088/0031-9155/58/22/R221
