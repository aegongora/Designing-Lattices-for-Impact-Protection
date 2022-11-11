# Designing-Lattices-for-Impact-Protection
## Overview
The code in this repository is related to the manuscript: Gongora, A. E., Snapp, K. L., Pang, R., Tiano, T. M., Reyes, K. G., Whiting, E., ... & Brown, K. A. (2022). Designing lattices for impact protection using transfer learning. Matter, 5(9), 2829-2846.

## Octet Model
The octet model is generated using force-displacement and acceleration measurements. The relevant measurements are stored in the pickle file "Data_OctetModel.pkl". The code to generate the model is located in "main_octetmodel.py".

## Octet and Octahedral Model
The octet-octahedral model is generated using force-displacement and acceleration measurements from both the octet and octahedral datasets. The relevant measurements are stored in the pickle file "Data_OctetOctahedralModel.pkl". The code to generate the model is located in "main_octetoctamodel.py". The relevant files for predicting the performance of the alternative designs are also in the pickle file "Data_OctetOctahedralModel.pkl". 

## General Notes
In the load pickle files, column header "WADL ID Number" corresponds to "WADL Test Number" in the shared dataset available at: https://www.kablab.org/lattice-impact. The column header. 

Additionally, the column header "Impact Testing Number" corresponds to the "Impact Test Number" in the shared dataset available at: https://www.kablab.org/lattice-impact. The column header. 
