# BayesApp (hosted on GenApp)
Source code for the BayesApp software package    

online web application at GenApp: https://somo.chem.utk.edu/bayesapp/    

Calculates the pair distance distribution function, p(r), from a SAXS/SANS dataset by a Bayesian indirect Fourier transformation (BIFT) algorithm    

Written by:              Steen Hansen    
With contributions from: Andreas Haahr Larsen, Martin Cramer Pedersen     



## File overview

#### bin/source/bift.f
source code (fortran)    
compile.sh: example bash script for compiling bift.f (system dependent)    
bift (no extension): executable (system dependent)     

#### bin/bayesapp.py
python wrapper for web appllication (takes input from GUI, runs bift, sends output to GUI)  

#### modules/bayesapp.json
GUI for web application   

#### .github/workflows
instructions for compilation on different OS. 

## Instructions for running the program

#### webinterface with GUI      
https://somo.chem.utk.edu/bayesapp/     

#### run locally (batch mode)    
1) Compilation (linux):  gfortran -march=native -O3 bift.f -o bift    
                         -march=native may be left out     
                         -m64 or -m32 may be added     
                         it may be necessary to include the "-static" flag    
                         ... depending on system  
                         

2) Run:                  bift < inputfile.d                        

3) The file: inputfile.d has to contain the 16 lines:    
                                                          input format    
line 1:  the name of the data file     - compulsory -    [string]    
line 2:  q_min                         or a blank line   [float]    
line 3:  q_max                         or a blank line   [float]    
line 4:  nrebin                        or a blank line   [integer]    
line 5:  d_max                         or a blank line   [float]*    
line 6:  eta (non-dilute solutions)    or a blank line   [float]**    
line 7:  alpha                         or a blank line   [float]*    
line 8:  smearing constant             or a blank line   [float]    
line 9:  ratio (non-dilute solutions)  or a blank line   [float]**    
line 10:  method (non-dilute solutions) or a blank line   [N]one or [M]oment or [E]vidence**    
line 11: number of points in p(r)      or a blank line   [integer]    
line 12: number of extra cal           or a blank line   [integer]    
line 13: transformation                or a blank line   [D]ebye (default) or [N]egative or [M]axEnt    
line 14: fit constant background       or a blank line   [Y]es or [N]o    
line 15: non-const rescaling           or a blank line   [N]on-constant or [C]onstant    
line 16: min points per Shannon bin    or a blank line   [integer]    

\* use prefix "f" to Fix value, i.e. f22.0 instead of 22.0 for d_max    
  if no prefix is given, the input value is used as initial value in the optimization search    
    
** This part of the code (p(r) for non-dilute scatterers) is not maintained and therefore not part of the GUI. There is no garantee that it is working, so use this option with care and be extra critical when interpreting the results. None: fit only alpha and dmax, Moment: fit alpha, d_max and eta, Evidence: fit alpha, dmax, eta and ratio    
    
NB NB The input values are the same as at the web site     
Only the first line has to be a non blank line    
For further information see the website    
The program was tested with gfortran 4.4 and it may     
be necessary to include the "-static" flag.     

