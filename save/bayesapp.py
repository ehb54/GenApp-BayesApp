#!/usr/bin/python3

import json
import io
import sys
import os
import socket # for sending progress messages to textarea
from genapp3 import genapp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess

if __name__=='__main__':

    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)

    ## read Json input
    data_file_path = json_variables['datafile'][0] # name of datafile
    data = data_file_path.split('/')[-1] 
    q_min = json_variables['qmin']
    q_max = json_variables['qmax']
    dmax = json_variables['dmax'] # Maximum diameter
    alpha = json_variables['alpha'] # alpha
    smear = json_variables['smearing'] # smearing constant
    prpoints = json_variables['prpoints'] # number of points in p(r)
    noextracalc = json_variables['noextracalc'] # number of extra calculations
    transformation = json_variables['transform'] # transformation method
    folder = json_variables['_base_directory'] # output folder dir
   
    ## read checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore I use try-except to import
    try:
        dummy = json_variables["alphafixed"]
        alpha = 'f%s' % alpha
    except:
        pass
    try: 
        dummy = json_variables["dmaxfixed"]
        dmax = 'f%s' % dmax 
    except:
        pass
    try:
        dummy = json_variables["fitbackground"]
        fitbackground = 'Y'
    except:
        fitbackground = 'N'
    try:
        dummy = json_variables['nondilute'] # non-dilute sample
        eta = json_variables['eta'] # eta
        method = json_variables['method']
        ratio = json_variables['ratio']
        try:
            dummy = json_variables['ratiofixed']
            ratio = 'f%s' % ratio
        except:
            pass
    except:
        eta = 'default'
        method = 'default'
        ratio = 'default'
    try:
        dummy = json_variables['nonconstantrescale']
        rescale = 'N'
        nbin = json_variables['binsize']
    except:
        rescale = 'C'
        nbin = '10'

    ## make input file with Json input for running iftci
    f = open("inputfile.d",'w')
    f.write('%s\n' % data)
    f.write('%s\n' % q_min)
    f.write('%s\n' % q_max)
    f.write('%s\n' % dmax)
    if eta != 'default':
        f.write('%s\n' % eta)
    else:
        f.write('\n')
    f.write('%s\n' % alpha)
    f.write('%s\n' % smear)
    if ratio != 'default':
        f.write('%s\n' % ratio)
    else:
        f.write('\n')
    if method != 'default':
        f.write('%s\n' % method )
    else:
        f.write('\n')
    f.write('%s\n' % prpoints)
    f.write('%s\n' % noextracalc)
    f.write('%s\n' % transformation)
    f.write('%s\n' % fitbackground)
    f.write('%s\n' % rescale) # rescale method. N: non-constant, C: constant
    if rescale == 'N':
        f.write('%s\n' % nbin)
    else:
        f.write('\n')

    f.write('\n')
    f.close()

    ## messaging
    d = genapp(json_variables)

    ## run bayesfit
    d.udpmessage({"_textarea":"run bayesapp\n"})

    def execute(command,f):
        popen = subprocess.Popen(command, stdout=subprocess.PIPE,bufsize=1)
        lines_iterator = iter(popen.stdout.readline, b"")
        while popen.poll() is None:
            for line in lines_iterator:
                nline = line.rstrip()
                nline_latin = nline.decode('latin')
                out_line = '%s\n' % nline_latin
                d.udpmessage({"_textarea": out_line})
                f.write(out_line)
    
    f = open('stdout.d','w')
    execute(['/opt/genapp/bayesapp/bin/source/iftci','<','inputfile.d'],f)
    f.close()

    ## retrive output from parameter file
    f = open('parameters.d','r')
    lines = f.readlines()
    for line in lines:
        if 'I(0) estimated             :' in line:
            tmp = line.split(':')[1]
            I0 = float(tmp.split('+-')[0])
            #d_I0 = tmp.split('+-')[1]
        if 'Maximum diameter           :' in line:
            tmp = line.split(':')[1]
            dmax = float(tmp.split('+-')[0])
            #d_dmax = tmp.split('+-')[1]
        if 'Radius of gyration         :' in line:
            tmp = line.split(':')[1]
            Rg = float(tmp.split('+-')[0])
            #d_Rg = tmp.split('+-')[1]
        if 'Axial ratio from p(r) (pro):' in line:
            tmp = line.split(':')[1]
            ax_pro = float(tmp.split('+-')[0])
            #d_ax_pro = tmp.split('+-')[1]
        if 'Axial ratio from p(r) (obl):' in line:
            tmp = line.split(':')[1]
            ax_obl = float(tmp.split('+-')[0])
            #d_ax_obl = tmp.split('+-')[1]            
        if 'Reduced Chi-square         :' in line:
            tmp = line.split(':')[1]
            chi2r = float(tmp.split('+-')[0])
            #d_chi2r = tmp.split('+-')[1]
        if 'Background estimated       :' in line:
            background =float( line.split(':')[1])
        if 'Log(alpha) (smoothness)    :' in line:
            tmp = line.split(':')[1]
            alpha = float(tmp.split('+-')[0])
            #d_alpha = tmp.split('+-')[1]
        if 'Number of good parameters  :' in line:
            tmp = line.split(':')[1]
            Ng = float(tmp.split('+-')[0])
            #d_Ng = tmp.split('+-')[1]
        if 'Number of Shannon channels :' in line:
            Ns = float(line.split(':')[1])
        if 'Evidence at maximum        :' in line:
            tmp = line.split(':')[1]
            evidence = float(tmp.split('+-')[0])
            #d_evidence = tmp.split('+-')[1]
        if 'Probability of chi-square  :' in line:
            Prob = float(line.split(':')[1])
            if Prob == 0.0:
                Prob_str = ' < 1e-20'
            else:
                Prob_str = '%1.2e' % Prob
        if 'The exp errors are probably:' in line:
            assessment = line.split(':')[1]
            assessment = assessment[1:] #remove space before the word
        if 'Correction factor          :' in line:
            beta = float(line.split(':')[1])
        line = f.readline()
    f.close()

    ## general plotting settings
    markersize=4
    linewidth=1
    
    ## import and plot p(r)
    r,pr,d_pr = np.genfromtxt('pr.d',skip_header=0,usecols=[0,1,2],unpack=True)

    plt.errorbar(r,pr,yerr=d_pr,marker='.',markersize=markersize,linewidth=linewidth,color='black')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$p(r)$')
    plt.title('p(r)')
    plt.savefig('pr.png',dpi=200)
    plt.close()

    ## import data and fit
    qdat,Idat,sigma = np.genfromtxt('data.d',skip_header=0,usecols=[0,1,2],unpack=True)
    sigma_rs = np.genfromtxt('rescale.d',skip_header=3,usecols=[2],unpack=True)
    qfit,Ifit = np.genfromtxt('fit.d',skip_header=1,usecols=[0,1],unpack=True)
    
    ## calculate residuals
    Ifit_interp = np.interp(qdat,qfit,Ifit)
    R = (Idat-Ifit_interp)/sigma
    maxR = np.ceil(np.amax(abs(R)))
    R_rs = (Idat-Ifit_interp)/sigma_rs
    maxR_rs = np.ceil(np.amax(abs(R_rs)))

    ## plot data, fit and residuals, not rescaled 
    f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]})
    p0.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label='data')
    p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='fit') 
    #p0.plot(qdat,Ifit_interp,color='green',linewidth=linewidth,label='fit_interp') 
    p0.set_ylabel(r'$I(q)$')
    p0.set_yscale('log')
    p0.set_title('fit to data')
    p0.legend(frameon=False)
    
    p1.plot(qdat,R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
    p1.plot(qdat,qdat*0,color='black',zorder=1)
    p1.set_xlabel(r'$q$')
    p1.set_ylabel(r'$I(q)/\sigma$')
    p1.set_ylim(-maxR,maxR)
    p1.set_yticks([-maxR,0,maxR])

    plt.savefig('Iq.png',dpi=200)
    plt.tight_layout()
    plt.close()

    ## import and plot data with rescaled errors
    qresc,Iresc,sigmaresc = np.genfromtxt('rescale.d',skip_header=2,usecols=[0,1,2],unpack=True)
    
    offset = 10
    plt.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',label='data',zorder=0)
    plt.errorbar(qresc,Iresc*offset,yerr=sigmaresc*offset,linestyle='none',marker='.',markersize=markersize,color='blue',label='data with rescaled errors, offset x10',zorder=1) 
    plt.ylabel(r'$I(q)$')
    plt.yscale('log')
    plt.xlabel(r'$q$')
    if rescale == 'N':
        plt.title('input data and data with q-dependent rescaling of errors')
    else:
        plt.title('input data and data with errors rescaled by a factor %1.2f' % beta)
    plt.legend(frameon=False)
    plt.savefig('rescale.png',dpi=200)
    plt.close()

    ## plot data, fit and residuals, rescaled
    f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]})
    p0.errorbar(qdat,Idat,yerr=sigma_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0,label='data with rescaled errors')
    p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='fit')
    p0.set_ylabel(r'$I(q)$')
    p0.set_yscale('log')
    p0.set_title('fit to data with rescaled errors')
    p0.legend(frameon=False)

    p1.plot(qdat,R_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0)
    p1.plot(qdat,qdat*0,color='black',zorder=1)
    p1.set_xlabel(r'$q$')
    p1.set_ylabel(r'$I(q)/\sigma_\mathrm{rescale}$')
    p1.set_ylim(-maxR_rs,maxR_rs)
    p1.set_yticks([-maxR_rs,0,maxR_rs])
    
    plt.savefig('Iq_rs.png',dpi=200)
    plt.tight_layout()
    plt.close()

    ## copy source code to output folder (and rename)
    os.system('cp /opt/genapp/bayesapp/bin/source/iftci_v10.f %s/bift.f' % folder)    

    ## compress output files to zip file
    os.system('zip results.zip pr.d data.d fit.d parameters.d rescale.d scale_factor.d stdout.d bift.f inputfile.d *.png')

    ## generate output
    output = {} # create an empty python dictionary
    
    # files
    output["pr"] = "%s/pr.d" % folder
    output["dataused"] = "%s/data.d" % folder
    output["rescaled"] = "%s/rescale.d" % folder
    output["scale_factor"] = "%s/scale_factor.d" % folder
    output["fitofdata"] = "%s/fit.d" % folder
    output["parameters"] = "%s/parameters.d" % folder
    output["file_stdout"] = "%s/stdout.d" % folder
    output["sourcecode"] = "%s/bift.f" % folder
    output["inputfile"] = "%s/inputfile.d" % folder
    output["prfig"] = "%s/pr.png" % folder
    output["iqfig"] = "%s/Iq.png" % folder
    output["rescalefig"] = "%s/rescale.png" % folder
    output["iqrsfig"] = "%s/Iq_rs.png" % folder
    output["zip"] = "%s/results.zip" % folder

    # values
    output["dmaxout"] = "%1.2f" % dmax
    output["Rg"] = "%1.2f" % Rg
    
    output["I0"] = "%1.2e" % I0
    output["background"] = "%2.5f" % background
    
    output["chi2"] = "%1.2f" % chi2r
    output["prob"] = "%s" % Prob_str
    output["assess"] = "%s" % assessment
    if rescale == 'N':
        output["beta"] = "see scale_factor.d"
    elif rescale == 'C':
        output["beta"] = "%1.2f" % beta 
    
    output["Ng"] = "%1.2f" % Ng
    output["shannon"] = "%1.2f" % Ns
    
    output["logalpha"] = "%1.2f" % alpha 
    output["evidence"] = "%1.2f" % evidence
    output["axratio_pro"] = "%1.2f" % ax_pro
    output["axratio_obl"] = "%1.2f" % ax_obl
    
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output


