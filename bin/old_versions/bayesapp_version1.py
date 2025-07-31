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
        fitratio = json_variables['fitratio'] # fitratio
        estimateratio = json_variables['estimateratio']
        try:
            dummy = json_variables['estimateratiofixed']
            estimateratio = 'f%s' % estimateratio
        except:
            pass
    except:
        pass

    ## make input file with Json input for running iftci
    f = open("inputfile.d",'w')
    f.write('%s\n' % data)
    f.write('%s\n' % q_min)
    f.write('%s\n' % q_max)
    f.write('%s\n' % dmax)
    f.write('\n')
    f.write('%s\n' % alpha)
    f.write('%s\n' % smear)
    f.write('\n' ) # ratio
    f.write('\n' ) #method
    f.write('%s\n' % prpoints)
    f.write('%s\n' % noextracalc)
    f.write('%s\n' % transformation)
    f.write('%s\n' % fitbackground)
    f.write('\n') # screensize
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

    ## import and plot data and fit
    qdat,Idat,sigma = np.genfromtxt('data.d',skip_header=0,usecols=[0,1,2],unpack=True)
    qfit,Ifit = np.genfromtxt('fit.d',skip_header=1,usecols=[0,1],unpack=True)
    
    markersize=4
    linewidth=1
    plt.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',label='data',zorder=0)
    plt.plot(qfit,Ifit,color='black',linewidth=linewidth,label='fit') 
    plt.ylabel(r'$I(q)$')
    plt.yscale('log')
    plt.xlabel(r'$q$')
    plt.title('fit to data')
    plt.legend()
    plt.savefig('Iq.png',dpi=200)
    plt.close()

    ## import and plot p(r)
    r,pr,d_pr = np.genfromtxt('pr.d',skip_header=0,usecols=[0,1,2],unpack=True)

    plt.errorbar(r,pr,yerr=d_pr,marker='.',markersize=markersize,linewidth=linewidth,color='black')
    plt.xlabel(r'$r$')
    plt.ylabel(r'$p(r)$')
    plt.title('p(r)')
    plt.savefig('pr.png',dpi=200)
    plt.close()

    ## import and plot data with rescaled errors
    qresc,Iresc,sigmaresc = np.genfromtxt('rescale.d',skip_header=2,usecols=[0,1,2],unpack=True)
    
    offset = 10
    plt.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',label='data',zorder=0)
    plt.errorbar(qresc,Iresc*offset,yerr=sigmaresc*offset,linestyle='none',marker='.',markersize=markersize,color='blue',label='data with rescaled errors, offset x10',zorder=1) 
    plt.ylabel(r'$I(q)$')
    plt.yscale('log')
    plt.xlabel(r'$q$')
    plt.title('input data and data with errors rescaled by a factor %1.2f' % beta)
    plt.legend()
    plt.savefig('rescale.png',dpi=200)
    plt.close()

    ## compress files to zip file
    os.system('zip results.zip pr.d data.d fit.d parameters.d rescale.d stdout.d *.png')
    
    ## generate output
    output = {} # create an empty python dictionary
    output["pr"] = "%s/pr.d" % folder
    output["dataused"] = "%s/data.d" % folder
    output["rescaled"] = "%s/rescale.d" % folder
    output["fitofdata"] = "%s/fit.d" % folder
    output["parameters"] = "%s/parameters.d" % folder
    output["file_stdout"] = "%s/stdout.d" % folder
    output["prfig"] = "%s/pr.png" % folder
    output["iqfig"] = "%s/Iq.png" % folder
    output["rescalefig"] = "%s/rescale.png" % folder
    output["zip"] = "%s/results.zip" % folder
    output["chi2"] = "%1.2f" % chi2r
    output["logalpha"] = "%1.2f" % alpha 
    output["dmaxout"] = "%1.2f" % dmax
    output["Rg"] = "%1.2f" % Rg
    output["I0"] = "%1.2e" % I0
    output["Ng"] = "%1.2f" % Ng
    output["shannon"] = "%1.2f" % Ns
    output["evidence"] = "%1.2f" % evidence
    output["axratio_pro"] = "%1.2f" % ax_pro
    output["axratio_obl"] = "%1.2f" % ax_obl
    output["background"] = "%2.5f" % background
    output["assess"] = "%s" % assessment
    output["prob"] = "%s" % Prob_str
    output["beta"] = "%1.2f" % beta 
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output


