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
import time

if __name__=='__main__':

    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)

    ## read Json input
    data_file_path = json_variables['datafile'][0] # name of datafile
    #data = data_file_path.split('/')[-1] 
    prefix = data_file_path.split('/')[-1] 
    q_min = json_variables['qmin']
    q_max = json_variables['qmax']
    nrebin = json_variables['nrebin'] # max number of points to rebin data to
    dmax = json_variables['dmax'] # Maximum diameter
    alpha = json_variables['alpha'] # alpha
    smear = json_variables['smearing'] # smearing constant
    prpoints = json_variables['prpoints'] # number of points in p(r)
    noextracalc = json_variables['noextracalc'] # number of extra calculations
    transformation = json_variables['transform'] # transformation method
    folder = json_variables['_base_directory'] # output folder dir

    ## messaging
    d = genapp(json_variables)
    
    ## fortran77 bug: cannot use long file names
    if len(prefix)>48:
        d.udpmessage({"_textarea":"-----------------------------------------------------------------------------------------\n"})
        d.udpmessage({"_textarea":"Warning:\n"})
        d.udpmessage({"_textarea":"long data name (>48 characters). Too much for Fortran77. renaming before running bift\n"})
        d.udpmessage({"_textarea":"filename used by bift fotran77 code: data_name_too_long_for_fortran77.dat\n"})
        d.udpmessage({"_textarea":"this will not affect the result, but the new name appears in the input file: inputfile.dat\n"})
        d.udpmessage({"_textarea":"------------------------------------------------------------------------------------------\n\n"})
        data = 'data_name_too_long_for_fortran77.dat'
        os.system('cp %s %s/%s' % (data_file_path,folder,data))
    else:
        data = prefix

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
    """
    try:
        dummy = json_variables['nondilute'] # non-dilute sample
        eta = json_variables['eta'] # eta
        R_HS = json_variables['R_HS']
    except:
        eta = 'default'
        R_HS = 'default'
    """
    try:
        dummy = json_variables['nonconstantrescale']
        rescale = 'N'
        nbin = json_variables['binsize']
    except:
        rescale = 'C'
        nbin = '10'
    try:
        dummy = json_variables['outlieranalysis']
        outlier = 1
    except: 
        outlier = 0
    try:
        dummy = json_variables['logx']
        logx = 1
    except:
        logx = 0
    try:
        dummy = json_variables['make_pr_bin']
        make_pr_bin = 1
        pr_binsize = float(json_variables['pr_binsize']) # binsize in pr_bin - p(r) interpolated on new r grid
        units = json_variables['units']
    except:
        make_pr_bin = 0

    ## make input file with Json input for running bift
    f = open("inputfile.dat",'w')
    f.write('%s\n' % data)
    f.write('%s\n' % q_min)
    f.write('%s\n' % q_max)
    f.write('%s\n' % nrebin)
    f.write('%s\n' % dmax)
    f.write('\n')
    f.write('%s\n' % alpha)
    f.write('%s\n' % smear)
    f.write('\n')
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

    ## check q range
    try:
        qmax = float(q_max)
    except:
        qmax = 10.0
    try:
        qmin = float(q_min)
    except:
        qmin = 0.0
    q_diff = qmax - qmin 
    if q_diff < 0.0:
        out_line = '\n\n!!!ERROR!!!\nqmin should be smaller than qmax.\n\n'
        d.udpmessage({"_textarea": out_line})
        sys.exit()

    ## run bayesfit
    d.udpmessage({"_textarea":"----------------------\n"})
    d.udpmessage({"_textarea":"running bayesapp...\n"})
    d.udpmessage({"_textarea":"----------------------\n\n"})

    def execute(command,f):
        start_time = time.time()
        maximum_output_size = 1000000 # maximum output size in number of characters
        maximum_time = 300
        total_output_size = 0
        popen = subprocess.Popen(command, stdout=subprocess.PIPE,bufsize=1)
        lines_iterator = iter(popen.stdout.readline, b"")
        while popen.poll() is None:
            for line in lines_iterator:
                nline = line.rstrip()
                nline_latin = nline.decode('latin')
                total_output_size += len(nline_latin)
                total_time = time.time() - start_time
                if total_output_size > maximum_output_size:
                    popen.terminate()
                    out_line = '\n\n!!!ERROR!!!\nProcess stopped - could not find solution. Is data input a SAXS/SANS dataset with format (q,I,dI)?\n\n'
                    d.udpmessage({"_textarea": out_line})
                    sys.exit()
                elif total_time > maximum_time:
                    popen.terminate()
                    out_line = '\n\n!!!ERROR!!!\nProcess stopped - reached max time of 5 min (300 sec). Is data input a SAXS/SANS dataset with format (q,I,dI)?. If data is large (several thousand data points), consider rebinning the data. Or reduce number of points in p(r).\n\n'
                    d.udpmessage({"_textarea": out_line})
                    sys.exit()
                else:
                    out_line = '%s\n' % nline_latin
                    d.udpmessage({"_textarea": out_line})
                f.write(out_line)

    f = open('stdout.dat','w')
    path = os.path.dirname(os.path.realpath(__file__))
    os.system('cp %s/source/p_table.dat %s' % (path,folder)) # copy cormap pvalue table to folder (for bift to read)
    execute([path + '/source/bift','<','inputfile.dat'],f)
    f.close()

    ## retrive output from parameter file
    f = open('parameters.dat','r')
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
        if 'Longest run                :' in line:
            Rmax = float(line.split(':')[1])
        if 'Expected longest run       :' in line:
            tmp = line.split(':')[1]
            Rmax_expect = float(tmp.split('+-')[0])
            dRmax_expect = float(tmp.split('+-')[1])
        if 'Prob., longest run (cormap):' in line:
            p_Rmax = float(line.split(':')[1])
            if p_Rmax == 0.0:
                p_Rmax_str = '< 1e-4'
            else:
                if p_Rmax<0.001:
                    p_Rmax_str = '%1.2e' % p_Rmax
                else:
                    p_Rmax_str = '%1.3f' % p_Rmax
        if 'Number of runs             :' in line:
            NR = float(line.split(':')[1])
        if 'Expected number of runs    :' in line:
            tmp = line.split(':')[1]
            NR_expect = float(tmp.split('+-')[0])
            dNR_expect = float(tmp.split('+-')[1])
        if 'Prob.,  number of runs     :' in line:
            p_NR = float(line.split(':')[1])
        line = f.readline()
    f.close()

    ## general plotting settings
    markersize=4
    linewidth=1
    
    ## import p(r)
    r,pr,d_pr = np.genfromtxt('pr.dat',skip_header=0,usecols=[0,1,2],unpack=True)
   
    if make_pr_bin:
        ## intepolate pr on grid with binsize of pr_binsize
        if units == 'nm':
            pr_binsize /= 10
        r_bin = np.arange(0,r[-1],pr_binsize)
        pr_bin = np.interp(r_bin,r,pr)
        n = len(r)/len(r_bin) 
        pr_bin_max = np.interp(r_bin,r,pr+d_pr)
        pr_bin_min = np.interp(r_bin,r,pr-d_pr)
        d_pr_bin = ((pr_bin_max-pr_bin_min)/2)/np.sqrt(n)
        with open('pr_bin.dat','w') as f:
            for x,y,z in zip(r_bin,pr_bin,d_pr_bin):
                f.write('%10.10f %10.10e %10.10e\n' % (x,y,z))

    ## import data and fit
    qdat,Idat,sigma = np.genfromtxt('data.dat',skip_header=0,usecols=[0,1,2],unpack=True)
    sigma_rs = np.genfromtxt('rescale.dat',skip_header=3,usecols=[2],unpack=True)
    qfit,Ifit = np.genfromtxt('fit.dat',skip_header=1,usecols=[0,1],unpack=True)
    
    ## interpolate fit on q-values from data
    Ifit_interp = np.interp(qdat,qfit,Ifit)
    with open('fit_q.dat','w') as f:
        for x,y in zip(qdat,Ifit_interp):
            f.write('%10.10f %10.10f\n' % (x,y))

    ## calculate residuals
    R = (Idat-Ifit_interp)/sigma
    maxR = np.ceil(np.amax(abs(R)))
    R_rs = (Idat-Ifit_interp)/sigma_rs
    maxR_rs = np.ceil(np.amax(abs(R_rs)))

    ## outlier analysis
    x = np.linspace(-10,10,1000)
    pdx = np.exp(-x**2/2)
    norm = np.sum(pdx)
    p = np.zeros(len(R))    
    for i in range(len(R)):
        idx_i = np.where(x>=abs(R[i]))
        p[i] = np.sum(pdx[idx_i])
    p /= norm
    p *= len(R) #correction for multiple testing
    idx = np.where(p<0.03)
    Noutlier = len(idx[0])
    with open('outlier_filtered.dat','w') as f:
        f.write('# data, with outliers filtered out\n')
        f.write('# %d outliers were removed\n' % Noutlier)
        for i in range(len(R)):
            if i in idx[0]:
                pass
            else:
                f.write('%e %e %e\n' % (qdat[i],Idat[i],sigma[i]))

    ## plot p(r)
    plt.errorbar(r,pr,yerr=d_pr,marker='.',markersize=markersize,linewidth=linewidth,color='black',label='p(r)')
    if make_pr_bin:
        plt.errorbar(r_bin,pr_bin,d_pr_bin,marker='.',markersize=markersize,linewidth=linewidth,color='green',label='p(r), fixed binsize')
        plt.legend(frameon=False)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$p(r)$')
    plt.title('p(r)')
    plt.tight_layout()
    plt.savefig('pr.png',dpi=200)
    plt.close()

    ## plot data, fit and residuals, not rescaled 
    f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
    p0.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label='data')
    if logx:
        p0.set_xscale('log')
        p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,label='fit')
    else:
        p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='fit') 
    p0.set_ylabel(r'$I(q)$')
    p0.set_yscale('log')
    p0.set_title('fit to data')

    p1.plot(qdat,R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
    if logx:
        p1.set_xscale('log')
        p1.plot(qdat,Idat*0,linewidth=linewidth,color='black',zorder=1)
        if outlier:
            p1.plot(qdat,-3*np.ones(len(Idat)),linewidth=linewidth,linestyle='--',color='grey',zorder=2,label=r'$\pm 3\sigma$')
            p1.plot(qdat,3*np.ones(len(Idat)),linewidth=linewidth,linestyle='--',color='grey',zorder=3)
    else:
        p1.plot(qfit,Ifit*0,linewidth=linewidth,color='black',zorder=1)
        if outlier:
            p1.plot(qfit,-3*np.ones(len(Ifit)),linewidth=linewidth,linestyle='--',color='grey',zorder=2,label=r'$\pm 3\sigma$')
            p1.plot(qfit,3*np.ones(len(Ifit)),linewidth=linewidth,linestyle='--',color='grey',zorder=3)

    ## plot outliers
    if outlier:
        if len(idx[0]>0):
            p0.plot(qdat[idx],Idat[idx],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='grey',zorder=4,label='potential outliers')
            p1.plot(qdat[idx],R[idx],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='grey',zorder=4)

    p1.set_xlabel(r'$q$')
    p1.set_ylabel(r'$I(q)/\sigma$')
    try:
        p1.set_ylim(-maxR,maxR)
        if outlier:
            p1.set_yticks([-maxR,-3,0,3,maxR])
        else:
            p1.set_yticks([-maxR,0,maxR])
    except:
        d.udpmessage({"_textarea":"WARNING: Some residuals are either NaN or inf - bad fit?\n"})
        d.udpmessage({"_textarea":"         probably just a numerical instability\n"})
        d.udpmessage({"_textarea":"         try changing the number of points in p(r)\n"})

    p0.legend(frameon=False)
    
    plt.savefig('Iq.png',dpi=200)
    plt.tight_layout()
    plt.close()
    
    ## import and plot data with rescaled errors
    if Prob < 0.003:
        qresc,Iresc,sigmaresc = np.genfromtxt('rescale.dat',skip_header=2,usecols=[0,1,2],unpack=True)
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
        if logx:
            plt.xscale('log')
        plt.savefig('rescale.png',dpi=200)
        plt.close()

        ## plot data, fit and residuals, rescaled
        f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
        p0.errorbar(qdat,Idat,yerr=sigma_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0,label='data with rescaled errors')
        if logx:
            p0.set_xscale('log')
            p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,zorder=1,label='fit')
        else:
            p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='fit')
        p0.set_ylabel(r'$I(q)$')
        p0.set_yscale('log')
        p0.set_title('fit to data with rescaled errors')
        p0.legend(frameon=False)

        p1.plot(qdat,R_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0)
        if logx:
            p1.set_xscale('log')
            p1.plot(qdat,Idat*0,linewidth=linewidth,color='black',zorder=1)
        else:
            p1.plot(qfit,Ifit*0,linewidth=linewidth,color='black',zorder=1)
        p1.set_xlabel(r'$q$')
        p1.set_ylabel(r'$I(q)/\sigma_\mathrm{rescale}$')
        try:
            p1.set_ylim(-maxR_rs,maxR_rs)
            p1.set_yticks([-maxR_rs,0,maxR_rs])
        except:
            d.udpmessage({"_textarea":"WARNING: Some residuals are either NaN or inf - bad fit?\n"})
            d.udpmessage({"_textarea":"         probably just a numerical instability\n"})
            d.udpmessage({"_textarea":"         try changing the number of points in p(r)\n"})
        plt.savefig('Iq_rs.png',dpi=200)
        plt.tight_layout()
        plt.close()

    ## copy source code to output folder (and rename)
    os.system('cp %s/source/bift.f %s' % (path,folder))

    ## compress output files to zip file
    os.system('zip results_%s.zip pr.dat pr_bin.dat data.dat fit.dat fit_q.dat parameters.dat rescale.dat outlier_filtered.dat scale_factor.dat stdout.dat p_table.dat bift.f inputfile.dat *.png' % prefix)

    ## generate output
    output = {} # create an empty python dictionary
    
    # files
    output["pr"] = "%s/pr.dat" % folder
    if make_pr_bin:
        output["pr_bin"] = "%s/pr_bin.dat" % folder
    output["dataused"] = "%s/data.dat" % folder
    output["rescaled"] = "%s/rescale.dat" % folder
    output["outlier_filtered"] = "%s/outlier_filtered.dat" % folder
    output["scale_factor"] = "%s/scale_factor.dat" % folder
    output["fitofdata"] = "%s/fit.dat" % folder
    output["fit_q"] = "%s/fit_q.dat" % folder
    output["parameters"] = "%s/parameters.dat" % folder
    output["file_stdout"] = "%s/stdout.dat" % folder
    output["p_table"] = "%s/p_table.dat" % folder
    output["sourcecode"] = "%s/bift.f" % folder
    output["inputfile"] = "%s/inputfile.dat" % folder
    output["prfig"] = "%s/pr.png" % folder
    output["iqfig"] = "%s/Iq.png" % folder
    if Prob<0.003:
        output["rescalefig"] = "%s/rescale.png" % folder
        output["iqrsfig"] = "%s/Iq_rs.png" % folder
    output["zip"] = "%s/results_%s.zip" % (folder,prefix)

    # values
    output["dmaxout"] = "%1.2f" % dmax
    output["Rg"] = "%1.2f" % Rg
    
    output["I0"] = "%1.2e" % I0
    output["background"] = "%2.5f" % background
    
    output["chi2"] = "%1.2f" % chi2r
    output["prob"] = "%s" % Prob_str
    output["assess"] = "%s" % assessment
    if rescale == 'N':
        output["beta"] = "see scale_factor.dat"
    elif rescale == 'C':
        output["beta"] = "%1.2f" % beta 
    
    output["Rmax"] = "%1.1f" % Rmax
    output["Rmax_expect"] = "%1.1f +/- %1.1f" % (Rmax_expect,dRmax_expect)
    output["p_Rmax"] = "%s" % p_Rmax_str
    output["NR"] = "%1.1f" % NR
    output["NR_expect"] = "%1.1f +/- %1.1f" % (NR_expect,dNR_expect)
    if p_NR < 0.001:
        output["p_NR"] = "%1.2e" % p_NR
    else:
        output["p_NR"] = "%1.3f" % p_NR
    output["Noutlier"] = "%d" % Noutlier
    output["Ng"] = "%1.2f" % Ng
    output["shannon"] = "%1.2f" % Ns
#    output["shannon_0"] = "%1.2f" % Ns_0 
    output["logalpha"] = "%1.2f" % alpha 
    output["evidence"] = "%1.2f" % evidence
#    output["axratio_pro"] = "%1.2f" % ax_pro
#    output["axratio_obl"] = "%1.2f" % ax_obl
    
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output

