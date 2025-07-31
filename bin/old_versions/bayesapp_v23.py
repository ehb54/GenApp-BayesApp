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
#import subprocess
#import time
from helpfunctions import *

if __name__=='__main__':

    #################################################################################
    ## set up Json-related variables
    #################################################################################

    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    
    ## set up messaging
    d = genapp(json_variables)

    #################################################################################
    ## read Json input
    #################################################################################
    
    # general input
    data_file_path = json_variables['datafile'][0] # name of datafile
    prefix = data_file_path.split('/')[-1] 
    q_min = json_variables['qmin']
    q_max = json_variables['qmax']
    units = json_variables['units']

    # about reading checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore I use try-except to import
    
    # read input Guinier analysis
    try:
        dummy = json_variables['Guinier']
        Guinier = 1
        qmaxRg_in = json_variables['Guinier_qmaxRg'] # qmax*Rg in Guinier analysis
        Guinier_skip_in = json_variables['Guinier_skip'] # skip first points in Guinier analysis   
    except:
        Guinier = 0

    # read input Kratky plot
    try:
        dummy = json_variables['Kratky']
        Kratky = 1
        try:
            dummy = json_variables['Kratky_dim']
            Kratky_dim = 1
        except:
            Kratky_dim = 0
        try:
            dummy = json_variables['Kratky_Mw']
            Kratky_Mw = 1
        except:
            Kratky_Mw = 0
    except:
        Kratky = 0

    # read input Porod plot
    try:
        dummy = json_variables['Porod']
        Porod = 1
        try:
            Porod_limit = float(json_variables['Porod_limit']) 
        except:
            Porod_limit = 0
    except:
        Porod = 0

    # read extra options
    try:
        dummy = json_variables['options']
        dmax = json_variables['dmax'] # Maximum diameter
        transformation = json_variables['transform'] # transformation method
        nrebin = json_variables['nrebin'] # max number of points to rebin data to
        alpha = json_variables['alpha'] # alpha
        smear = json_variables['smearing'] # smearing constant
        prpoints = json_variables['prpoints'] # number of points in p(r)
        noextracalc = json_variables['noextracalc'] # number of extra calculations
        rescale_mode = json_variables['rescale_mode'] # model name
        Bg = json_variables['Bg']
        try:
            dummy = json_variables['dmaxfixed']
            dmax = 'f%s' % dmax
        except:
            pass
        if rescale_mode == 'N':
            nbin = json_variables['binsize']
        try:
            dummy = json_variables["alphafixed"]
            alpha = 'f%s' % alpha
        except:
            pass
        try:
            dummy = json_variables["fitbackground"]
            fitbackground = 'Y'
        except:
            fitbackground = 'N'
        try:
            dummy = json_variables['logx']
            logx = 1
        except:
            logx = 0
        try:
            dummy = json_variables['make_pr_bin']
            make_pr_bin = 1
            pr_binsize = float(json_variables['pr_binsize']) # binsize in pr_bin - p(r) interpolated on new r grid
        except:
            make_pr_bin = 0
        try:
            skip_first = int(json_variables['skip_first']) # skip first points
        except:
            skip_first = ''
        try:
            dummy = json_variables["outlier_ite"]
            outlier_ite = 1
        except:
            outlier_ite = 0
        try:
            dummy = json_variables["fast_run"]
            fast_run = 1
        except:
            fast_run = 0
    except:
        transformation = 'A'
        dmax = ''
        prpoints = '' # set default value
        nrebin = '500'
        alpha = ''
        fitbackground = 'Y'
        logx = 0
        make_pr_bin = 0
        smear = ''
        noextracalc = ''
        rescale_mode = 'C'
        skip_first = ''
        outlier_ite = 0
        fast_run = 0
        Bg = ''
    qmax_ite = 0 # disable this function for now...

    ## read output folder
    folder = json_variables['_base_directory'] # output folder dir

    #################################################################################
    ## adjust input
    #################################################################################
    
    ## remove spaces and brackets from name
    prefix = prefix.replace(" ","_").replace("(","").replace(")","").replace("[","").replace("]","")
   
    ## naming bug fix: fortran77 cannot use long file names
    if len(prefix)>48:
        d.udpmessage({"_textarea":"-----------------------------------------------------------------------------------------\n"})
        d.udpmessage({"_textarea":"Warning:\n"})
        d.udpmessage({"_textarea":"long data name (>48 characters). Too much for Fortran77. renaming before running bift\n"})
        d.udpmessage({"_textarea":"filename used by bift fotran77 code: data_name_too_long_for_fortran77.dat\n"})
        d.udpmessage({"_textarea":"this will not affect the result, but the new name appears in the input file: inputfile.dat\n"})
        d.udpmessage({"_textarea":"------------------------------------------------------------------------------------------\n\n"})
        data = 'data_name_too_long_for_fortran77.dat'
        os.system('cp "%s" "%s/%s"' % (data_file_path,folder,data))
    else:
        data = prefix
        os.system('cp "%s" "%s/%s"' % (data_file_path,folder,data))

    ## check if qmin should be updated (due to skip_first option)
    try:
        qmax = float(q_max)
    except:
        qmax = 100.0
    try:
        qmin = float(q_min)
    except:
        qmin = 0.0
    header,footer = get_header_footer(data)
    try:
        if skip_first == '':
            q_check = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0],unpack=True)
        else:
            q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True)
    except:
        if skip_first == '':
            q_check = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
        else:
            q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
    if q_check[0] > qmin:
        qmin = q_check[0]
    if q_check[-1] < qmax:
        qmax = q_check[-1]

    ## automatically detect units
    if units == 'auto':
        if qmax > 2.0:
            units = 'nm'
            if qmax > 7:
                qmax = 7
        else:
            units = 'A'
            if qmax > 0.7:
                qmax = 0.7

    ## check validity of q range
    q_diff = qmax - qmin
    if q_diff < 0.0:
        out_line = '\n\n!!!ERROR!!!\nqmin should be smaller than qmax.\n\n'
        d.udpmessage({"_textarea": out_line})
        sys.exit()

    ## print start bayesapp message
    d.udpmessage({"_textarea":"=================================================================================\n"})
    d.udpmessage({"_textarea":"running bayesapp...\n"})
    d.udpmessage({"_textarea":"=================================================================================\n"})
    out_line = 'header lines in datafile:  %d\n' % header
    d.udpmessage({"_textarea":out_line})
    out_line = 'footer lines in datafile:  %d\n' % footer
    d.udpmessage({"_textarea":out_line})
    d.udpmessage({"_textarea":"=================================================================================\n"})

    ##################################
    # beginning of outlier while loop 
    ##################################
    CONTINUE_OUTLIER = 1
    count_ite,max_ite,Noutlier_prev,outliers_removed = 0,20,1e3,0
    while CONTINUE_OUTLIER:

        ##################################
        # beginning of qmax while loop 
        ##################################
        CONTINUE_QMAX = 1
        count_ite_qmax,max_ite_qmax = 0,1
        while CONTINUE_QMAX: 

            if (not dmax or transformation == 'A' or not prpoints or skip_first == ''):
                ## initial (fast) run to estimate best parameters automatically
                ## make input file with Json input for running bift
                f = open("inputfile.dat",'w')
                f.write('%s\n' % data)
                f.write('%f\n' % qmin)
                f.write('%f\n' % qmax)
                f.write('%s\n' % Bg)
                f.write('200\n') # nrebin set to 200
                f.write('%s\n' % dmax)
                f.write('\n')
                if alpha:
                    if alpha[0] == 'f':
                        f.write('%s\n' % alpha)
                    else:
                        f.write('f%s\n' % alpha)
                else:
                    f.write('f5\n') # fix alpha
                f.write('%s\n' % smear)
                f.write('\n')
                f.write('\n')
                if not prpoints:
                    f.write('50\n') # pr points set to 50
                else:
                    f.write('%s\n' % prpoints)
                f.write('\n') # 0 extra error calculations
                if transformation == 'A':
                    f.write('D\n') # use Debye transformations, if nothing is opted for
                else:
                    f.write('%s\n' % transformation)
                f.write('%s\n' % fitbackground)
                f.write('%s\n' % rescale_mode) # rescale method. N: non-constant, C: constant
                if rescale_mode == 'N':
                    f.write('%s\n' % nbin)
                else:
                    f.write('\n')
                f.write('\n')
                f.close()

                ## run bayesfit (fast run)
                f = open('stdout.dat','w')
                path = os.path.dirname(os.path.realpath(__file__))
                execute([path + '/source/bift','<','inputfile.dat'],f)
                f.close()
               
                ## get dmax initial guess from fast run
                if not dmax:
                    ## retrive dmax from parameter file
                    dmax_value = read_params(qmin,qmax)[1]
                    dmax = '%s' % dmax_value
                    d.udpmessage({"_textarea":"=================================================================================\n"})
                    d.udpmessage({"_textarea":"dmax %f\n" % dmax_value})
                    d.udpmessage({"_textarea":"=================================================================================\n"})

                ## set prpoints from dmax in fast run
                if not prpoints:
                    if dmax[0] == 'f':
                        dmax_aa = float(dmax[1:])
                    else:
                        dmax_aa = float(dmax)
                    
                    if units == 'nm':
                        dmax_aa *= 10 # convert dmax from nm to angstrom
                   
                    threshold_dmax_aa = 180
                    if dmax_aa > threshold_dmax_aa:
                        tmp = int(np.amin([dmax_aa/3,150])) # set prpoints to dmax/3, but maximum 150
                        prpoints = '%d' % tmp
                    else:
                        prpoints = '60' # min value of pr_points
                    d.udpmessage({"_textarea":"=================================================================================\n"})
                    d.udpmessage({"_textarea":"number of points in pr: %s\n" % prpoints})
                    d.udpmessage({"_textarea":"=================================================================================\n"})
                
                ## set transformation depending on negative points in pr in fast run
                if transformation == 'A':
                    r,pr,d_pr = np.genfromtxt('pr.dat',skip_header=0,usecols=[0,1,2],unpack=True)
                    threshold_pr = np.max(pr)*1E-4
                    contain_negative_entries = np.any(pr<-threshold_pr)
                    if contain_negative_entries:
                        transformation = 'N' # N for negative (allow negative values)
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"transformation: %s\n" % transformation})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                    else:
                        transformation = 'D'
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"transformation: %s\n" % transformation})
                        d.udpmessage({"_textarea":"=================================================================================\n"})

                ## increase noextracalc
                if not noextracalc:
                    noextracalc = '100'
                 
                ## determine skip_first
                if skip_first == '':
                    Rg_value = read_params(qmin,qmax)[2]
                    try:
                        q,I,dI = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True)
                    except:
                        q,I,dI = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True,encoding='cp855')
                    xx = q**2
                    yy = np.log(abs(I))
                    dyy = dI/abs(I)
                    M = len(q)
                    qmax_Guinier = 1.25/Rg_value
                    if qmax_Guinier > qmin:
                        last = M-np.where(q<qmax_Guinier)[0][-1]
                        skip_first_max = 25
                        n = np.min([len(q[:-last]),skip_first_max+2])
                        skip_first_array = range(0,n)
                        i = 0
                        chi2r_prev = 1e99
                        YELLOW_CARDS = 0
                        while i<n and YELLOW_CARDS<2:
                            s = skip_first_array[i]
                            x = xx[s:-last]
                            y = yy[s:-last]
                            dy = dyy[s:-last]
                            N = len(x)
                            degree = 2
                            c = np.polyfit(x,y,degree)
                            poly2_function = np.poly1d(c)
                            yfit = poly2_function(x)
                            R = (yfit-y)/dy
                            chi2r = np.sum(R**2)/(N-3)
                            chi2r_dif = (chi2r_prev - chi2r)/chi2r
                            if abs(R[0]) < 1.5 and chi2r_dif<0.3:
                                YELLOW_CARDS += 1
                            i +=1
                            chi2r_prev = chi2r
                        skip_first = np.max([s-2,0])
                        #check that slope is negative
                        a,b = np.polyfit(xx[skip_first:-last],yy[skip_first:-last],1)
                        if a>=0:
                            skip_first = 0
                        else:
                            Rg_calc = np.sqrt(-3*a)
                        #check qmaxRg is below 1.4
                        qmaxRg = qmax_Guinier*Rg_value
                        qmaxRg_calc = qmax_Guinier*Rg_calc
                        if qmaxRg > 1.4:
                            skip_first = 0
                        elif qmaxRg_calc > 1.4:
                            skip_first = 0
                    else:
                        skip_first = 0
                    d.udpmessage({"_textarea":"=================================================================================\n"})
                    d.udpmessage({"_textarea":"skip first points: %d\n" % skip_first})
                    d.udpmessage({"_textarea":"=================================================================================\n"})
                    if skip_first>5:
                        # convert skip_first to new qmin
                        try:
                            q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True)
                        except:
                            q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
                        if q_check[0] > qmin:
                            qmin = q_check[0]
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"qmin: %e\n" % qmin})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        # downscale prpoints (likely smaller dmax)
                        if float(prpoints) > 70:
                            prpoints = '70'
                            d.udpmessage({"_textarea":"=================================================================================\n"})
                            d.udpmessage({"_textarea":"number of points in pr: %s\n" % prpoints})
                            d.udpmessage({"_textarea":"=================================================================================\n"})
                        transformation = 'D'
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"transformation: %s\n" % transformation})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        try:
                            dmax = ''
                        except:
                            pass

            CONTINUE_Trans = True
            SECOND_TRY = False
            if fast_run:
                CONTINUE_Trans = False
            while CONTINUE_Trans:
                try:
                    ## make input file with Json input for running bift
                    f = open("inputfile.dat",'w')
                    f.write('%s\n' % data)
                    f.write('%f\n' % qmin)
                    f.write('%f\n' % qmax)
                    f.write('%s\n' % Bg)
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
                    f.write('%s\n' % rescale_mode) # rescale method. N: non-constant, C: constant, I: intensity-dependent
                    if rescale_mode == 'N':
                        f.write('%s\n' % nbin)
                    else:
                        f.write('\n')
                    f.write('\n')
                    f.close()

                    ## run bayesfit
                    os.remove('parameters.dat') # remove parameters file from initial fast run
                    f = open('stdout.dat','w')
                    path = os.path.dirname(os.path.realpath(__file__))
                    execute([path + '/source/bift','<','inputfile.dat'],f)
                    f.close()
            
                    ## import params data to check that bift was running ok (if not, algorithm will change transformation and try again)
                    dmax_value = read_params(qmin,qmax)[1] # if there is no parameters.dat file, this will give error
                    int(dmax_value) # if dmax_valule is nan, this will give error
                    dmax = '%f' % dmax_value

                    CONTINUE_Trans = False

                except:
                    if SECOND_TRY:
                        CONTINUE_Trans = False
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"Could not find a solution. Try chaning Maximum distance, Transformation, Alpha, Number of points in pr, or Skip first points\n"})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        exit()
                    elif transformation in ['N','M']:
                        transformation = 'D'
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"transformation: %s\n" % transformation})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        SECOND_TRY = True
                    elif transformation == 'D':
                        transformation = 'N'
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        d.udpmessage({"_textarea":"transformation: %s\n" % transformation})
                        d.udpmessage({"_textarea":"=================================================================================\n"})
                        SECOND_TRY = True
                    
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
            p *= len(R) # correction for multiple testing
            idx = np.where(p<0.01)
            Noutlier = len(idx[0])
            idx_max = np.argmax(abs(R))
            filename_outlier = 'outlier_filtered.dat'
            if Noutlier:
                with open(filename_outlier,'w') as f:
                    f.write('# data, with worst outlier filtered out\n')
                    for i in range(len(R)):
                        if i!=idx_max:
                            f.write('%e %e %e\n' % (qdat[i],Idat[i],sigma[i]))
        
            ## retrive output from parameter file
            I0,dmax_out,Rg,chi2r,background,alpha_out,Ng,Ns,evidence,Prob,Prob_str,assessment,beta,Run_max,Run_max_expect,dRun_max_expect,p_Run_max_str,NR,NR_expect,dNR_expect,p_NR,qmax_useful = read_params(qmin,qmax)
        
            if qmax_ite:
                qmax = qmax_useful
            else:
                CONTINUE_QMAX = 0

            count_ite_qmax += 1
            if count_ite_qmax > max_ite_qmax:
                CONTINUE_QMAX = 0

        ###########################
        # end of qmax while loop 
        ###########################

        if Noutlier > 1 and Noutlier < Noutlier_prev:
            Noutlier_prev = Noutlier
            n_pr = int(prpoints)
            dmax = '%f' % dmax_out # update dmax value
 #           alpha = '%f' % alpha_out # update logalpha value # not faster...
            if n_pr <= 190:
                n_pr += 50
                prpoints = '%d' % n_pr
                d.udpmessage({"_textarea":"=================================================================================\n"})
                d.udpmessage({"_textarea":"number of outliers: %d\n" % Noutlier})
                d.udpmessage({"_textarea":"trying to improve fit by increasing number of points in p(r)\n"})
                d.udpmessage({"_textarea":"number of points in p(r): %s\n" % prpoints})
                d.udpmessage({"_textarea":"=================================================================================\n"})
        elif outlier_ite and Noutlier:
            data = filename_outlier
            d.udpmessage({"_textarea":"=================================================================================\n"})
            d.udpmessage({"_textarea":"number of outliers: %d\n" % Noutlier})
            d.udpmessage({"_textarea":"removing worst oulier and rerunning\n"})
            #n_pr = int(prpoints)
            #if n_pr > 90:
            #    prpoints = '70'
            d.udpmessage({"_textarea":"number of points in p(r): %s\n" % prpoints})
            d.udpmessage({"_textarea":"=================================================================================\n"})
            CONTINUE_OUTLIER = Noutlier
            outliers_removed += 1
        else:
            CONTINUE_OUTLIER = 0

        count_ite += 1
        if count_ite >= max_ite:
            CONTINUE_OUTLIER = 0
            out_line = 'max iterations in outlier removal reached (=%d). prabably something wrong with error estimates in data' % max_ite
            d.udpmessage({"_textarea":out_line})


    ###########################
    # end of oulier while loop 
    ###########################
    
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

    ## general plotting settings
    markersize=4
    linewidth=1

    ## plot p(r)
    plt.plot(r,np.zeros(len(r)),linestyle='--',color='grey',zorder=0)
    plt.errorbar(r,pr,yerr=d_pr,marker='.',markersize=markersize,linewidth=linewidth,color='black',label='p(r)')
    if make_pr_bin:
        plt.errorbar(r_bin,pr_bin,d_pr_bin,marker='.',markersize=markersize,linewidth=linewidth,color='green',label='p(r), fixed binsize')
        plt.legend(frameon=False)
    plt.xlabel(r'$r$ [%s]' % units)
    plt.ylabel(r'$p(r)$')
    plt.title('p(r)')
    plt.tight_layout()
    plt.savefig('pr.png',dpi=200)
    plt.close()

    ## plot data, fit and residuals, not rescaled 
    TRUNCATE_ANALYSIS = 0
    if TRUNCATE_ANALYSIS:
        f,(p0,p1,p2) = plt.subplots(3,1,gridspec_kw={'height_ratios': [4,1,5]},sharex=True)
    else:
        f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
    p0.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label='data')
    if logx:
        p0.set_xscale('log')
        p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,label='p(r) fit')
    else:
        p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='p(r) fit') 
    p0.set_ylabel(r'$I(q)$')
    p0.set_yscale('log')
    p0.set_title('p(r) fit to data')

    p1.plot(qdat,R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
    if logx:
        p1.set_xscale('log')
        p1.plot(qdat,Idat*0,linewidth=linewidth,color='black',zorder=1)
        if Noutlier:
            p1.plot(qdat,-3*np.ones(len(Idat)),linewidth=linewidth,linestyle='--',color='grey',zorder=2,label=r'$\pm 3\sigma$')
            p1.plot(qdat,3*np.ones(len(Idat)),linewidth=linewidth,linestyle='--',color='grey',zorder=3)
    else:
        p1.plot(qfit,Ifit*0,linewidth=linewidth,color='black',zorder=1)
        if Noutlier:
            p1.plot(qfit,-3*np.ones(len(Ifit)),linewidth=linewidth,linestyle='--',color='grey',zorder=2,label=r'$\pm 3\sigma$')
            p1.plot(qfit,3*np.ones(len(Ifit)),linewidth=linewidth,linestyle='--',color='grey',zorder=3)
    if TRUNCATE_ANALYSIS:
        p2.plot(qdat,Idat/sigma,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
        p2.plot(qdat,Idat*0,linewidth=linewidth,color='black',zorder=1)
        p2.plot(qdat,Idat/Idat*2,linewidth=linewidth,linestyle='--',color='grey',zorder=1)
        p2.set_ylim(-5,10)
        #p2.set_yscale('log')
        p2.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
    else:
        p1.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        
    ## plot outliers
    if Noutlier>1:
        p0.plot(qdat[idx],Idat[idx],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='grey',zorder=4,label='potential outliers')
        p1.plot(qdat[idx],R[idx],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='grey',zorder=4)
        p0.plot(qdat[idx_max],Idat[idx_max],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='black',zorder=4,label='worst outlier')
        p1.plot(qdat[idx_max],R[idx_max],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='black',zorder=4)
    elif Noutlier == 1:
        p0.plot(qdat[idx_max],Idat[idx_max],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='black',zorder=4,label='potential outlier')
        p1.plot(qdat[idx_max],R[idx_max],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='black',zorder=4)

    p1.set_ylabel(r'$\Delta I(q)/\sigma$')
    try:
        p1.set_ylim(-maxR,maxR)
        if Noutlier:
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
    
    ## Guinier analysis
    if Guinier:
        try:
            qmaxRg = float(qmaxRg_in)
        except:
            qmaxRg = 1.25
            """
            #determine qmaxRg automatically
            idx_maxpr = np.argmax(pr)
            ratioRg = Rg/r[idx_maxpr]
            if ratioRg < 1.0:
                qmaxRg = 1.32
            elif ratioRg > 1.5:
                qmaxRg = 1.02
            else:
                qmaxRg = ((ratioRg-1.0)*1.32 + (1.5-ratioRg)*1.02)/(1.5-0.9)
            """
        Rg_Guinier = Rg
        try:
            Guinier_skip = int(Guinier_skip_in) 
        except:
            idx = np.where(qdat*Rg_Guinier<=qmaxRg)
            q2 = qdat[idx]**2
            lnI = np.log(Idat[idx])
            dlnI = sigma[idx]/Idat[idx]
            
            Guinier_skip = 0
            CONTINUE_GUINIER = True
            while CONTINUE_GUINIER:
                try:
                    a,b = np.polyfit(q2[Guinier_skip:],lnI[Guinier_skip:],1,w=1/dlnI[Guinier_skip:])
                    fit = b+a*q2[Guinier_skip:]
                    R = (lnI[Guinier_skip:]-fit)
                    Rmean = np.mean(abs(R))
                    if abs(R[0]) > Rmean*2:
                        Guinier_skip += 1
                    else:
                        CONTINUE_GUINIER = False
                except:
                    CONTINUE_GUINIER = False
        
        if qdat[Guinier_skip]*Rg<=qmaxRg:

            for i in range(7):
                idx = np.where(qdat*Rg_Guinier<=qmaxRg)
                q2 = qdat[idx]**2
                lnI = np.log(Idat[idx])
                dlnI = sigma[idx]/Idat[idx]
        
                n = len(idx[0])-Guinier_skip
                while (Guinier_skip > 0) and (n<10):
                    Guinier_skip = Guinier_skip-1
                    n = n+1
                try:
                    a,b = np.polyfit(q2[Guinier_skip:],lnI[Guinier_skip:],1,w=1/dlnI[Guinier_skip:])
                    fit = b+a*q2[Guinier_skip:]
                    Rg_Guinier = (Rg_Guinier + np.sqrt(-3*a))/2
                    Error_Guinier = False
                except:
                    Error_Guinier = True
            if Error_Guinier:
                error_message = '\nERROR in Guinier fit\n - do you have a defined Guinier region?\n - maybe try to skip some of the first points?\n - interparticle interactions may lead to a negative slope at low q\n - contrast match may lead to a negative slope at low q'
                d.udpmessage({"_textarea":error_message})
                f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
                p0.text(0.1,0.7,error_message,transform=p0.transAxes)
                plt.savefig('Guinier.png',dpi=200)
                plt.close()
            else:
                qmaxRg = np.sqrt(q2[-1])*Rg_Guinier
                R = (lnI[Guinier_skip:]-fit)/dlnI[Guinier_skip:]
                Rmax = np.ceil(np.amax(abs(R)))
                chi2r_Guinier = np.sum(R**2)/(n-2)

                f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
                p0.errorbar(q2,lnI,yerr=dlnI,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
                #p0.plot(q2[Guinier_skip:],fit,color='black',linewidth=linewidth,zorder=1,label='Guinier fit: $R_g$=%1.2f $(q_{max}R_g$=%1.2f, $\chi^2_r$=%1.1f, points skipped = %d)' % (Rg_Guinier,qmaxRg,chi2r_Guinier,Guinier_skip))
                p0.plot(q2[Guinier_skip:],fit,color='black',linewidth=linewidth,zorder=1,label='$R_g$=%1.2f, $q_{max}R_g$=%1.2f, $\chi^2_r$=%1.1f, skipped_points=%d' % (Rg_Guinier,qmaxRg,chi2r_Guinier,Guinier_skip))
                p1.plot(q2[Guinier_skip:],R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
                p1.plot(q2,q2-q2,color='black',linewidth=linewidth,zorder=1)
                p0.set_ylabel(r'$ln(I)$')
                p1.set_xlabel(r'$q^2$ [%s$^{-2}$]' % units)
                p1.set_ylabel(r'$\Delta lnI/\sigma_{lnI}$')
                p1.set_ylim([-Rmax,Rmax])
                p1.set_yticks([-Rmax,0,Rmax])
                p0.set_title('Guinier plot')
                p0.legend(frameon=False)
                plt.savefig('Guinier.png',dpi=200)
                plt.close()
        else:
            """
            Rg_Guinier = 0
            idx = np.where(qdat<0.05)
            q2 = qdat[idx]**2
            lnI = np.log(Idat[idx])
            dlnI = sigma[idx]/Idat[idx]
        
            plt.errorbar(q2,lnI,yerr=dlnI,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
            plt.xlabel(r'$q^2$ [%s$^{-2}$]' % units)
            plt.ylabel(r'$ln(I)$')
            plt.title('Guinier plot')
            plt.legend(frameon=False)
            plt.savefig('Guinier.png',dpi=200)
            plt.close()
            """
            Rg_Guinier = 0
            error_message = '\nERROR in Guinier fit\n - do you have a defined Guinier region?\n - maybe you skipped too many points?\n - maybe your sample is large (>hundreds of nm)?'
            d.udpmessage({"_textarea":error_message})
            f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
            p0.text(0.1,0.7,error_message,transform=p0.transAxes)
            plt.savefig('Guinier.png',dpi=200)
            plt.close()

    ## Kratky
    if Kratky:
        y,y0 = Idat-background,I0-background

        qRg = qdat*Rg

        if Kratky_dim:
            x = qRg
            xxI = x*x*y/y0
            dxxI = x*x*sigma/y0
        else:
            x = qdat
            xxI = x*x*y
            dxxI = x*x*sigma

        if Kratky_Mw:
            if units == 'nm':
                qdat_aa = qdat*0.1
                Rg_aa = Rg*10
            else:
                qdat_aa = qdat
                Rg_aa = Rg
            qm = np.amin([8.0/Rg_aa,np.amax(qdat_aa)])
            relative_uncertainty = np.max([Rg_aa/300,0.1]) # Ficher et al 2010 J. Appl. Cryst. (2010). 43, 101-109

            idx = np.where(qRg <= 8.0)
            yy = qdat_aa**2*y
            dq_aa = (np.amax(qdat_aa[idx])-np.amin(qdat_aa[idx]))/len(idx[0])
            Qt = np.sum(yy[idx])*dq_aa
            Vt = 2*np.pi**2*y0/Qt
            
            #MwP = 0.625/1000 * Vt # Petoukhov et al 2012, 0.625 kDa/nm3 -> 0.625/1000 kDa/A3
           
            # Piiadov et al 2018 Protein Science  https://doi.org/10.1002/pro.3528
            qm2,qm3,qm4 = qm**2,qm**3,qm**4
            A = -2.114e6*qm**4 + 2.920e6*qm3 - 1.472e6*qm2 + 3.349e5*qm - 3.577e4
            B =                  12.09*qm3   - 9.39*qm2    + 3.03*qm    + 0.29
            Vm = A+B*Vt # A
           
            MwF = 0.83/1000 * Vm # Squire and Himmel 1979, 0.83 kDa/nm3 --> 0.83/1000 kDa/A3
            dMwF = MwF * relative_uncertainty
            #label = 'Mw = %1.1f kDa (+/-10%s)' % (MwF,'%')
            label = 'Mw = %1.1f+/- %1.1fkDa' % (MwF,dMwF)
            
        else:
            label = ''

        plt.errorbar(x,xxI,yerr=dxxI,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label=label)
        plt.plot(x,np.zeros(len(x)),linestyle='--',color='grey',zorder=1)
        plt.legend(frameon=False)
        plt.title('Kratky plot')

        if Kratky_dim:
            plt.xlabel(r'$q R_g$')
            plt.ylabel(r'$I/I(0) (q R_G)^2$')
        else:
            plt.xlabel(r'$q$ [%s$^{-1}$]' % units)
            plt.ylabel(r'$I q^2$')
        plt.tight_layout()
        plt.savefig('Kratky.png',dpi=200)
        plt.close()
    
    ## Porod
    if Porod:
        y = qdat**4 * (Idat - background)
        dy = qdat**4 * sigma
        if Porod_limit:
            qm_Porod = Porod_limit
        else:
            qm_Porod = qmax_useful*0.95 #np.pi*Ng/dmax
        if np.amax(qdat) <= qm_Porod:
            qm_Porod = 0.9*np.amax(qdat)
        idx = np.where(qdat>qm_Porod)
        a = np.polyfit(qdat[idx],y[idx],0,w=1/dy[idx])
        #R = (y-a)/dy
        #slope,bb = np.polyfit(qdat[idx],y[idx],1,w=1/dy[idx])
        #if abs(slope)<2e-5:
        #    recommend = 'background subtraction fine'
        #elif slope>0:
        #    recommend = 'recommend: subtract constant from data'
        #elif slope<1:
        #    recommend = 'recommend: add constant to data'
        
        #f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
        f,p0 = plt.subplots(1,1)
        p0.errorbar(qdat,y,yerr=dy,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
        p0.plot([qm_Porod,qm_Porod],[np.amin(y),np.amax(y)],color='grey',linestyle='--')
       
        p0.plot(qdat[idx],qdat[idx]/qdat[idx]*a,color='black',label='fit with constant')
        #p0.plot(qdat[idx],qdat[idx]/qdat[idx]*a,color='black',label='const fit, data slope: %1.0e, %s' % (slope,recommend))
        p0.set_title('Porod plot')
        p0.set_ylabel(r'$I q^4$')
        #p1.plot(qdat,R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
        #p1.plot(qdat[idx],R[idx]*a,color='black')
        #Rmax = np.ceil(np.amax(abs(R[idx])))
        #p1.set_ylim(-Rmax,Rmax)
        #if Rmax > 3:
        #    p1.plot(qdat,R/R*3,color='grey',linestyle='--')
        #    p1.plot(qdat,-R/R*3,color='grey',linestyle='--')
        #else:
        #    p1.set_yticks([-Rmax,0,Rmax])
        #p1.plot([qm_Porod,qm_Porod],[-Rmax,Rmax],color='grey',linestyle='--')
        #p1.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        p0.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        #p1.set_ylabel(r'$\Delta Iq^4$/$\sigma_{Iq^4}$')

        p0.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('Porod.png',dpi=200)
        plt.close()

    ## import and plot data with rescaled errors
    if Prob < 0.003:
        qresc,Iresc,sigmaresc = np.genfromtxt('rescale.dat',skip_header=2,usecols=[0,1,2],unpack=True)
        offset = 10
        plt.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',label='data',zorder=0)
        plt.errorbar(qresc,Iresc*offset,yerr=sigmaresc*offset,linestyle='none',marker='.',markersize=markersize,color='blue',label='data with rescaled errors, offset x10',zorder=1) 
        plt.ylabel(r'$I(q)$')
        plt.yscale('log')
        plt.xlabel(r'$q$ [%s$^{-1}$]' % units)
        if rescale_mode == 'N':
            plt.title('input data and data with q-dependent rescaling of errors')
        elif rescale_mode == 'I':
            plt.title('input data and data with I-dependent rescaling of errors')
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
            p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,zorder=1,label='p(r) fit')
        else:
            p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='p(r) fit')
        p0.set_ylabel(r'$I(q)$')
        p0.set_yscale('log')
        p0.set_title('p(r) fit to data with rescaled errors')
        p0.legend(frameon=False)

        p1.plot(qdat,R_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0)
        if logx:
            p1.set_xscale('log')
            p1.plot(qdat,Idat*0,linewidth=linewidth,color='black',zorder=1)
        else:
            p1.plot(qfit,Ifit*0,linewidth=linewidth,color='black',zorder=1)
        p1.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        p1.set_ylabel(r'$\Delta I(q)/\sigma_\mathrm{rescale}$')
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
    os.system('zip results_%s.zip pr.dat pr_bin.dat data.dat fit.dat fit_q.dat parameters.dat rescale.dat outlier_filtered.dat scale_factor.dat stdout.dat bift.f inputfile.dat *.png' % prefix)

    ## generate output
    output = {} # create an empty python dictionary
    
    # files
    if make_pr_bin:
        output["pr"] = "%s/pr_bin.dat" % folder
    else:
        output["pr"] = "%s/pr.dat" % folder
    output["dataused"] = "%s/data.dat" % folder
    #if outlier_ite: 
    if Noutlier: 
        output["outlier_filtered"] = "%s/outlier_filtered.dat" % folder
    output["fitofdata"] = "%s/fit_q.dat" % folder
    output["parameters"] = "%s/parameters.dat" % folder
    output["file_stdout"] = "%s/stdout.dat" % folder
    output["sourcecode"] = "%s/bift.f" % folder
    output["inputfile"] = "%s/inputfile.dat" % folder
    output["prfig"] = "%s/pr.png" % folder
    output["iqfig"] = "%s/Iq.png" % folder
    output["guinierfig"] = "%s/Guinier.png" % folder
    output["kratkyfig"] = "%s/Kratky.png" % folder
    output["porodfig"] = "%s/Porod.png" % folder
    if Prob<0.003:
        if abs(1-beta)>=0.1:
            output["rescaled"] = "%s/rescale.dat" % folder
            output["scale_factor"] = "%s/scale_factor.dat" % folder
            output["rescalefig"] = "%s/rescale.png" % folder
            output["iqrsfig"] = "%s/Iq_rs.png" % folder
    output["zip"] = "%s/results_%s.zip" % (folder,prefix)

    # values
    output["dmaxout"] = "%1.2f" % dmax_out
    output["Rg"] = "%1.2f" % Rg
    if Guinier:
        if Rg_Guinier == 0:
            output["Rg_Guinier"] = "No Guinier region"
        else:
            output["Rg_Guinier"] = "%1.2f" % Rg_Guinier
    
    output["I0"] = "%1.2e" % I0
    output["background"] = "%1.1e" % background
    
    output["chi2"] = "%1.2f" % chi2r
    output["prob"] = "%s" % Prob_str
    output["assess"] = "%s" % assessment
    if Prob>=0.003:
        output["beta"] = "No correction"
    elif rescale_mode == 'C':
        if abs(1-beta)>=0.1:
            output["beta"] = "%1.2f" % beta 
        else:
            output["beta"] = "No correction"
    else:
        output["beta"] = "see scale_factor.dat"

    output["Rmax"] = "%1.1f" % Run_max
    output["Rmax_expect"] = "%1.1f +/- %1.1f" % (Run_max_expect,dRun_max_expect)
    output["p_Rmax"] = "%s" % p_Run_max_str
    output["NR"] = "%1.1f" % NR
    output["NR_expect"] = "%1.1f +/- %1.1f" % (NR_expect,dNR_expect)
    if p_NR < 0.001:
        output["p_NR"] = "%1.2e" % p_NR
    else:
        output["p_NR"] = "%1.3f" % p_NR
    output["Noutlier"] = "%d" % Noutlier
    output["outliers_removed"] = "%d" % outliers_removed
    output["skip_first_out"] = "%d" % skip_first
    output["Ng"] = "%1.2f" % Ng
    output["shannon"] = "%1.2f" % Ns
    output["logalpha"] = "%1.2f" % alpha_out 
    output["evidence"] = "%1.2f" % evidence
    output["prpoints_out"] = prpoints
#    output["axratio_pro"] = "%1.2f" % ax_pro
#    output["axratio_obl"] = "%1.2f" % ax_obl
    output["qmax_useful"] = "%1.2f" % qmax_useful
    #output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    #output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output

