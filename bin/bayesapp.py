#!/usr/bin/python3

import sys
import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import time
import argparse
from sys import argv
from bayesapp_helpfunctions import *

if __name__=='__main__':

    ### start timing
    start_time = time.time()

    ## current version
    version = 2.1

    ### welcome message
    printt('#######################################################################################')
    printt('RUNNING bayesapp.py version %s \n - for instructions type: python bayesapp.py -h' % version)
    command = "python bayesapp.py"
    for aa in argv[1:]:
        if os.path.sep in aa:
            display_aa = os.path.basename(aa) 
        else:
            display_aa = aa
        if ' ' in display_aa:
            command += " \"%s\"" % display_aa
        else:
            command += " %s" % display_aa
    printt('command used: %s' % command)
    printt('#######################################################################################')

    ### input values 
    parser = argparse.ArgumentParser(description='BayesApp - calculates the pair distance distribution from SAXS or SANS data.')
      
    # mandatory inputs
    parser.add_argument('-f', '--data_file', required=True, help='path to data')

    # optional general inputs
    parser.add_argument('-qmin', '--qmin', default='0', help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', default='0.7', help='Maximum q-value for the scattering curve.')
    parser.add_argument('-units', '--units', default='auto', help='units of data: A or nm')

    parser.add_argument('-G', '--Guinier', action='store_true', default=False, help='Guinier plot')
    parser.add_argument('-qmaxRg', '--Guinier_qmaxRg', default='0.7', help='qmax*Rg in Guinier analysis')
    parser.add_argument('-G_skip', '--Guinier_skip', default='0', help='skip first points in Guinier analysis')

    parser.add_argument('-K', '--Kratky', action='store_true', default=False, help='Kratky plot')
    parser.add_argument('-K_dim', '--Kratky_dim', action='store_true', default=False, help='Dimensionless Kratky plot')
    parser.add_argument('-Mw', '--Kratky_Mw', action='store_true', default=False, help='Mw from Kratky plot')

    parser.add_argument('-P', '--Porod', action='store_true', default=False, help='Porod plot')
    parser.add_argument('-P_lim', '--Porod_limit', default='0.0', help='Porod limit start')

    parser.add_argument('-d', '--dmax', default='', help='dmax (add f in front to fix, e.g. f100)')
    parser.add_argument('-t', '--transformation', default='A', help='transformation')
    parser.add_argument('-b', '--nrebin', default='500', help='rebin to approximately this number of points')
    parser.add_argument('-a', '--alpha', default='', help='alpha (add f in front to fix, e.g. f100)')
    parser.add_argument('-s', '--smear', default='', help='smear')
    parser.add_argument('-n', '--prpoints', default='', help='prpoints')
    parser.add_argument('-c', '--noextracalc', default='100', help='number of extra calculations')
    parser.add_argument('-rescale_mode', '--rescale_mode', default='C', help='error rescale mode, if wrong')
    parser.add_argument('-nbin', '--nbin', default=' ', help='nbin')
    parser.add_argument('-Bg', '--Bg', default='0', help='Bg')

    parser.add_argument('-fitbackground', '--fitbackground', default='Y', help='fitbackground: Y or N, default is Y')
    parser.add_argument('-logx', '--logx', action='store_true', default=False, help='logarithmic log axis')
    parser.add_argument('-make_pr_bin', '--make_pr_bin', action='store_true', default=False, help='make_pr_bin')
    parser.add_argument('-pr_binsize', '--pr_binsize', default='1', help='pr_binsize')
    parser.add_argument('-skip_first', '--skip_first', default='', help='skip_first')
    parser.add_argument('-outlier_ite', '--outlier_ite', action='store_true', default=False, help='outlier_ite')
    parser.add_argument('-fast', '--fast_run', action='store_true', default=False, help='fast_run')
    
    parser.add_argument('-show', '--show', action='store_true', default=False, help='show plots') 
    parser.add_argument('-z', '--zip_compress', action='store_true', default=False, help='compress output files in zip file')

    args = parser.parse_args()

    #################################################################################
    ## adjust input
    #################################################################################
    
    ## remove spaces and brackets from name
    # data_file = args.data_file
    prefix = os.path.basename(args.data_file)
    # prefix = data_file.split('/')[-1] 
    prefix = prefix.replace(" ","_").replace("(","").replace(")","").replace("[","").replace("]","")
    
    ## naming bug fix: fortran77 cannot use long file names
    if len(prefix)>48:
        data = 'data_name_too_long_for_fortran77.dat'
    else:
        data = prefix

    ## remove path from filename (for output)
    filename = os.path.basename(args.data_file)
    
    ## import skip first, qmin and qmax
    try:
        skip_first = int(args.skip_first) # skip first points
    except:
        skip_first = ''
    try:
        qmax = float(args.qmax)
    except:
        qmax = 100.0
    try:
        qmin = float(args.qmin)
    except:
        qmin = 0.0
    header,footer = get_header_footer(args.data_file)
    try:
        if skip_first:
            q_check = np.genfromtxt(args.data_file,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True)
        else:
            q_check = np.genfromtxt(args.data_file,skip_header=header,skip_footer=footer,usecols=[0],unpack=True)
    except:
        if skip_first:
            q_check = np.genfromtxt(args.data_file,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
        else:
            q_check = np.genfromtxt(args.data_file,skip_header=header,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
    if q_check[0] > qmin:
        qmin = q_check[0]
    if q_check[-1] < qmax:
        qmax = q_check[-1]

    ## automatically detect units
    if args.units == 'auto':
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
        printt('\n\n!!!ERROR!!!\nqmin should be smaller than qmax.\n')
        sys.exit()

    ## rename and convert input
    dmax = args.dmax
    transformation = args.transformation
    pr_binsize = float(args.pr_binsize)
    prpoints = args.prpoints
    qmaxRg_in = float(args.Guinier_qmaxRg)
    Guinier_skip_in = int(args.Guinier_skip)
    Porod_limit = float(args.Porod_limit)
    
    ## print start bayesapp message
    printt("=================================================================================")
    printt('    Reading data:                   %s' % filename)
    printt('        header lines in datafile:   %d' % header)
    printt('        footer lines in datafile:   %d' % footer)
    printt('        q min:                      %f' % q_check[0])
    printt('        q min used                  %f' % qmin)
    printt('        q max:                      %f' % q_check[-1])
    printt('        q max used                  %f' % qmax)
    printt('        number of points in data:   %d' % len(q_check))
    printt('        data rebinned to around:    %s points' % args.nrebin)
    printt('        units (of q) assumed to be: 1/%s' % units)
    printt("=================================================================================")

    ##################################
    # beginning of outlier while loop 
    ##################################
    CONTINUE_OUTLIER = True
    count_ite,max_ite,Noutlier_prev,outliers_removed = 0,20,1e3,0
    while CONTINUE_OUTLIER:
        count_auto = 0
        #############################################################
        # beginning of auto dmax/transformation/skip_first while loop
        #############################################################
        while (dmax == '' or transformation == 'A' or skip_first == '') and count_auto < 3:
            f = open("inputfile.dat",'w')
            f.write('%s\n' % data)
            f.write('%f\n' % qmin)
            f.write('%f\n' % qmax)
            f.write('%s\n' % args.Bg)
            f.write('200\n') # nrebin set to 200
            f.write('%s\n' % dmax)
            f.write('\n')
            if args.alpha:
                if args.alpha[0] == 'f':
                    f.write('f%s\n' % args.alpha)
                else:
                    f.write('%s\n' % args.alpha)
            else:
                f.write('f5\n') # fix alpha
            f.write('%s\n' % args.smear)
            f.write('\n')
            f.write('\n')
            if prpoints == '':
                f.write('70\n') #set pr points in fast run
            else:
                f.write('%s\n' % prpoints)
            f.write('\n') # 0 extra error calculations
            if transformation == 'A':
                f.write('D\n') # use Debye transformations, if nothing is opted for
            else:
                f.write('%s\n' % transformation)
            f.write('%s\n' % args.fitbackground)
            f.write('%s\n' % args.rescale_mode) # rescale method. N: non-constant, C: constant
            if args.rescale_mode == 'N':
                f.write('%s\n' % args.nbin)
            else:
                f.write('\n')
            f.write('\n')
            f.close()
            
            ## ensure bift and data is at the current location
            if not (os.path.exists(filename) and os.path.samefile(args.data_file, filename)):
                shutil.copy2(args.data_file, '.') # copy data file to current location
            path = os.path.dirname(os.path.realpath(__file__))
            exe = 'bift.exe' if os.name == 'nt' else 'bift'
            if not (os.path.exists(exe) and os.path.samefile(os.path.join(path, exe), exe)):
                shutil.copy2(os.path.join(path, exe), '.') # copy bift executable to current location
            if not (os.path.exists('bift.f') and os.path.samefile(os.path.join(path, 'bift.f'), 'bift.f')):
                shutil.copy2(os.path.join(path, 'bift.f'), '.') # copy fortran code to current location

            ## run bayesfit (fast run)
            printt("=================================================================================")
            printt("    Fast run for input parameter estimation")
            printt("=================================================================================")
            with open('inputfile.dat', 'r') as input_file:
                #result = subprocess.run([exe], stdin=input_file)
                result = subprocess.run([os.path.join(os.getcwd(), exe)], stdin=input_file)

            ## estimate dmax from fast run
            if dmax == '':
                ## retrive dmax from parameter file
                dmax_value = read_params()[1]
                dmax = '%s' % dmax_value

            ## set transformation depending on negative points in pr in fast run
            if transformation == 'A':
                r,pr,d_pr = np.genfromtxt('pr.dat',skip_header=0,usecols=[0,1,2],unpack=True)
                threshold_pr = np.max(pr)*1E-5
                contain_negative_entries = np.any(pr<-threshold_pr)
                if contain_negative_entries:
                    transformation = 'N' # N for negative (allow negative values)
                    if dmax[0] == 'f':
                        pass
                    else:
                        dmax = args.dmax # reset dmax
                        if dmax == '':
                            printt("\n\n\nChanged transformation - new fast run with this tranformation\n\n\n")
                else:
                    transformation = 'D'

            ## set prpoints from dmax in fast run
            if (prpoints == '' and dmax): 
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
                    prpoints = '60' # min value of prpoints

            ## determine skip_first
            if (skip_first == '' and dmax):
                Rg_value = read_params()[2]
                try:
                    q,I,dI = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True)
                except:
                    q,I,dI = np.genfromtxt(data,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True,encoding='cp855')
                ## remove entries with zero for the error
                idx_nonzero = np.where(dI!=0)
                q,I,dI = q[idx_nonzero],I[idx_nonzero],dI[idx_nonzero]

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
                    try:
                        skip_first = np.max([s-2,0])
                    except:
                        skip_first = 0
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
                if skip_first>5:
                    # convert skip_first to new qmin
                    try:
                        q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True)
                    except:
                        q_check = np.genfromtxt(data,skip_header=header+skip_first,skip_footer=footer,usecols=[0],unpack=True,encoding='cp855')
                    if q_check[0] > qmin:
                        qmin = q_check[0]
                    # downscale prpoints (likely smaller dmax)
                    if prpoints: 
                        if float(prpoints) > 70:
                            prpoints = '70'
                    # find new dmax
                    if dmax:
                        if dmax[0] == 'f':
                            pass
                        else:
                            dmax = '' # reset dmax
        printt("\n\n\n")
        printt("    Estimated input parameters from fast run:")
        printt("        dmax:                           %s" % dmax)
        printt("        transformation:                 %s" % transformation)
        printt("        number of first points skipped: %d" % skip_first)
        printt("        qmin:                           %e" % qmin)
        printt("        number of points in p(r):       %s" % prpoints)
        printt("\n\n\n")

        ###############################################################
        ## make actual run with determined (or manually set) values
        ###############################################################
        # if it fails, change the transformation and try again (SECOND TRY), if if fails again, then exit

        CONTINUE_Trans = True
        SECOND_TRY = False
        while CONTINUE_Trans and not args.fast_run:
            try:
                ## make input file for running bift
                f = open("inputfile.dat",'w')
                f.write('%s\n' % data)
                f.write('%f\n' % qmin)
                f.write('%f\n' % qmax)
                f.write('%s\n' % args.Bg)
                f.write('%s\n' % args.nrebin)
                f.write('%s\n' % dmax)
                f.write('\n')
                f.write('%s\n' % args.alpha)
                f.write('%s\n' % args.smear)
                f.write('\n')
                f.write('\n')
                f.write('%s\n' % prpoints)
                f.write('%s\n' % args.noextracalc)
                f.write('%s\n' % transformation)
                f.write('%s\n' % args.fitbackground)
                f.write('%s\n' % args.rescale_mode) # rescale method. N: non-constant, C: constant, I: intensity-dependent
                if args.rescale_mode == 'N':
                    f.write('%s\n' % args.nbin)
                else:
                    f.write('\n')
                f.write('\n')
                f.close()

                ## run bift
                os.remove('parameters.dat') # remove parameters file from initial fast run
                printt("=================================================================================")
                printt("    Running BayesApp with estimated input parameters")
                printt("=================================================================================") 
                with open('inputfile.dat', 'r') as input_file:
                    result = subprocess.run([os.path.join(os.getcwd(), exe)], stdin=input_file)
                    
                ## import params data to check that bift was running ok (if not, algorithm will change transformation and try again)
                dmax_value = read_params()[1] # if there is no parameters.dat file, this will give error
                int(dmax_value) # if dmax_valule is nan, this will give error
                if dmax[0] == 'f':
                    dmax = 'f%f' % dmax_value # set dmax
                else:
                    dmax = '%f' % dmax_value # set dmax
                CONTINUE_Trans = False
            except:
                if SECOND_TRY:
                    CONTINUE_Trans = False
                    printt("=================================================================================")
                    printt("    Could not find a solution. Try chaning Maximum distance, Transformation, alpha, Number of points in pr, or Skip first points")
                    printt("=================================================================================")
                    exit()
                elif transformation in ['N','M']:
                    transformation = 'D'
                    printt("=================================================================================")
                    printt("    No solution - running again with new transformation: %s" % transformation)
                    printt("=================================================================================")
                    SECOND_TRY = True
                elif transformation == 'D':
                    transformation = 'N'
                    printt("=================================================================================")
                    printt("    No solution - running again with new transformation: %s" % transformation)
                    printt("=================================================================================")
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
                for i in range(len(R))          :
                    if i!=idx_max:
                        f.write('%e %e %e\n' % (qdat[i],Idat[i],sigma[i]))
    
        ## retrive output from parameter file
        I0,dmax_out,Rg,chi2r,background,alpha_out,Ng,Ns,evidence,Prob,Prob_str,assessment,beta,Run_max,Run_max_expect,dRun_max_expect,p_Run_max_str,NR,NR_expect,dNR_expect,p_NR,prpoints_float = read_params()

        ## if there are many outliers, then try to gradually (in steps of 50) increase number of points in p(r) and rerun (until prpoints is above 180)
        if Noutlier > 1 and Noutlier < int(Noutlier_prev*0.8) and not args.fast_run:
            Noutlier_prev = Noutlier
            if dmax[0] == 'f':
                dmax = 'f%f' % dmax_out
            else:
                dmax = '%f' % dmax_out # update dmax value
            if prpoints_float < 190:
                printt("=================================================================================")
                printt("    number of outliers: %d" % Noutlier)
                printt("        trying to improve fit by increasing number of points in p(r)")
                printt("        old number of points in p(r): %.0f" % prpoints_float)
                prpoints_float += 50
                printt("        new number of points in p(r): %.0f" % prpoints_float)
                prpoints = '%.0f' % prpoints_float
            else:
                CONTINUE_OUTLIER = False
            printt("=================================================================================")
        # remove worst outlier and run again
        elif args.outlier_ite and Noutlier:
            data = filename_outlier
            printt("=================================================================================")
            printt("number of outliers: %d" % Noutlier)
            printt("removing worst oulier and rerunning")
            printt("=================================================================================")
            CONTINUE_OUTLIER = Noutlier
            outliers_removed += 1
        else:
            CONTINUE_OUTLIER = False
        count_ite += 1
        if count_ite >= max_ite:
            CONTINUE_OUTLIER = False
            printt('max iterations in outlier removal reached (=%d). prabably something wrong with error estimates in data' % max_ite)

    ###########################
    # end of oulier while loop 
    ###########################
    
    ## import p(r)
    r,pr,d_pr = np.genfromtxt('pr.dat',skip_header=0,usecols=[0,1,2],unpack=True)

    if args.make_pr_bin:
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

    ## extend pr on denss r grid to get a continous function
    if pr[1] > 0:
        r_new,pr_new,d_pr_new = np.insert(r, 1, r[1]*0.1), np.insert(pr, 1, pr[1]*0.08), np.insert(d_pr, 1, d_pr[1]) # add virtual point to enforce positive slope at r=0
    else:
        r_new,pr_new,d_pr_new = r,pr,d_pr
    r_new,pr_new,d_pr_new = np.insert(r_new, 0, -2), np.insert(pr_new, 0, 0.0), np.insert(d_pr_new, 0, d_pr_new[1]) # add virtual point at minus to avoid boundary oscillation
    try:
        d_pr_new[d_pr_new == 0] = np.sort(np.unique(d_pr_new))[1]*0.01  # avoid division by zero
        w_pr = 1.0 / d_pr_new # define weights 
    except:
        w_pr = np.ones_like(pr_new) # define weights
    spline = UnivariateSpline(r_new, pr_new, w=w_pr, s=0.05 * len(r_new)) # smoothing spline and interpolation on dense r grid
    # spline2 = UnivariateSpline(r_new, pr_new, w=w_pr, s=0.1 * len(r_new)) # smoothing spline and interpolation on dense r grid
    # spline3 = UnivariateSpline(r_new, pr_new, w=w_pr, s=0.2 * len(r_new)) # smoothing spline and interpolation on dense r grid
    # spline4 = UnivariateSpline(r_new, pr_new, w=w_pr, s=0.5 * len(r_new)) # smoothing spline and interpolation on dense r grid
    r_dense = np.linspace(0, r_new.max(), len(r_new)*20)
    pr_dense = spline(r_dense)
    with open('pr_smooth.dat','w') as f:
        for x,y in zip(r_dense,pr_dense):
            f.write('%10.10f %10.10e\n' % (x,y))

    ## general plotting settings
    markersize = 4
    linewidth = 1

    ## plot p(r)
    plt.plot(r,np.zeros(len(r)),linestyle='--',color='grey',zorder=0)
    # plt.errorbar(r,pr,yerr=d_pr,marker='.',markersize=markersize,linestyle='None',color='black')
    plt.plot(r,pr,marker='.',markersize=markersize,linestyle='None',color='black')
    if args.make_pr_bin:
        plt.errorbar(r_bin,pr_bin,d_pr_bin,marker='.',markersize=markersize,linewidth=linewidth,color='green',label='p(r), fixed binsize')
        plt.legend(frameon=False)
    plt.plot(r_dense,pr_dense,linewidth=linewidth,color='black')
    # plt.plot(r_dense,spline2(r_dense),linewidth=linewidth,color='red',label='p(r) 0.1')
    # plt.plot(r_dense,spline3(r_dense),linewidth=linewidth,color='green',label='p(r) 0.2')
    # plt.plot(r_dense,spline4(r_dense),linewidth=linewidth,color='blue',label='p(r) 0.5')
    plt.xlabel(r'$r$ [%s]' % units)
    plt.ylabel(r'$p(r)$')
    plt.title('p(r)')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('pr.png',dpi=200)

    ## plot data, fit and residuals, not rescaled 
    TRUNCATE_ANALYSIS = 0
    if TRUNCATE_ANALYSIS:
        f,(p0,p1,p2) = plt.subplots(3,1,gridspec_kw={'height_ratios': [4,1,5]},sharex=True)
    else:
        f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
    p0.errorbar(qdat,Idat,yerr=sigma,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label='data')
    if args.logx:
        p0.set_xscale('log')
        p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,label='p(r) fit')
    else:
        p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='p(r) fit') 
    p0.set_ylabel(r'$I(q)$')
    p0.set_yscale('log')
    p0.set_title('p(r) fit to data')

    p1.plot(qdat,R,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
    if args.logx:
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
        printt("    WARNING: Some residuals are either NaN or inf - bad fit?")
        printt("         probably just a numerical instability")
        printt("         try changing the number of points in p(r)")
    p0.legend(frameon=False)
    plt.savefig('Iq.png',dpi=200)
    plt.tight_layout()
    
    ## Guinier analysis
    if args.Guinier:
        try:
            qmaxRg = float(qmaxRg_in)
        except:
            qmaxRg = 1.25
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
                printt('\nERROR in Guinier fit\n - do you have a defined Guinier region?\n - maybe try to skip some of the first points?\n - interparticle interactions may lead to a negative slope at low q\n - contrast match may lead to a negative slope at low q')
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
        else:
            Rg_Guinier = 0
            error_message = '\nERROR in Guinier fit\n - do you have a defined Guinier region?\n - maybe you skipped too many points?\n - maybe your sample is large (>hundreds of nm)?'
            printt(error_message)
            f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)
            p0.text(0.1,0.7,error_message,transform=p0.transAxes)
            plt.savefig('Guinier.png',dpi=200)

    ## Kratky
    if args.Kratky or args.Kratky_dim:

        # subtract constant from data
        I_sub,I0_sub = Idat-background,I0-background
        
        # calculate Mw from integrating the Kratky plot
        qRg = qdat*Rg
        if args.Kratky_Mw:
            if units == 'nm':
                qdat_aa = qdat*0.1
                Rg_aa = Rg*10
            else:
                qdat_aa = qdat
                Rg_aa = Rg
            qm = np.amin([8.0/Rg_aa,np.amax(qdat_aa)])
            relative_uncertainty = np.max([Rg_aa/300,0.1]) # Ficher et al 2010 J. Appl. Cryst. (2010). 43, 101-109
            idx = np.where(qRg <= 8.0)
            yy = qdat_aa**2*I_sub
            dq_aa = (np.amax(qdat_aa[idx])-np.amin(qdat_aa[idx]))/len(idx[0])
            Qt = np.sum(yy[idx])*dq_aa # scattering invariant
            Vt = 2*np.pi**2*I0_sub/Qt
            #MwP = 0.625/1000 * Vt # Petoukhov et al 2012, 0.625 kDa/nm3 -> 0.625/1000 kDa/A3
           
            # Piiadov et al 2018 Protein Science  https://doi.org/10.1002/pro.3528
            qm2,qm3,qm4 = qm**2,qm**3,qm**4
            A = -2.114e6*qm**4 + 2.920e6*qm3 - 1.472e6*qm2 + 3.349e5*qm - 3.577e4
            B =                  12.09*qm3   - 9.39*qm2    + 3.03*qm    + 0.29
            Vm = A+B*Vt # A
            MwF = 0.83/1000 * Vm # Squire and Himmel 1979, 0.83 kDa/nm3 --> 0.83/1000 kDa/A3
            dMwF = MwF * relative_uncertainty
            label = 'Mw = %1.1f +/- %1.1fkDa' % (MwF,dMwF)
        else:
            label = ''

        # Kratky plot
        plt.figure()
        if args.Kratky_dim:
            # dimensionless Kratky plot
            x, y, dy = qRg, qRg**2*I_sub/I0_sub, qRg*qRg*sigma/I0_sub
            plt.ylabel(r'$I/I(0) (q R_G)^2$')
            plt.xlabel(r'$q R_g$')
        else:
            # standard Kratky plot
            x, y, dy = qdat, qdat**2*I_sub, qdat*qdat*sigma
            plt.ylabel(r'$I q^2$')
            plt.xlabel(r'$q$ [%s$^{-1}$]' % units)
        plt.errorbar(x,y,yerr=dy,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0,label=label)
        plt.plot(x,np.zeros_like(x),linestyle='--',color='grey',zorder=1) 
        if label: 
            plt.legend(frameon=False)
        plt.title('Kratky plot')
        plt.tight_layout()
        plt.savefig('Kratky.png',dpi=200)
    
    ## Porod
    if args.Porod:
        # transform data
        y = qdat**4 * (Idat - background)
        dy = qdat**4 * sigma

        # constant fit at Porod limit
        if Porod_limit:
            qm_Porod = Porod_limit
        else:
            useful_qmax = np.pi*Ng/dmax_out
            qm_Porod = useful_qmax*0.95
        if np.amax(qdat) <= qm_Porod:
            qm_Porod = 0.9*np.amax(qdat)
        idx = np.where(qdat>qm_Porod)
        a = np.polyfit(qdat[idx],y[idx],0,w=1/dy[idx])

        # Porod plot
        f,p0 = plt.subplots(1,1)
        p0.errorbar(qdat,y,yerr=dy,linestyle='none',marker='.',markersize=markersize,color='red',zorder=0)
        p0.plot([qm_Porod,qm_Porod],[np.amin(y),np.amax(y)],color='grey',linestyle='--')
        p0.plot(qdat[idx],qdat[idx]/qdat[idx]*a,color='black',label='fit with constant')
        p0.set_title('Porod plot')
        p0.set_ylabel(r'$I q^4$')
        p0.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        p0.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('Porod.png',dpi=200)

    ## import and plot data with rescaled errors
    if Prob < 0.003:
        f,(p0,p1) = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},sharex=True)

        # rescaled data
        p0.errorbar(qdat,Idat,yerr=sigma_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0,label='data with rescaled errors')
        if args.logx:
            p0.set_xscale('log')
            p0.plot(qdat,Ifit_interp,color='black',linewidth=linewidth,zorder=1,label='p(r) fit')
        else:
            p0.plot(qfit,Ifit,color='black',linewidth=linewidth,zorder=1,label='p(r) fit')
        p0.set_ylabel(r'$I(q)$')
        p0.set_yscale('log')
        p0.set_title('p(r) fit to data with rescaled errors')
        p0.legend(frameon=False)

        # residuals
        p1.plot(qdat,R_rs,linestyle='none',marker='.',markersize=markersize,color='blue',zorder=0)
        if args.logx:
            p1.set_xscale('log')
            p1.plot(qdat,np.zeros_like(qdat),linewidth=linewidth,color='black',zorder=1)
        else:
            p1.plot(qfit,np.zeros_like(qfit),linewidth=linewidth,color='black',zorder=1)
        p1.set_xlabel(r'$q$ [%s$^{-1}$]' % units)
        p1.set_ylabel(r'$\Delta I(q)/\sigma_\mathrm{rescale}$')
        try:
            p1.set_ylim(-maxR_rs,maxR_rs)
            p1.set_yticks([-maxR_rs,0,maxR_rs])
        except:
            printt("WARNING: Some residuals are either NaN or inf - bad fit?")
            printt("         probably just a numerical instability")
            printt("         try changing the number of points in p(r)")
        plt.savefig('Iq_rs.png',dpi=200)
        plt.tight_layout()

    ## output values
    printt("\n\n\n")
    printt("    Estimated parameters after runnning BayesApp:")
    printt("        dmax:                       %s" % dmax)
    printt("        dmax:                       %f" % dmax_out)
    printt("        transformation:             %s" % transformation)
    printt("        skip first points:          %d" % skip_first)
    printt("        number of points in p(r):   %s" % prpoints)
    printt("        number of outliers:         %d" % Noutlier)
    printt("        number of outliers removed: %d" % outliers_removed)
    if args.Guinier:
        printt("        Rg from Guinier analysis:   %f" % Rg_Guinier)
    if args.Kratky_Mw:
        printt("        Mw from Kratky integration: %1.1f +/- %1.1f kDa" % (MwF,dMwF))

    ### end timing
    end_time = time.time()-start_time
    printt('\n    total time:                   %0.1f seconds' % end_time)
    printt("\n\n\n")

    ## compress output files to zip file
    if args.zip_compress:
        import zipfile
        import glob 
        zip_filename = f'results_{prefix}.zip'
        printt('\n    compressing output to zip file: %s' % zip_filename)
        files_to_zip = ['filename', 'bift.f', exe, 'pr.dat', 'pr_bin.dat', 'pr_smooth.dat','data.dat', 'fit.dat', 'fit_q.dat', 'parameters.dat', 'rescale.dat','outlier_filtered.dat', 'scale_factor.dat', 'bayesapp.log', 'inputfile.dat']
        files_to_zip.extend(glob.glob('*.png')) # add png images
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files_to_zip:
                if os.path.exists(file):
                    zipf.write(file)
    
    ## show plots
    if args.show:
        plt.show()
