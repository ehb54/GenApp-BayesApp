#!/opt/miniconda3/bin/python

import json
import io
import sys
import os
import time
import subprocess

from genapp3 import genapp
from bayesapp_helpfunctions import *

def execute(command,f):
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    message = genapp(json_variables)
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
                out_line = '\n\n!!!ERROR!!!\nProcess stopped - could not find solution. Is data input a SAXS/SANS dataset with format (q,I,sigma)?\n\n'
                message.udpmessage({"_textarea": out_line})
                sys.exit()
            elif total_time > maximum_time:
                popen.terminate()
                out_line = '\n\n!!!ERROR!!!\nProcess stopped - reached max time of 5 min (300 sec). Is data input a SAXS/SANS dataset with format (q,I,sigma)?. If data is large (several thousand data points), consider rebinning the data.\n\n'
                message.udpmessage({"_textarea": out_line})
                sys.exit()
            else:
                out_line = '%s\n' % nline_latin
                message.udpmessage({"_textarea": out_line})
            f.write(out_line)
    return out_line

def read_logfile():
    f = open('bayesapp.log','r')
    lines = f.readlines()
    for line in lines:
        if 'Rg from Guinier analysis:' in line:
            Rg_Guinier = float(line.split(':')[1].strip())
        if 'number of outliers:' in line:
            Noutlier = int(line.split(':')[1].strip())
        if 'number of outliers removed:' in line:
            outliers_removed = int(line.split(':')[1].strip())
        if 'skip first points:' in line:
            skip_first_out = int(line.split(':')[1].strip())
    f.close()
    return Rg_Guinier,Noutlier,outliers_removed,skip_first_out

if __name__=='__main__':

    ## set up Json-related variables
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    message = genapp(json_variables) # set up messaging

    ## initialize execute command
    f = open('stdout.dat','w')
    path = os.path.dirname(os.path.realpath(__file__))
    command_to_execute = ['/opt/miniconda3/bin/python','-u',path + '/bayesapp.py']
    #################################################################################
    ## read Json input and add to execute commmand
    #################################################################################
    
    # general input
    data_file_path = json_variables['datafile'][0] # name of datafile
    command_to_execute.extend(['--data_file', data_file_path])
    prefix = data_file_path.split('/')[-1] 
    if json_variables['qmin']:
        command_to_execute.extend(['--qmin',json_variables['qmin']])
    if json_variables['qmax']:
        command_to_execute.extend(['--qmax',json_variables['qmax']])
    command_to_execute.extend(['--units',json_variables['units']]) 

    # about reading checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore I use try-except to import
    
    # read input Guinier analysis
    try:
        dummy = json_variables['Guinier']
        command_to_execute.extend(['--Guinier'])
        Guinier = True
        if json_variables['Guinier_qmaxRg']:
            command_to_execute.extend(['--Guinier_qmaxRg',json_variables['Guinier_qmaxRg']])
        if json_variables['Guinier_skip']:    
            command_to_execute.extend(['--Guinier_skip',json_variables['Guinier_skip']])
    except:
        Guinier = False

    # read input Kratky plot
    try:
        dummy = json_variables['Kratky']
        command_to_execute.extend(['--Kratky'])
        try:
            dummy = json_variables['Kratky_dim']
            command_to_execute.extend(['--Kratky_dim'])
        except:
            pass
        try:
            dummy = json_variables['Kratky_Mw']
            command_to_execute.extend(['--Kratky_Mw'])
        except:
            pass
    except:
        pass

    # read input Porod plot
    try:
        dummy = json_variables['Porod']
        command_to_execute.extend(['--Porod'])
        if json_variables['Porod_limit']:
            command_to_execute.extend(['--Porod_limit',json_variables['Porod_limit']])
    except:
        pass

    # read extra options
    try:
        dummy = json_variables['options']
        dmax = json_variables['dmax'] # Maximum diameter
        if dmax:
            command_to_execute.extend(['--dmax',dmax])
        command_to_execute.extend(['--transformation',json_variables['transform']])
        command_to_execute.extend(['--nrebin',json_variables['nrebin']])
        alpha = json_variables['alpha'] # alpha
        if alpha:
            command_to_execute.extend(['--alpha',alpha])
        if json_variables['smearing']:
            command_to_execute.extend(['--smear',json_variables['smearing']])
        prpoints = json_variables['prpoints'] # number of points in p(r)
        if prpoints:
            command_to_execute.extend(['--prpoints',prpoints])
        if json_variables['noextracalc']:
            command_to_execute.extend(['--noextracalc',json_variables['noextracalc']])
        rescale_mode = json_variables['rescale_mode']
        command_to_execute.extend(['--rescale_mode',rescale_mode])
        if json_variables['Bg']:
            command_to_execute.extend(['--Bg',json_variables['Bg']])
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
            command_to_execute.extend(['--fitbackground','Y'])
        except:
            command_to_execute.extend(['--fitbackground','N'])
        try:
            dummy = json_variables['logx']
            command_to_execute.extend(['--logx'])
        except:
            pass
        try:
            dummy = json_variables['make_pr_bin']
            command_to_execute.extend(['--make_pr_bin'])
            make_pr_bin = True
            command_to_execute.extend(['--pr_binsize',json_variables['pr_binsize']])
        except:
            make_pr_bin = False
        skip_first = json_variables['skip_first'] # skip first points
        if skip_first:
            command_to_execute.extend(['--skip_first',json_variables['skip_first']])
        try:
            dummy = json_variables["outlier_ite"]
            command_to_execute.extend(['--outlier_ite'])
        except:
            pass
        try:
            dummy = json_variables["fast_run"]
            command_to_execute.extend(['--fast_run'])
        except:
            pass
    except:
        make_pr_bin = False
        rescale_mode = 'C'

    ## ask bayesapp to compress results to a zip file    
    command_to_execute.extend(['--zip_compress'])

    ## read genapp output folder
    folder = json_variables['_base_directory'] # output folder dir

    ## run bayesapp
    message.udpmessage({"_textarea": 'Running BayesApp...\n'}) 
    f = open('stdout.dat','w')
    execute(command_to_execute,f)
    f.close()

    ## retrive output from parameter file
    I0,dmax_out,Rg,chi2r,background,alpha_out,Ng,Ns,evidence,Prob,Prob_str,assessment,beta,Run_max,Run_max_expect,dRun_max_expect,p_Run_max_str,NR,NR_expect,dNR_expect,p_NR,prpoints = read_params()
   
    ## retrieve output from bayesapp.log file
    Rg_Guinier,Noutlier,outliers_removed,skip_first_out = read_logfile()

    ## generate output
    output = {} # create an empty python dictionary
    
    # files
    if make_pr_bin:
        output["pr"] = "%s/pr_bin.dat" % folder
    else:
        output["pr"] = "%s/pr_smooth.dat" % folder
    output["dataused"] = "%s/data.dat" % folder
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

    # output values
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
    output["Noutlier"] = "%d" % Noutlier # get from python script
    output["outliers_removed"] = "%d" % outliers_removed # get from python script
    output["skip_first_out"] = "%d" % skip_first_out # get from python script
    output["Ng"] = "%1.2f" % Ng
    output["shannon"] = "%1.2f" % Ns
    output["logalpha"] = "%1.2f" % alpha_out 
    output["evidence"] = "%1.2f" % evidence
    output["prpoints_out"] = prpoints ## read from parameter file

    # print all json input and output, for debugging
    # output['_textarea'] = "JSON output from executable:\n" + json.dumps( output, indent=4 ) + "\n\n";
    # output['_textarea'] += "JSON input to executable:\n" + json.dumps( json_variables, indent=4 ) + "\n";
    
    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output


