# helpfunctions for bayesapp

import time
import subprocess
from genapp3 import genapp
import io
import json
import sys
import numpy as np

def get_header_footer(file):
    """
    get number of headerlines and footerlines
    """

    header,footer = 0,0
    f = open(file)
    try:
        lines = f.readlines()
    except:
        print('Error: cannot read lines of file. Do you have some special characters in the file? Try removing them and rerun')
        print('file: %s' % file)

    CONTINUE_H,CONTINUE_F = True,True
    j = 0
    while CONTINUE_H or CONTINUE_F:
        line_h = lines[j]
        line_f = lines[-1-j]
        tmp_h = line_h.split()
        tmp_f = line_f.split()
        try:
            NAN = 0
            for i in range(len(tmp_h)):
                1/float(tmp_h[i]) # divide to ensure non-zero values
                if np.isnan(float(tmp_h[i])):
                    NAN = 1
            if NAN:
                header+=1
            else:
                CONTINUE_H = False
        except:
            header+=1
        try:
            NAN = 0
            for i in range(len(tmp_f)):
                1/float(tmp_f[i]) # divide to ensure non-zero values
                if np.isnan(float(tmp_f[i])):
                    NAN = 1
            if NAN:
                footer+=1
            else:
                CONTINUE_F = False
        except:
            footer+=1
        j+=1

    return header,footer

def execute(command,f):
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    d = genapp(json_variables)
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
                d.udpmessage({"_textarea": out_line})
                sys.exit()
            elif total_time > maximum_time:
                popen.terminate()
                out_line = '\n\n!!!ERROR!!!\nProcess stopped - reached max time of 5 min (300 sec). Is data input a SAXS/SANS dataset with format (q,I,sigma)?. If data is large (several thousand data points), consider rebinning the data. Or reduce number of points in p(r).\n\n'
                d.udpmessage({"_textarea": out_line})
                sys.exit()
            else:
                out_line = '%s\n' % nline_latin
                d.udpmessage({"_textarea": out_line})
            f.write(out_line)
    return out_line
