import numpy as np

def printt(s): 
    """ print and write to log file"""
    print(s)
    with open('bayesapp.log','a') as f:
        f.write('%s\n' %s)

def get_header_footer(file):
    """
    get number of headerlines and footerlines
    """

    header,footer = 0,0
    f = open(file,errors='ignore')
    lines = f.readlines()

    CONTINUE_H,CONTINUE_F = True,True
    j = 0
    while CONTINUE_H or CONTINUE_F:
        line_h = lines[j]
        line_f = lines[-1-j]
        tmp_h = line_h.split()
        tmp_f = line_f.split()
        if CONTINUE_H:
            try:
                NAN = 0
                imax = min(len(tmp_h),3)
                for i in range(imax):
                    1/float(tmp_h[i]) # divide to ensure non-zero values
                    if np.isnan(float(tmp_h[i])):
                        NAN = 1
                if (NAN or imax == 1):
                    header+=1
                else:
                    CONTINUE_H = False
            except:
                header+=1
        if CONTINUE_F:
            try:
                NAN = 0
                imax = min(len(tmp_f),3)
                for i in range(imax):
                    1/float(tmp_f[i]) # divide to ensure non-zero values
                    if np.isnan(float(tmp_f[i])):
                        NAN = 1
                if (NAN or imax == 1):
                    footer+=1
                else:
                    CONTINUE_F = False
            except:
                footer+=1
        j+=1

    return header,footer

def read_params():
    """
    retrive output from BIFT parameter file parameters.dat
    """
    f = open('parameters.dat','r')
    lines = f.readlines()
    for line in lines:
        if 'Number of points in p(r)            =' in line:
            tmp = line.split('=')[1]
            prpoints_float = float(tmp)
        if 'I(0) estimated             :' in line:
            tmp = line.split(':')[1]
            I0 = float(tmp.split('+-')[0])
        if 'Maximum diameter           :' in line:
            tmp = line.split(':')[1]
            dmax = float(tmp.split('+-')[0])
        if 'Radius of gyration         :' in line:
            tmp = line.split(':')[1]
            Rg = float(tmp.split('+-')[0])
        if 'Reduced Chi-square         :' in line:
            tmp = line.split(':')[1]
            chi2r = float(tmp.split('+-')[0])
        if 'Background estimated       :' in line:
            background = float(line.split(':')[1])
        if 'Log(alpha) (smoothness)    :' in line:
            tmp = line.split(':')[1]
            alpha = float(tmp.split('+-')[0])
        if 'Number of good parameters  :' in line:
            tmp = line.split(':')[1]
            Ng = float(tmp.split('+-')[0])
        if 'Number of Shannon channels :' in line:
            Ns = float(line.split(':')[1])
        if 'Evidence at maximum        :' in line:
            tmp = line.split(':')[1]
            evidence = float(tmp.split('+-')[0])
        if 'Probability of chi-square  :' in line:
            Prob = float(line.split(':')[1])
            if Prob == 0.0:
                Prob_str = ' < 1e-20'
            elif Prob >= 0.001:
                Prob_str = '%1.3f' % Prob
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

    return I0,dmax,Rg,chi2r,background,alpha,Ng,Ns,evidence,Prob,Prob_str,assessment,beta,Rmax,Rmax_expect,dRmax_expect,p_Rmax_str,NR,NR_expect,dNR_expect,p_NR,prpoints_float
