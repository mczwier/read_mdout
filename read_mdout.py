'''
Parse an AMBER output file into a bunch of Numpy arrays.
This currently assumes that fields do not appear or disappear during the course
of MD.
'''

from __future__ import division, print_function


import re, numpy
from collections import deque
from itertools import ifilter, imap

class MDOutParser(object):
    '''Parses energies and related terms from AMBER mdout files. To use:
        parser = MDOutParser()
        variables = parser.parse(open('mdout', 'rt'))
    
    The return value of ``parse()`` is a dictionary of Numpy arrays; if the run
    completed successfully, then the last two entries (indices -2 and -1,
    respectively) are the average value and RMS fluctuation over the run.
    '''     
    
    
    re_new_section       = re.compile(r'   \d+\.')
    re_sim_params_begin  = re.compile(r'   \d+\.  CONTROL  DATA  FOR  THE  RUN')
    re_time_series_begin = re.compile(r'   \d+\.  RESULTS')
    re_is_float          = re.compile(r'\.|inf|nan', re.IGNORECASE)
    re_is_bool           = re.compile(r'true|false',re.IGNORECASE)
        
    re_space_equals = re.compile(r'\s*=\s*')
    re_split_line = re.compile(r'\s*,\s*|\s+')
    
    # Split on any digit, absorbing whitespace, but only if that digit is
    # not preceded by a '-' or a letter (i.e. is not the '4' in '1-4 NB' or 0 in 'TEMP0').
    re_split_timeseries_line = re.compile(r'(?<=(?<!-|[A-Za-z])\d)\s*,?\s*',re.IGNORECASE)
    
    # Split about the equal sign
    re_split_pair = re.compile(r'\s*=\s*')
    
    initial_chunksize = 128
    default_type = float
    type_overrides = {'NSTEP': int} 
    block_start = ' NSTEP'
    
    def __init__(self, block_start=None, default_type=None, type_overrides=None):
        self.block_start = block_start or self.__class__.block_start
        self.default_type = default_type or self.__class__.default_type
        self.type_overrides = type_overrides if type_overrides else dict(self.__class__.type_overrides)
        
        self.simulation_params = {}
        self.time_series = {}
        self.time_series_averages = {}
        self.time_series_rmsds = {}
        
        self.mdout_file = None
        self.line = None
        
    def _discard_until_matches(self, regexp):
        mdout_file = self.mdout_file
        line = self.line
        while line and not regexp.match(line):
            line = mdout_file.readline()
        self.line = line
        
    def _parse_keyvalue_line(self, line):
        re_space_equals = self.re_space_equals
        re_split_line = self.re_split_line
        
        kvpairs = []
        
        if '=' in line:
            line = re_space_equals.sub(' = ', line)
            fields = deque(ifilter(None, imap(str.strip, re_split_line.split(line))))
            while(fields):
                key_fields = []
                field = fields.popleft()
                if field.startswith('('):
                    # ignore rest of line
                    break
                while field != '=':
                    key_fields.append(field)
                    field = fields.popleft()
                # discard equal
                key_word = ' '.join(key_fields)
                value_word = fields.popleft()
                kvpairs.append((key_word,value_word))
        return kvpairs
    
    def _make_keyvalue_dict(self, pairs):
        re_is_float = self.re_is_float
        re_is_bool  = self.re_is_bool
        kvdict = {}
        for (key,valuetxt) in pairs:
            if key.startswith('|'): continue
            
            if re_is_float.search(valuetxt):
                value = float(valuetxt)
            elif re_is_bool.match(valuetxt):
                valuetxt = valuetxt.lower()
                if valuetxt == 'true':
                    value = True
                else:
                    value = False
            else:
                try:
                    value = int(valuetxt)
                except ValueError:
                    value = valuetxt
            kvdict[key] = value
        return kvdict

    def parse(self, mdout_file):
        '''Parse ``mdout_file``, populating the instance variables ``simulation_params``,
        ``time_series``, ``time_series_averages``, and ``time_series_rmsds`` (all
        dictionaries) with results.
        '''
        
        self.mdout_file = mdout_file
        self.line = mdout_file.readline()
        self._discard_until_matches(self.re_sim_params_begin)
        self._parse_sim_params()
        self._discard_until_matches(self.re_time_series_begin)
        self._parse_timeseries()
        self.line = self.mdout_file = None
        

    def _parse_sim_params(self):
        '''Parse simulation parameters'''
        
        '''\
General flags:
     imin    =       0, nmropt  =       1

Nature and format of input:
     ntx     =       5, irest   =       1, ntrx    =       1

Nature and format of output:
     ntxo    =       1, ntpr    =     500, ntrx    =       1, ntwr    =     500
     iwrap   =       1, ntwx    =     500, ntwv    =       0, ntwe    =       0
     ioutfm  =       0, ntwprt  =       0, idecomp =       0, rbornstat=      0

Potential function:
     ntf     =       2, ntb     =       2, igb     =       0, nsnb    =      25
     ipol    =       0, gbsa    =       0, iesp    =       0
     dielc   =   1.00000, cut     =  10.00000, intdiel =   1.00000

Frozen or restrained atoms:
     ibelly  =       0, ntr     =       0

Molecular dynamics:
     nstlim  =    500000, nscm    =      1000, nrespa  =         1
     t       =   0.00000, dt      =   0.00200, vlimit  =  20.00000

Langevin dynamics temperature regulation:
     ig      =  631601
     temp0   = 293.00000, tempi   =   0.00000, gamma_ln=   1.00000

Pressure regulation:
     ntp     =       1
     pres0   =   1.00000, comp    =  44.60000, taup    =   5.00000
'''
        
        re_new_section = self.re_new_section

        parse_keyvalue_line = self._parse_keyvalue_line
        make_keyvalue_dict = self._make_keyvalue_dict
        
        variables = self.simulation_params = {}
        
        mdout_file = self.mdout_file
        line = self.line
        assert self.re_sim_params_begin.match(line)
        line = mdout_file.readline()        
        while line and not re_new_section.match(line):
            line = mdout_file.readline()
            if '=' in line:
                variables.update(make_keyvalue_dict(parse_keyvalue_line(line)))
        self.line = line
    
    def _parse_timeseries(self):
        '''Parse time series of dynamical quantities'''
        
        '''\

 NSTEP =      500   TIME(PS) =      51.000  TEMP(K) =   292.08  PRESS =  -749.1
 Etot   =    -85335.3869  EKtot   =     21219.1275  EPtot      =   -106554.5144
 BOND   =       913.8125  ANGLE   =      3093.9119  DIHED      =         0.0000
 1-4 NB =       -84.7569  1-4 EEL =    -21044.6175  VDWAALS    =      6750.2866
 EELEC  =    -96130.8970  EHBOND  =         0.0000  RESTRAINT  =         0.2673
 PM6ESCF=       -52.5213
 EAMBER (non-restraint)  =   -106554.7817
 EKCMT  =      8672.3424  VIRIAL  =     16086.9889  VOLUME     =    458419.0577
                                                    Density    =         0.7750
 Ewald error estimate:   0.2240E-01
 ------------------------------------------------------------------------------

 NMR restraints: Bond =    0.267   Angle =     0.000   Torsion =     0.000
===============================================================================

 NSTEP =     1000   TIME(PS) =      52.000  TEMP(K) =   291.75  PRESS =  -598.4
 Etot   =    -85470.0558  EKtot   =     21195.1120  EPtot      =   -106665.1678
 BOND   =       892.4097  ANGLE   =      3101.0571  DIHED      =         0.0000
 1-4 NB =       -84.2149  1-4 EEL =    -21058.3297  VDWAALS    =      6931.9891
 EELEC  =    -96383.3114  EHBOND  =         0.0000  RESTRAINT  =         0.0092
 PM6ESCF=       -64.7769
 EAMBER (non-restraint)  =   -106665.1770
 EKCMT  =      8865.2810  VIRIAL  =     14757.5145  VOLUME     =    456028.3312
                                                    Density    =         0.7791
 Ewald error estimate:   0.2208E-01
 ------------------------------------------------------------------------------

 NMR restraints: Bond =    0.009   Angle =     0.000   Torsion =     0.000
===============================================================================

 NSTEP =     1500   TIME(PS) =      53.000  TEMP(K) =   293.05  PRESS =  -495.0
 Etot   =    -85461.3057  EKtot   =     21289.5937  EPtot      =   -106750.8993
 BOND   =       918.7028  ANGLE   =      3123.5800  DIHED      =         0.0000
 1-4 NB =       -84.5184  1-4 EEL =    -21047.5166  VDWAALS    =      7049.9141
 EELEC  =    -96653.0998  EHBOND  =         0.0000  RESTRAINT  =         0.0734
 PM6ESCF=       -58.0349
 EAMBER (non-restraint)  =   -106750.9728
 EKCMT  =      8785.8676  VIRIAL  =     13638.4577  VOLUME     =    454035.9351
                                                    Density    =         0.7825
 Ewald error estimate:   0.2263E-01
 ------------------------------------------------------------------------------

 NMR restraints: Bond =    0.073   Angle =     0.000   Torsion =     0.000
===============================================================================
'''

        
        type_overrides = self.type_overrides
        default_type = self.default_type
        block_start = self.block_start
        re_new_section = self.re_new_section
        re_split_line = self.re_split_timeseries_line
        re_split_pair = self.re_split_pair
        
        variables = {}
        nblocks = 0
        
        mdout_file = self.mdout_file
        line = self.line
        assert self.re_time_series_begin.match(line)
        line = mdout_file.readline()        
        while line and not re_new_section.match(line):
            if line.startswith(block_start):
                # beginning of a block
                blockvars = {}                
                while '=' in line:
                    pairlist = re_split_line.split(line.strip())
                    for pairtext in pairlist:
                        key, vtext = re_split_pair.split(pairtext.strip())
                        blockvars[key] = type_overrides.get(key,default_type)(vtext)
                    line = mdout_file.readline()

                if nblocks > 0:
                    # Resize variables if necessary
                    for name in variables:
                        array = variables[name]
                        if len(array) == nblocks:
                            variables[name] = numpy.resize(array, (int(nblocks*1.5),))
                            
                    # Assign current value
                    for key, value in blockvars.iteritems():
                        variables[key][nblocks] = value
                else:
                    # Create arrays, one for each variable, and assign current value
                    for key, value in blockvars.iteritems():
                        variables[key] = numpy.empty((self.initial_chunksize,), dtype=type(value))
                        variables[key][0] = value
                nblocks += 1
            else:
                # Skip irrelevant line
                line = mdout_file.readline()
        
        # Trim output arrays to the appropriate length
        for name in variables:
            variables[name] = numpy.resize(variables[name], (nblocks,))
        
        # Account for average and RMSD values
        nstep = variables['NSTEP']
        if (nstep[-3] == nstep[-2] == nstep[-1]):
            for variable in variables:
                data = variables[variable]
                self.time_series_averages[variable] = data[-2]
                self.time_series_rmsds[variable] = data[-1]
                variables[variable] = numpy.resize(data, (len(data)-2,))  
    
        self.time_series = variables
        self.line = line
    
if __name__ == '__main__':        
    import argparse, h5py
    parser = argparse.ArgumentParser(description='''\
Parse AMBER mdout files into HDF5 files. Each quantity is stored in a data set
identified by the label within the mdout file (e.g. "NSTEP", "EPtot", "Density", 
"TIME(PS), and "EAMBER (non-restraint)"). In addition, average average values and RMS
fluctuations are stored as attributes on each data set.''')
    parser.add_argument('-o', '--output', default='mdout.h5',
                        help='''Store data in OUTPUT (default: %(default)s).''')
    parser.add_argument('input', nargs='?', metavar='MDOUT', default='mdout', 
                        help='Use MDOUT for input (default: %(default)s)')
    
    args = parser.parse_args()
    
    output_h5 = h5py.File(args.output, 'w')
    input_file = open(args.input, 'rt')
    parser = MDOutParser()
    parser.parse(input_file)
    
    attrs = output_h5.attrs
    for (k,v) in parser.simulation_params.iteritems():
        attrs[k] = v
    
    
    for name, array in parser.time_series.iteritems():
        output_ds = output_h5.create_dataset(name, data=array)
        
        if name in parser.time_series_averages:
            output_ds.attrs['average'] = parser.time_series_averages[name]
        
        if name in parser.time_series_rmsds:
            output_ds.attrs['rmsfluct'] = parser.time_series_rmsds[name]
