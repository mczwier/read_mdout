'''
Parse an AMBER output file into a bunch of Numpy arrays.
This currently assumes that fields do not appear or disappear during the course
of MD.
'''

from __future__ import division, print_function

'''

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

import re, numpy

class MDOutParser(object):
    # Split on any digit boundary, absorbing whitespace, but only if that digit is
    # not preceded by a '-' (i.e. is not the '4' in '1-4 NB').
    re_split_line = re.compile(r'(?<=(?<!-)\d)\s*')
    
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
    
    def parse(self, mdout_file):
        type_overrides = self.type_overrides
        default_type = self.default_type
        block_start = self.block_start
        re_split_line = self.re_split_line
        re_split_pair = self.re_split_pair
        
        variables = {}
        nblocks = 0
        
        line = input_file.readline()        
        while line:
            if line.startswith(block_start):
                # beginning of a block
                blockvars = {}                
                while '=' in line:
                    pairlist = re_split_line.split(line.strip())
                    for pairtext in pairlist:
                        key, vtext = re_split_pair.split(pairtext.strip())
                        blockvars[key] = type_overrides.get(key,default_type)(vtext)
                    line = input_file.readline()

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
                line = input_file.readline()
        
        # Trim output arrays to the appropriate length
        for name in variables:
            variables[name] = numpy.resize(variables[name], (nblocks,))

    
        return variables
    
if __name__ == '__main__':        
    import argparse, h5py
    parser = argparse.ArgumentParser(description='''\
Parse AMBER mdout files into HDF5 files. Each quantity is stored in a data set
identified by the label within the mdout file (e.g. "NSTEP", "EPtot", "Density", 
and "EAMBER (non-restraint)"). In addition, average average values and RMS
fluctuations are stored as attributes on each data set.''')
    parser.add_argument('-o', '--output', default='mdout.h5',
                        help='''Store data in OUTPUT (default: %(default)s).''')
    parser.add_argument('input', nargs='?', metavar='MDOUT', default='mdout', 
                        help='Use MDOUT for input (default: %(default)s)')
    
    args = parser.parse_args()
    
    output_h5 = h5py.File(args.output, 'w')
    input_file = open(args.input, 'rt')
    parser = MDOutParser()
    variables = parser.parse(input_file)
    for name, array in variables.iteritems():
        output_ds = output_h5.create_dataset(name, data=array[:-2])
        output_ds.attrs['average'] = array[-2]
        output_ds.attrs['rmsfluct'] = array[-1]
