#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import optparse
import cPickle as pkl
from os import path
from glob import glob
import time
from pprint import pprint
from hashlib import sha1
import cPickle as pkl

import scipy as sp
from scipy import linalg, signal
from numpy import testing

DEFAULT_NOVERIFY = False
DEFAULT_N_WARMUPS = 2
DEFAULT_N_RUNS = 10
DEFAULT_METAPARAMS = {}

# tmp
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

# NOTE: metaprog/autotune parameters
# will come from a second arg (default to None for no autotune)
# that points to a file with the parameters to explore
# bruteforce at first and then possibly with ES in the future

FACTOR = 16
FILTER_TYPE = 'correlate'
RSEED = 123

from fbconv3_utils import InvalidConfig, MYPATH

import fbconv3_cuda as mod

def benchmark_run(
    #method,
    # -- input parameters
    nimgs = 1,
    height = 16*FACTOR,
    width = 16*FACTOR,
    depth = 4,
    n_filters = 4,
    fsize = 9,
    # -- metaprog parameters
    metaparams = DEFAULT_METAPARAMS,
    # -- benchmark parameters
    n_warmups = DEFAULT_N_WARMUPS,
    n_runs = DEFAULT_N_RUNS,
    noverify = DEFAULT_NOVERIFY,
    ):

    assert height-fsize+1 > 0
    assert width-fsize+1 > 0
    assert n_filters % 4 == 0

    # -- Generate the data
    sp.random.seed(RSEED)
    in_data = sp.random.randn(height, width, depth).astype('float32')
    fb_data = sp.random.randn(n_filters, fsize, fsize, depth).astype('float32')
    out_data = sp.empty((height-fsize+1, width-fsize+1, n_filters), dtype='float32')
        
    gflop = out_data.size * (fsize * fsize * depth * 2) / (1000.**3.)

    if not noverify:
        hashme = pkl.dumps([in_data, fb_data, FILTER_TYPE],
                           protocol=pkl.HIGHEST_PROTOCOL)
        data_hash = sha1(hashme).hexdigest()
        fname = path.join("out_gt", data_hash)
        if not path.exists(fname):
            print "Computing and caching ground truth (CPU/numpy) ..."
            for n in xrange(n_filters):
                if FILTER_TYPE == 'correlate':
                    out_data[:,:,n] = signal.correlate(in_data,
                                                       fb_data[n],
                                                       mode='valid')[:,:,0]
                elif FILTER_TYPE == 'convolve':
                    filt = fb_data[n].flat[::-1].reshape(fb_data[n].shape)
                    out_data[:,:,n] = signal.fftconvolve(in_data,
                                                         filt,
                                                         mode='valid')[:,:,0]
                else:
                    raise ValueError("FILTER_TYPE '%s' not understood" % FILTER_TYPE)
            out_gt = out_data.copy()
            pkl.dump(out_gt, open(fname, 'w+'),
                     protocol=pkl.HIGHEST_PROTOCOL)
        else:
            print "Loading ground truth (CPU/numpy) ..."            
            out_gt = pkl.load(open(fname))

    all_timings = {}

    timings = dict([(key, [])
                    for key in ('upload',
                                #'set_up',
                                'process',
                                'cuda',
                                'download',                                    
                                )
                    ]
                   )

    in_ = mod.Input(height, width, depth)
    fb_ = mod.Filterbank(n_filters, fsize, fsize, depth)
    out_ = mod.Output(height-fsize+1, width-fsize+1, n_filters)        

    # -- set-up operation (e.g. compilation)
    fb_[:] = 0
    fop = mod.FilterOp(in_, fb_, out_, **metaparams)

    for i in xrange(n_warmups + n_runs):
        print "=" * 80 
        print "Trial %03d (%s)" % (i+1, 'run' if i >= n_warmups else 'warmup')

        # -- upload data
        start = time.time()
        in_[:] = in_data
        out_[:] = 0
        fb_[:] = fb_data
        end = time.time()
        t_upload = end-start

        # -- process convolution
        # XXX: Filter != Conv 
        start = time.time()
        t_cuda = fop()
        end = time.time()
        t_process = end-start

        start = time.time()
        out_data = out_[:]
        end = time.time()
        t_download = end-start

        if i >= n_warmups:
            timings['upload'] += [t_upload]
            timings['process'] += [t_process]
            timings['cuda'] += [t_cuda]
            timings['download'] += [t_download]

        gflops_cuda = gflop / t_cuda
        gflops_proc = gflop / t_process
        gflops_tot = gflop / (t_process+t_upload+t_download)

        print "gflops_cuda", gflops_cuda
        print "gflops_proc", gflops_proc
        print "gflops_tot", gflops_tot


    if not noverify:
        print "Verify last output..."
        #diffmax = max(sp.absolute(out_data-out_gt).ravel())
        testing.assert_array_almost_equal(out_data, out_gt, 1e-3)

    timings_stats = dict([(key, {'median': sp.median(t),
                                 'mean': sp.mean(t),
                                 'std': sp.std(t),
                                 'max': max(t),
                                 'min': min(t),                                     
                                 }
                           )
                          for key, t in timings.iteritems()
                          ]
                         )

    gflops_cuda = {
        'median': gflop / timings_stats['cuda']['median'],
        'mean': gflop / timings_stats['cuda']['mean'],
        'max': gflop / timings_stats['cuda']['min'],
        'min': gflop / timings_stats['cuda']['max'],
        }

    pprint(timings_stats)
    pprint(gflops_cuda)

    return timings, gflop
    
    
def benchmark(
    output_path,
    inputs_fname,
    metaparams_fname = None,
    # -- benchmark parameters
    n_warmups = DEFAULT_N_WARMUPS,
    n_runs = DEFAULT_N_RUNS,
    noverify = DEFAULT_NOVERIFY,
    ):

    assert path.exists(output_path) and path.isdir(output_path)

    in_dict = {}
    execfile(inputs_fname, {}, in_dict)
    inputs_list = in_dict['inputs_list']

    if metaparams_fname is not None:
        mp_dict = {}
        execfile(metaparams_fname, {}, mp_dict)
        metaparams_list = mp_dict['metaparams_list']
    else:
        metaparams_list = [{}]

    print "=" * 80
    print len(inputs_list), "benchs to run ..."
    print len(metaparams_list), "metaparams to evaluate ..."
    print "=" * 80

    #sp.random.seed(RSEED)
    #sp.random.shuffle(inputs_list)
    #sp.random.shuffle(metaparams_list)

    n_tot = len(inputs_list) * len(metaparams_list)
    it = 0

    for ii, inputs in enumerate(inputs_list):
        pprint(inputs)
        
        kwargs = inputs

        out_fname = mod.device_name + "__"
        out_fname += sha1(open('fbconv3_cuda.py').read()).hexdigest() + "__"
        out_fname += sha1(open('fbconv3_cuda.template.cu').read()).hexdigest() + "__"        
        out_fname += "__".join(["%s=%s" % (key, value) for key, value in inputs.iteritems()])
        out_fname = path.join(output_path, out_fname + '.pkl')

        if path.exists(out_fname):
            continue

        results = []
        for im, metaparams in enumerate(metaparams_list):
            
            kwargs.update({'metaparams': metaparams,
                           'n_warmups': n_warmups,
                           'n_runs': n_runs,
                           'noverify': noverify})

            try:
                timings, gflop = benchmark_run(**inputs)
                results += [{
                    'metaparams': metaparams,
                    'timings': timings,
                    'gflop': gflop,
                    }]
            except InvalidConfig, err:
                print err
                pass

            it += 1

            print "*" * 80
            print "*" * 80
            pprint(kwargs)
            print "." * 80
            print "Inputs: %02.3f %%" % (100.*(ii+1)/len(inputs_list))
            print "Metaparams: %02.3f %%" % (100.*(im+1)/len(metaparams_list))
            print "." * 80
            print "Total: %02.3f %%" % (100.*(it+1)/n_tot)
            print "*" * 80
            print "*" * 80


        print "Writing", out_fname
        pkl.dump(results, open(out_fname, 'w+'), protocol=pkl.HIGHEST_PROTOCOL)

# ------------------------------------------------------------------------------
def main():

    usage = "Usage: %prog [options] <output_path> <inputs_fname> [<metaparams_fname>] "

#     detected_methods = [path.splitext(fname)[-2].split('fbconv3_')[-1]
#                         for fname in glob('fbconv3_*.py')
#                         ]
    
#     usage = "Usage: %prog [options] <method> <inputs_fname> [<metaparams_fname>] "
#     usage += "\nDetected methods: "
#     usage += ", ".join(detected_methods)
#     usage += "\nExample: python %prog " + detected_methods[0]
#     usage += " fbconv3_inputs.py fbconv3_%s_metaparams.py" % detected_methods[0]

    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--n_warmups",
                      type = "int",
                      metavar = "INT",
                      default=DEFAULT_N_WARMUPS,
                      help="number of warmup runs before benchmark " 
                      "[default=%default]")
    
    parser.add_option("--n_runs",
                      type = "int",
                      metavar = "INT",
                      default=DEFAULT_N_RUNS,
                      help="number of runs (benchmark) " 
                      "[default=%default]")
    
    parser.add_option("--noverify",
                      default=DEFAULT_NOVERIFY,
                      action="store_false" if DEFAULT_NOVERIFY else "store_true",
                      help="don't verify output "
                      "against CPU/numpy implementation [default=%default]")

    opts, args = parser.parse_args()

    if not 2 <= len(args) <= 3 :
        parser.print_help()
    else:
        output_path = args[0]
        inputs_fname = args[1]
        if len(args) == 3:
            metaparams_fname = args[2]
        else:
            metaparams_fname = None
            
        kwargs = eval(str(opts))

        benchmark(
            output_path,
            inputs_fname,
            metaparams_fname,
            **kwargs)
                       
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






