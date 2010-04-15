#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import optparse
import cPickle as pkl
import hashlib
from os import path
import time
from pprint import pprint

import scipy as sp
from scipy import linalg, signal
from numpy import testing

DEFAULT_NOVERIFY = False
DEFAULT_N_WARMUPS = 2
DEFAULT_N_RUNS = 10

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
    
def default(
    method,
    # -- input parameters
    nimgs = 1,
    height = 16*FACTOR,
    width = 16*FACTOR,
    depth = 4,
    nfilters = 16,
    fsize = 8,
    # -- benchmark parameters
    n_warmups = 2,
    n_runs = 10,
    noverify = DEFAULT_NOVERIFY,
    ):

    # -- Generate the data
    sp.random.seed(RSEED)
    in_data = sp.random.randn(height, width, depth).astype('float32')
    fb_data = sp.random.randn(nfilters, fsize, fsize, depth).astype('float32')
    out_data = sp.empty((height-fsize+1, width-fsize+1, nfilters), dtype='float32')
        
    gflop = out_data.size * (fsize * fsize * depth * 2) / (1000.**3.)

    if not noverify:
        hashme = pkl.dumps([in_data, fb_data, FILTER_TYPE],
                           protocol=pkl.HIGHEST_PROTOCOL)
        data_hash = hashlib.sha1(hashme).hexdigest()
        fname = path.join("out_gt", data_hash)
        if not path.exists(fname):
            print "Computing and caching ground truth (CPU/numpy) ..."
            for n in xrange(nfilters):
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

    outputs = []
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

    outputs = []

    mod = __import__("fbconv3_" + method)

    in_ = mod.Input(height, width, depth)
    fb_ = mod.Filterbank(nfilters, fsize, fsize, depth)
    out_ = mod.Output(height-fsize+1, width-fsize+1, nfilters)

    # -- set-up operation (e.g. compilation)
    fb_[:] = 0
    fop = mod.FilterOp(in_, fb_, out_)

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
            #timings['set_up'] += [t_set_up]
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
        diffmax = max(sp.absolute(out_data-out_gt).ravel())
        #print out_data
        #print out_gt
        testing.assert_array_almost_equal(out_data, out_gt, 1e-3)

    timings_stats = dict([(key, {'median':sp.median(t),
                                 'mean':sp.mean(t),
                                 'std':sp.std(t),
                                 'max':max(t),
                                 'min':min(t),                                     
                                 }
                           )
                          for key, t in timings.iteritems()
                          ]
                         )

    pprint(timings_stats)
    gflops_median = gflop / timings_stats['process']['median']
    gflops_mean = gflop / timings_stats['process']['mean']
    gflops_max = gflop / timings_stats['process']['min']
    gflops_min = gflop / timings_stats['process']['max']

    print "gflops_median", gflops_median
    print "gflops_mean", gflops_mean
    print "gflops_max", gflops_max
    print "gflops_min", gflops_min

# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <method> "
    
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

    if len(args) != 1:
        parser.print_help()
    else:

        method = args[0]

        kwargs = eval(str(opts))

        default(method, **kwargs)
        
                       
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






