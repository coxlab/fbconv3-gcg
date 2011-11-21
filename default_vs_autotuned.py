#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import optparse
import cPickle as pkl
from os import path
from pprint import pprint

import scipy as sp

# ------------------------------------------------------------------------------
default_metaparams = dict(
    block_w = 8,
    block_h = 8,
    n_filter_rows = 1,
    n_output4s = 'all',
    spill = False,
    imul_fast = True,
    pad_shared = True,
    use_tex1dfetch = True,
    maxrregcount = None,
    )

# ------------------------------------------------------------------------------
def default_vs_autotuned(fname):

    fin = open(fname)
    all_timings = pkl.load(fin)

    print fname

    # --
    gflop = all_timings[0]['gflop']
    tot_runs = len(all_timings[0]['timings']['cuda'])
    half_runs = int(tot_runs / 2)
    metaparams_names = default_metaparams.keys()
    constant_metaparams = [all_timings]

    # -- performance of the default configuration
    default_timings_l = [item for item in all_timings
                         if item['metaparams'] == default_metaparams]
    assert len(default_timings_l) == 1
    default_timings = default_timings_l[0]['timings']
    default_gflops = sp.median(gflop/sp.array(default_timings['cuda'])[half_runs:])
    default_std = sp.std(gflop/sp.array(default_timings['cuda'])[half_runs:])

    # -- performance of the autotuned configuration
    # first, select the best performing configuration on the first half
    sel_gflops = [sp.median(gflop/sp.array(t['timings']['cuda'])[:half_runs])
                  for t in all_timings]
    autotuned_idx = sp.argmax(sel_gflops)
    # then, compute (unbiased) performance
    autotuned_timings = sp.array(all_timings[autotuned_idx]['timings']['cuda'][half_runs:])
    autotuned_gflops = sp.median(gflop/autotuned_timings)
    autotuned_std = sp.std(gflop/autotuned_timings)
    autotuned_metaparams = all_timings[autotuned_idx]['metaparams']

    percent_boost = 100. * ((autotuned_gflops - default_gflops) / default_gflops)

    # -- print the info
    fname_info = fname.split("__")
    print "=" * 80
    print "GPU:", fname_info[0]
    print "Input:", " ".join(fname_info[3:])
    print "-" * 80
    print "[DEFAULT]"
    print "gflops: median=%.3f std=%.3f" % (default_gflops, default_std)
    print "metaparams:"
    pprint(default_metaparams)
    print "-" * 80
    print "[AUTOTUNED]"
    print "gflops: median=%.3f std=%.3f" % (autotuned_gflops, autotuned_std)
    print "metaparams:"
    pprint(autotuned_metaparams)
    print "-" * 80
    print "percent boost: %.2f%%" % percent_boost
    print "=" * 80

    return default_gflops, default_std, autotuned_gflops, autotuned_std, percent_boost


# ------------------------------------------------------------------------------
def main():

    usage = "Usage: %prog [options] <filename> "

    parser = optparse.OptionParser(usage=usage)

    opts, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
    else:
        fname = args[0]

        kwargs = eval(str(opts))

        default_vs_autotuned(fname, **kwargs)

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






