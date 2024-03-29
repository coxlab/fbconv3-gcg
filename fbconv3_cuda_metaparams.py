#!/usr/bin/env python
# -*- coding: utf-8 -*-

block_w_l = [2**i for i in range(2,7+1)]
block_h_l = [2**i for i in range(2,7+1)]
n_filter_rows_l = [1, 2]
n_output4s_l = ['all'] + [1, 2]
spill_l = [False, True]
imul_fast_l = [True]#, False]
pad_shared_l = [True, False]
use_tex1dfetch_l = [True]#, False]
maxrregcount_l = [None]# + [2**i for i in xrange(3,5+1)]

metaparams_list = [
    dict(
        block_w = block_w,
        block_h = block_h,
        n_filter_rows = n_filter_rows,
        n_output4s = n_output4s ,
        spill = spill,
        imul_fast = imul_fast,
        pad_shared = pad_shared,
        use_tex1dfetch = use_tex1dfetch,
        maxrregcount = maxrregcount,
        )
    for block_w in block_w_l
    for block_h in block_h_l
    for n_filter_rows in n_filter_rows_l
    for n_output4s in n_output4s_l
    for spill in spill_l
    for imul_fast in imul_fast_l
    for pad_shared in pad_shared_l
    for use_tex1dfetch in use_tex1dfetch_l
    for maxrregcount in maxrregcount_l
    ]

#print metaparams_list
print len(metaparams_list)
