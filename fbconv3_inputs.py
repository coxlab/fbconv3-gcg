#!/usr/bin/env python
# -*- coding: utf-8 -*-

isize_l = [2**i for i in range(8,12)]
depth_l = [4**i for i in range(1,3+1)]
n_filters_l = [4**i for i in range(1,3+1)]
fsize_l = range(4, 9+1)

inputs_list = [
    dict(
        height = s,
        width = s,
        depth = d,
        n_filters = n,
        fsize = f,
        )
    for s in isize_l
    for d in depth_l
    for n in n_filters_l
    for f in fsize_l
    ]

print inputs_list
print len(inputs_list)
