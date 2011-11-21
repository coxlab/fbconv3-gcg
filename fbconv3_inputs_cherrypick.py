#!/usr/bin/env python
# -*- coding: utf-8 -*-

isize_l = [256, 2048]
depth_l = [4, 64]
n_filters_l = [4, 64]
fsize_l = [4, 5, 8, 9, 16, 17]

inputs_list = [

    # -- default
    # i.e. the input used on the 8600 GT
    # to get coarse default parameters
    dict(height = 256,
         width = 256,
         depth = 8,
         n_filters = 64,
         fsize = 9,
         ),

    # -- others
    dict(height = 512,
         width = 512,
         depth = 4,
         n_filters = 32,
         fsize = 13,
         ),

    dict(height = 1024,
         width = 1024,
         depth = 8,
         n_filters = 16,
         fsize = 5,
         ),

    dict(height = 2048,
         width = 2048,
         depth = 4,
         n_filters = 4,
         fsize = 8,
         ),

    ]

print inputs_list
print len(inputs_list)
