""" XXX: doc """

import numpy as np
import ctypes
from copy import copy, deepcopy
from hashlib import sha1

import pycuda.autoinit
from pycuda import gpuarray
from pycuda import driver
from pycuda import compiler

PADFACTOR_H = 16
PADFACTOR_W = 16

#import os
from os import path

from pythor2.utils import mkdir_p

import pycuda.autoinit
from pycuda import gpuarray
from pycuda import driver
from Cheetah.Template import Template

# -----------------------------------------------------------------------------
# XXX: tmp?
THREADS = 8,8,1
KEEP = False
DEBUG = False

MYPATH = path.dirname(path.realpath(__file__))

class InvalidParameter(Exception):
    pass

class FilterOp(object):

    # -------------------------------------------------------------------------
    def __init__(self,
                 in_, fb_, out_,
                 # -- meta-programming parameters
                 n_filter_rows = 1,
                 n_output4s = 'all',
                 spill = False,
                 imul_fast = True,
                 pad_shared = True,                 
                 use_tex1dfetch = True,
                 ):

        self.in_ = in_
        self.fb_ = fb_
        self.out_ = out_

        garr_out_l = out_._garr_l
        garr_in_l = in_._garr_l        

        # original shapes
        in_h, in_w, in_d = in_.height, in_.width, in_.depth
        fb_n, fb_h, fb_w, fb_d = fb_.nfilters, fb_.height, fb_.width, fb_.depth
        out_h, out_w, out_d = out_.height, out_.width, out_.depth

        assert out_d == fb_n

        # XXX: metaprog parameters (clean this up)

        if n_filter_rows == 'all':
            n_filter_rows = fb_h

        if n_output4s == 'all':
            n_output4s = len(garr_out_l)         

        if fb_h % n_filter_rows != 0:
            raise InvalidParameter("fb_h (%d) "
                                   "is not a multiple of n_filter_rows (%d)"
                                   % (fb_h, n_filter_rows))

        if len(garr_out_l) % n_output4s != 0:
            raise InvalidParameter("len(garr_out_l) (%d) "
                                   "is not a multiple of n_output4s (%d)"
                                   % (len(garr_out_l), n_output4s))

        # padded shapes
        garr_in_h, garr_in_w, garr_in_d = in_._garr_l[0].shape
        garr_out_h, garr_out_w, garr_out_d = out_._garr_l[0].shape

        block_w,block_h,block_d = threads = THREADS
        grid = (int(np.ceil(1.*garr_out_w/block_w)),(int(np.ceil(1.*garr_out_h/block_h))), 1)
        self._threads = threads
        self._grid = grid

        # -- generate gpu code

        # - define template options
        topts = {}
        # filters
        topts['FILTER_H'] = fb_h
        topts['FILTER_W'] = fb_w
        # input
        topts['INPUT_H'] = garr_in_h
        topts['INPUT_W'] = garr_in_w
        topts['INPUT_D'] = garr_in_d
        # output
        topts['OUTPUT_H'] = garr_out_h
        topts['OUTPUT_W'] = garr_out_w
        # blocks
        topts['BLOCK_W'] = block_w
        topts['BLOCK_H'] = block_h
        topts['INPUT_BLOCK_W'] = (block_w+fb_w-1) if ((block_w+fb_w-1)<garr_in_w) else garr_in_w
        topts['N_LOAD_ITERATIONS'] = int(np.ceil((block_w+fb_w-1.)/block_w))

        topts['PAD_SHARED'] = pad_shared
        topts['SPILL'] = spill

        # XXX: review
        topts['USE_TEX1DFETCH'] = use_tex1dfetch
        topts['N_OUTPUT4S'] = n_output4s
        topts['N_FILTERS'] = 4

        # XXX: tempppp
        assert fb_h % n_filter_rows == 0
        topts['N_FILTER_ROWS'] = n_filter_rows
        n_kernels = fb_h / n_filter_rows
        topts['N_KERNELS'] = n_kernels

        # XXX: TEMP
        topts['IMUL_FAST'] = imul_fast

        # - generate source from template
        basename = path.join(MYPATH, path.splitext(__file__)[-2])
        tmpl_basename = path.join(basename)
        tmpl_fname = tmpl_basename + ".template.cu"
        tmpl = Template(file=tmpl_fname, searchList=[topts])
        outstr = tmpl.respond()

        # - compile source
        opt_l = []
        opt_l += ["--opencc-options=-v,-OPT:0limit=0,-O3,-LIST:source=on"]
        opt_l += ["--ptxas-options=-v"]
        cubin_str = compiler.compile(outstr, options=opt_l)

        # - XXX
        mod = driver.module_from_buffer(cubin_str)

        cudafunc_l = [mod.get_function('cudafilter_kernel_%d' % nk)
                      .prepare("P"*(n_output4s+1), block=threads)
                      for nk in xrange(n_kernels)]

        # -- reference to texture memory
        if use_tex1dfetch:
            if fb_d == 1: 
                tex = mod.get_texref("tex_float")
                tex.set_format(driver.array_format.FLOAT, 1)
            else:
                tex = mod.get_texref("tex_float4")
                tex.set_format(driver.array_format.FLOAT, 4)

        # -- reference to constant memory
        const = mod.get_global("constant")[0]

        # -- prepare function calls
        grid2 = grid[:2]

        def fill_const_nocache(j, iz, oz):

            fb_sub = fb_[oz*4*n_output4s : (oz+1)*4*n_output4s,
                         j*n_filter_rows : (j+1)*n_filter_rows,
                         :,
                         iz*4 : (iz+1)*4]
            
            fb_sub = np.swapaxes(fb_sub, 0, 3)

            fb_sub = np.ascontiguousarray(fb_sub)
            
            # XXX: constant size check ?
            driver.memcpy_htod(const, fb_sub.data)

        # update fb_sub if necessary
        cudafunc_call_l = []

        for o4z in xrange(len(garr_out_l)/n_output4s):

            for iz, garr_in in enumerate(garr_in_l):

                # bind input texture
                if use_tex1dfetch:
                    cudafunc_call_l += [(garr_in.bind_to_texref, (tex,))]

                for j in xrange(fb_h/n_filter_rows):

                    # get cuda kernel function
                    cudafunc = cudafunc_l[j]
                    print cudafunc.registers

                    # fill constant memory
                    cudafunc_call_l += [(fill_const_nocache, (j, iz, o4z))]

                    # compute
                    cudafunc_call_l += [(cudafunc.prepared_call,
                                         [grid2, garr_in.gpudata] +
                                         [garr_out_l[o4z*n_output4s + i].gpudata
                                          for i in xrange(n_output4s)]
                                         )]

                # -
            # -
            
        # -
        # - private XXX
        self._cudafunc_call_l = cudafunc_call_l
#         self._mod = mod
#         self._cudafunc_l = cudafunc_l
#         self._const = const

    # -------------------------------------------------------------------------
    def __call__(self, **plugin_args):
        start = driver.Event()
        end = driver.Event()
        start.record()
        [func(*args) for func, args in self._cudafunc_call_l]         
        end.record()
        end.synchronize()
        return end.time_since(start)*1e-3


class Input(object):

    # -------------------------------------------------------------------------
    def __init__(self, height, width, depth, dtype='float32'):

        # constraints
        #assert nimgs == 1
        assert height == width
        assert depth % 4 == 0 or depth == 1
        assert dtype == 'float32'        

        #self.nimgs = nimgs
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype

        # padding
        padh = int(np.ceil(1.*height/PADFACTOR_H))*PADFACTOR_H
        padw = int(np.ceil(1.*width/PADFACTOR_W))*PADFACTOR_W
        padd = depth
        self._padded_shape = padh, padw, padd

        # gpuarray alloc
        if depth == 1:
            self._garr_l = [gpuarray.GPUArray((padh,padw,1),'float32')]
            ngarrs = 1
        else:
            ngarrs = int(np.ceil(depth / 4.))
            self._garr_l = [gpuarray.GPUArray((padh,padw,4),'float32') for _ in xrange(ngarrs)]
        
        self._garr_tmp = driver.pagelocked_empty(self._garr_l[0].shape, self.dtype)
        self._arr_tmp = np.empty((ngarrs,)+self._padded_shape[:2]+(self._garr_l[0].shape[-1],), dtype='float32')

    # -------------------------------------------------------------------------
    def __getitem__(self, index):

        g_l = self._garr_l
        
        my_h, my_w, my_d = self.height, self.width, self.depth
        data = self._garr_tmp

        # --
        if my_d == 1 or my_d == 4:
            g_l[0].get(data)
            return data[:my_h,:my_w,:my_d][index].copy()
        
        # -- 
        ngarrs = len(g_l)

        harr = self._arr_tmp
        for i in xrange(ngarrs):
            g_l[i].get(data)
            harr[i] = data

        out = harr[:,:my_h,:my_w,:]
        out = out.reshape(ngarrs, my_h, my_w, 1, self._garr_l[0].shape[-1])
        out = out.swapaxes(0,3)
        out = out.reshape((my_h, my_w, my_d))

        if index == slice(None, None, None):
            return out.copy()
        else:
            return out[index].copy()
        
    # -------------------------------------------------------------------------
    def setitem_opt_tmp(self, value): # pragma: no cover
        g_l = self._garr_l
        
        for i in xrange(len(g_l)):
            g_l[i].set(value[:,:,i*4:(i+1)*4])

    # -------------------------------------------------------------------------
    def __setitem__(self, index, value):
        g_l = self._garr_l

        # -- clear data with unique value ?
        if (np.array(index, dtype=object)==slice(None,None,None)).all() \
               and np.array(value).size == 1:
            for i in xrange(len(g_l)):
                g_l[i].fill(float(value))
                
        # -- full update
        elif index == slice(None,None,None) \
                 and value.shape == self._padded_shape:
            for i in xrange(len(g_l)):
                g_l[i].set(np.ascontiguousarray(value[:,:,i*4:(i+1)*4]))
                
        # -- standard update
        else:
            if index != slice(None,None,None):                
                harr = self[:]
                harr[index] = value
            else:
                harr = value
            tmp = driver.pagelocked_empty(g_l[0].shape, 'float32')
            h, w = self.height, self.width
            for i in xrange(len(g_l)):
                tmp[:h,:w,:4] = harr[:,:,i*4:(i+1)*4]
                g_l[i].set(tmp)

Output = Input

class Filterbank(object):

    # -------------------------------------------------------------------------
    def __init__(self, nfilters, height, width, depth, dtype='float32'):

        # constraints
        assert nfilters == 1 or nfilters % 4 == 0
        assert height == width
        assert depth == 1 or depth % 4 == 0

        assert dtype == 'float32' # for now

        self.nfilters = nfilters
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype
        
        self._ndarray = np.ndarray((nfilters,height,width,depth), dtype=dtype)

    # -------------------------------------------------------------------------
    def __getitem__(self, key):
        return self._ndarray[key].copy()

    # -------------------------------------------------------------------------
    def __setitem__(self, key, value):

        shape = self._ndarray[key].shape
        value = np.array(value)

        if value.size > 1:
            self._ndarray[key] = value.reshape(shape).astype(self.dtype)
        else:
            self._ndarray[key] = value.astype(self.dtype)


# # ------------------------------------------------------------------------------
# import hashlib
# def get_lib_fname(base, topts, copts):
#     strval = repr(topts)+repr(copts)
#     hashval = hashlib.sha1(strval).hexdigest()
#     return "%s__%s" % (base,hashval)

# # ------------------------------------------------------------------------------
# import subprocess
# def execute(cmd): # pragma: no cover
#     p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
#     out,err = p.communicate()
#     if p.returncode:
#         raise Exception, err
#     return out, err

