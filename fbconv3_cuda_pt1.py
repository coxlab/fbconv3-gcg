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

class FilterOp(object):

    # -------------------------------------------------------------------------
    def __init__(self, in_, fb_, out_, **kwargs):

        self.in_ = in_
        self.fb_ = fb_
        self.out_ = out_

        # original shapes
        in_h, in_w, in_d = in_.height, in_.width, in_.depth
        fb_n, fb_h, fb_w, fb_d = fb_.nfilters, fb_.height, fb_.width, fb_.depth
        out_h, out_w, out_d = out_.height, out_.width, out_.depth

        assert out_d == fb_n
        
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
        # output
        topts['OUTPUT_H'] = garr_out_h
        topts['OUTPUT_W'] = garr_out_w
        # blocks
        topts['BLOCK_W'] = block_w
        topts['BLOCK_H'] = block_h
        topts['INPUT_BLOCK_W'] = (block_w+fb_w-1) if ((block_w+fb_w-1)<garr_in_w) else garr_in_w
        topts['N_LOAD_ITERATIONS'] = int(np.ceil((block_w+fb_w-1.)/block_w))

        # - generate source from template
        basename = path.join(MYPATH, path.splitext(__file__)[-2]+"_z%d" % (1 if in_d == 1 else 4))
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

        cudafunc_l = [mod.get_function('cudafilter_kernel_j%d' % j)
                      .prepare("PP", block=threads)
                      for j in xrange(fb_h)]

        # -- reference to texture memory
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
        garr_out_l = out_._garr_l
        garr_in_l = in_._garr_l

        def cut_fb():
            return [[[
                np.ascontiguousarray(
                    np.swapaxes(fb_[oz*4:(oz+1)*4,j,:,iz*4:(iz+1)*4], 0, 2)
                    ).data
                for j in xrange(fb_h)]
                for iz in xrange(len(garr_in_l))]
                for oz in xrange(len(garr_out_l))]
        
        fb_view_uint8 = fb_._ndarray.view('uint8')
        self._fb_hash = sha1(fb_view_uint8).hexdigest()        
        self._fb_sub_l = cut_fb()

        def fb_sub_l_update():
            fb_hash = sha1(fb_view_uint8).hexdigest()
            # handle the case when the fb data has changed
            if fb_hash != self._fb_hash:
                self._fb_sub_l = cut_fb()
                self._fb_hash = fb_hash
            else:
                print "HASH HIT!"

        #def fill_const(j, iz, oz):
        #    fb_sub = self._fb_sub_l[oz][iz][j]
        #    driver.memcpy_htod(const, fb_sub)

        def fill_const_nocache(j, iz, oz):
            fb_sub = np.swapaxes(fb_[oz*4:(oz+1)*4,j,:,iz*4:(iz+1)*4], 0, 2)
            fb_sub = np.ascontiguousarray(fb_sub)
            # XXX: constant size check ?
            driver.memcpy_htod(const, fb_sub.data)

        # update fb_sub if necessary
        cudafunc_call_l = []

        #cudafunc_call_l += [(fb_sub_l_update, ())]
        
        for oz, garr_out in enumerate(garr_out_l):

            # XXX: clear hout
            #cudafunc_call_l += [(garr_out.fill, (0,))]
            #garr_out.fill(0)
            # get gpu data pointer
            garr_out_ptr = garr_out.gpudata
            
            for iz, garr_in in enumerate(garr_in_l):

                # bind input texture
                #garr_in.bind_to_texref(tex)
                cudafunc_call_l += [(garr_in.bind_to_texref, (tex,))]
                # get gpu data pointer
                garr_in_ptr = garr_in.gpudata
                                         
                for j in xrange(fb_h):

                    # get kernel
                    cudafunc = cudafunc_l[j]
                    print cudafunc.registers

                    # fill constant memory
                    #cudafunc_call_l += [(driver.memcpy_htod, (const, fb_sub.data))]
                    #cudafunc_call_l += [(fill_const, (j, iz, oz))]
                    cudafunc_call_l += [(fill_const_nocache, (j, iz, oz))]
                        
                    # compute
                    #cudafunc.prepared_call(grid2, garr_in_ptr, garr_out_ptr)
                    cudafunc_call_l += [(cudafunc.prepared_call, (grid2, garr_in_ptr, garr_out_ptr))]

                # -
            # -
        # -
        # - private XXX
        self._cudafunc_call_l = cudafunc_call_l
        self._mod = mod
        self._cudafunc_l = cudafunc_l
        self._tex = tex
        self._const = const

    # -------------------------------------------------------------------------
    def __call__(self, **plugin_args):
        start = driver.Event()
        end = driver.Event()
        start.record()
        [func(*args) for func, args in self._cudafunc_call_l]         
        end.record()
        end.synchronize()
        return end.time_since(start)*1e-3

    # -------------------------------------------------------------------------
    def __call__2(self, **plugin_args):

        fb_ = self.fb_

        # original shapes
        fb_n, fb_h, fb_w, fb_d = fb_.nfilters, fb_.height, fb_.width, fb_.depth
        #out_h, out_w, out_d = out_.height, out_.width, out_.depth
        
        garr_in_l = self.in_._garr_l
        garr_out_l = self.out_._garr_l

        # -- prepare texture, constant, etc.
        threads = self._threads
        grid = self._grid
        mod = self._mod
        tex = self._tex
        const = self._const

        cudafunc_l = self._cudafunc_l

        # -- go!
        start = driver.Event()
        end = driver.Event()
        start.record()
        grid2 = grid[:2]
        for oz, garr_out in enumerate(garr_out_l):

            # clear hout 
            #garr_out.fill(0)
            # get gpu data pointer
            garr_out_ptr = garr_out.gpudata
            
            for iz, garr_in in enumerate(garr_in_l):

                # bind input texture and get gpu data pointer
                garr_in.bind_to_texref(tex)
                garr_in_ptr = garr_in.gpudata
                                         
                for j in xrange(fb_h):

                    # get kernel
                    cudafunc = cudafunc_l[j]

                    # fill constant memory
                    fb_sub = np.swapaxes(fb_[oz*4:(oz+1)*4,j,:,iz*4:(iz+1)*4], 0, 2)
                    fb_sub = np.ascontiguousarray(fb_sub)
                    driver.memcpy_htod(const, fb_sub.data)

                    # compute
                    cudafunc.prepared_call(grid2, garr_in_ptr, garr_out_ptr)

                # -
            # -
        # -
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


# ------------------------------------------------------------------------------
import hashlib
def get_lib_fname(base, topts, copts):
    strval = repr(topts)+repr(copts)
    hashval = hashlib.sha1(strval).hexdigest()
    return "%s__%s" % (base,hashval)

# ------------------------------------------------------------------------------
import subprocess
def execute(cmd): # pragma: no cover
    p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    out,err = p.communicate()
    if p.returncode:
        raise Exception, err
    return out, err

