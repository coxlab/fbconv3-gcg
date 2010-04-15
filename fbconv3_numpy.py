""" XXX: doc """

__all__ = ['NumpyFilterOp']

import numpy as np
import weakref

from scipy import signal

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

    # -------------------------------------------------------------------------
    def __call__(self, **plugin_args):

        in_ = self.in_
        fb_ = self.fb_
        out_ = self.out_
        
        fb_n, fb_h, fb_w, fb_d = fb_.nfilters, fb_.height, fb_.width, fb_.depth
        out_h, out_w, out_d = out_.height, out_.width, out_.depth
        
        # -- handle multiple dot products
        if max(fb_h, fb_w) > 13:
            convolve = signal.fftconvolve
        else:
            convolve = signal.convolve
            
        # for each filter
        for i in xrange(fb_n):
            filt = fb_[i].ravel()
            filt = np.flipud(filt)
            filt = np.reshape(filt, (fb_h, fb_w, fb_d))
            result = convolve(in_[:], filt, 'valid')
            result = result.astype(out_.dtype)
            out = np.reshape(result, result.shape[0:2])
            out_[:, :, i] = out
    
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


        self._ndarray = np.ndarray((height, width, depth), dtype=dtype)

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

Output = Input

class Filterbank(Input):
    
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
    
