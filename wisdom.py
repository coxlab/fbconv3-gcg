import time
import fbconv3_cuda
import scipy as sp
from hyperopt.ht_dist2 import one_of, rSON2
import pycuda._driver

RSEED = 123

class OpSpec(object):
    def __init__(self,
            block_w,
            block_h,
            n_filter_rows,
            n_output4s,
            spill,
            imul_fast,
            pad_shared,
            use_tex1dfetch,
            maxrregcount,
            use_fast_math,
            ):
        self.__dict__.update(locals())
        del self.self

    def FilterOp(self, imgs, filters, outs):
        return fbconv3_cuda.FilterOp(imgs, filters, outs,
                **self.__dict__)

def random_op_spec(rng):
    dct = rSON2(
        "block_w" , one_of(4, 8, 16, 32, 64, 128, 256, 512),
        "block_h" , one_of(4, 8, 16, 32, 64, 128, 256, 512),
        "n_filter_rows" , one_of(1, 2),
        "n_output4s" , one_of("all", 1, 2),
        "spill" , one_of(False, True),
        "imul_fast" , one_of(False, True),
        "pad_shared" , one_of(False, True),
        "use_tex1dfetch" , one_of(False, True),
        "maxrregcount" , one_of(None, 8, 16, 20, 24, 28, 32),
        "use_fast_math", one_of(False, True),
        ).sample(rng)
    return OpSpec(**dct)

def reference_op_spec():
    return OpSpec(block_w=8,
            block_h=8,
            n_filter_rows=1,
            n_output4s='all',
            spill=False,
            imul_fast=False,
            pad_shared=True,
            use_tex1dfetch=False,
            maxrregcount=None,
            use_fast_math=False,)


class ProblemSpec(object):
    def __init__(self,
            n_imgs,        # images processed at once
            height,        # image height
            width,         # image width
            depth,         # image depth
            n_filters,     # number of filters
            filter_height,       # filter height
            filter_width,        # filter width
            img_strides,   # how is image physically strided
            filter_strides,# how is filter physically strided
            border_mode,   # one of 'valid', 'full', 'same'
            ):
        self.__dict__.update(locals())
        del self.self
        if self.border_mode == 'valid':
            self.out_height = self.height - self.filter_height + 1
            self.out_width = self.width - self.filter_width + 1
        elif self.border_mode == 'full':
            self.out_height = self.height + self.filter_height - 1
            self.out_width = self.width + self.filter_width - 1
        elif self.border_mode == 'same':
            self.out_height = self.height
            self.out_width = self.width
        else:
            raise ValueError(self.border_mode)

        if n_imgs != 1:
            raise NotImplementedError()
        if border_mode != 'valid':
            raise NotImplementedError()

    def gflops(self):
        return (self.n_imgs
                * self.out_height * self.out_width * self.n_filters
                * self.filter_height * self.filter_width * self.depth
                * 2  # mul and add
                / (1000.**3.)) #return as giga float ops

    # relevant hand-designed features of problem specification
    def features():
        """
        all imgs c contiguous
        each img c contiguous
        all imgs f contiguous
        each img f contiguous
        all imgs size
        each img size
        img channel major
        img channel minor
        all filters c contiguous
        each filter c contiguous
        all filters f contiguous
        each filter f contiguous
        all filters size
        each filter size
        filter channel major
        filter channel minor
        filters flipped vertically (conv vs. corr)
        filters flipped horizontally (conv vs. corr)
        """
        raise NotImplementedError()

    def autotune_random(self, timeout=float('inf'), max_trials=100, n_warmups=2, n_runs=4):
        """
        Run a random search on FBConv3 bandit to find the best settings for
        this problem spec, return optimal FilterOp.
        """
        # basically call fbconv3_benchmark.benchmark_run with self as
        # argument.
        t_start = time.time()

        raise NotImplementedError()

    def plan(self, patience=0.1, wisdom=None, approx_n_uses=1000, verbose=0):
        """
        problem_spec - ProblemSpec instance
        patience - return a plan within this many seconds.
        wisdom - a Wisdom object (for reading and writing)
        approx_n_uses - estimated number of times this plan will be used (for budgeting search)

        Returns a FilterOp object
        """
        t_start = time.time()
        # -- start by getting something to return ASAP
        if wisdom is None:
            wisdom = Wisdom()

        candidates = wisdom.ranked_suggestions(self)
        encumbent = candidates[0]
        if (time.time() - t_start) >= patience:
            return encumbent

        def clock_candidate():
            try:
                return self.measure_speed(candidate,
                        n_warmups=2,
                        n_runs=3,
                        abort_thresh=encumbent_speed * 0.75)
            except fbconv3_cuda.InvalidConfig:
                return 0
            except pycuda._driver.LogicError:
                #XXX: cuModuleGetTexRef not found
                return 0

        if verbose > 0:
            print "Cycling through %i candidates from wisdom db" % (
                    len(candidates))

        encumbent_speed = self.measure_speed(encumbent, n_warmups=2, n_runs=3)
        for candidate in candidates[1:]:
            if (time.time() - t_start) >= patience:
                if verbose > 0:
                    print "Breaking at position %i" % (
                            candidates.index(candidate))
                return encumbent
            candidate_speed = clock_candidate()
            if candidate_speed > encumbent_speed:
                encumbent = candidate
                encumbent_speed = candidate_speed

        while (time.time() - t_start) < patience:
            candidate = random_op_spec(sp.random)
            candidate_speed = clock_candidate()
            if candidate_speed > 0:
                if verbose > 0:
                    print "Recording new candidate", candidate
                wisdom.record(self, candidate, candidate_speed)
                if candidate_speed > encumbent_speed:
                    encumbent = candidate
                    encumbent_speed = candidate_speed

        return encumbent


    def measure_speed(self, op_spec, n_warmups, n_runs, abort_thresh=None):
        """Return GFLOPS/S of op, not counting transfer times.

        abort_thresh - return 0 if the true value appears to be
                       lower than this (for early exit).
        """
        all_timings = {}
        gflop = self.gflops()

        img_shp = (self.height, self.width, self.depth)
        ker_shp = (self.n_filters, self.filter_height, self.filter_width,
                self.depth)
        out_shp = (self.out_height, self.out_width, self.n_filters)

        sp.random.seed(RSEED)
        in_data = sp.random.randn(*img_shp).astype('float32')
        fb_data = sp.random.randn(*ker_shp).astype( 'float32')
        out_data = sp.empty(out_shp, dtype='float32')

        timings = dict([(key, [])
                        for key in ('upload',
                                    #'set_up',
                                    'process',
                                    'cuda',
                                    'download',
                                    )])

        # XXX one image at a time
        in_ = fbconv3_cuda.Input(*img_shp)
        fb_ = fbconv3_cuda.Filterbank(*ker_shp)
        out_ = fbconv3_cuda.Output(*out_shp)

        # -- set-up operation (i.e. compilation)
        fb_[:] = 0
        fop = op_spec.FilterOp(in_, fb_, out_)

        for i in xrange(n_warmups + n_runs):

            # -- upload data
            start = time.time()
            in_[:] = in_data
            # XXX: is writing a 0 here important for correctness?
            out_[:] = 0
            fb_[:] = fb_data
            end = time.time()
            t_upload = end - start

            # -- process convolution
            # Note Bene: Filter != Conv
            start = time.time()
            t_cuda = fop()
            end = time.time()
            t_process = end - start

            if i > 0 and (gflop / t_cuda < abort_thresh):
                return 0

            start = time.time()
            out_data = out_[:]
            end = time.time()
            t_download = end - start

            if i >= n_warmups:
                timings['upload'] += [t_upload]
                timings['process'] += [t_process]
                timings['cuda'] += [t_cuda]
                timings['download'] += [t_download]

            if 0:
                gflops_cuda = gflop / t_cuda
                gflops_proc = gflop / t_process
                gflops_tot = gflop / (t_process+t_upload+t_download)
                print "gflops_cuda", gflops_cuda
                print "gflops_proc", gflops_proc
                print "gflops_tot", gflops_tot

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

        return gflops_cuda['min']

    def assert_correct(self, op):
        """raise assertion error if op() produces incorrect output
        """
        raise NotImplementedError()


class Wisdom(object):
    """
    Wisdom takes the form of a set of good kernels and a decision function for
    choosing between them.
    """

    def __init__(self):
        self._observations = []
        self._dtree = None

    def ranked_suggestions(self, prob_spec):
        if not self._dtree:
            return [reference_op_spec()]

        # 1. loop over all leaf nodes of self._dtree
        # 2. if leaf is consistent with prob_spec, then
        #    add the best (speed, op) from the leaf set
        # 3. sort all matching pairs by decreasing speed
        #    and return the ops of that list
        raise NotImplementedError()

    def record(self, prob_spec, op_spec, speed):
        self._observations.append((prob_spec, op_spec, speed))

    def build_dtree(self):
        raise NotImplementedError()

        """
         - distribution (empirical or analytic) of problem space
             (e.g. filter sizes, nfilt, etc.)
         - distribution (empirical or analytic) of configuration space
             (e.g. blocking, threading, spilling, unrolling, texturemem, regcount)


        1. sample some number N of problems X_1 ... X_N
        2. autotune for each problem X_i. Draw (or optimize) M configurations
            a_{m,n} and time them, measuring GLOPS y_{m,n}.

        Thus we produce a dataset of the form
        (X_1, (X_1, a_1, y_1), X_1

        """


