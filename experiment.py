"""
For a given patience in obtaining each plan,
how many gigaflops can you get on average from a particular problem space?
"""

import cPickle
import logging
import sys
import time

import numpy

import wisdom
from hyperopt.ht_dist2 import one_of, rSON2
import fbconv3_cuda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# XXX : should the average GFLOP/S be measured by dividing by trials or by
#       time? (Should it be more important to tune the more expensive calls? I
#       think yes)

_context = [None]

def init_cuda():
    # Initialize CUDA
    import pycuda
    from pycuda import driver, gpuarray, compiler, tools
    driver.init()
    context = tools.make_default_context()
    device = context.get_device()
    #import atexit
    #atexit.register(context.pop)

    device_name = device.name().replace(' ', '_')

    print "=" * 80
    print "Using:", device_name
    print "=" * 80

    assert _context[0] is None
    _context[0] = context

    return context




def problem_generator(rng):
    # TODO: sample fbcorr parameters from within LFW models
    space = rSON2(
            "nimgs" , 1, #one_of(1, 2, 4, 8, 16),
            "isize" , one_of(32, 64, 96, 128, 160, 256, 512, 1024),
            "depth" , one_of(1, 4, 8, 16, 32, 64), # XXX: 3 for rgb
            "nfilters" , one_of(1, 4, 8, 16, 32, 64), # must be 1 or 4k
            "fsize" , one_of(3, 5, 7, 9, 11),
            )
    while True:
        s = space.sample(rng=rng)
        prob_spec = wisdom.ProblemSpec(
                n_imgs=s['nimgs'],
                height=s['isize'],
                width=s['isize'],
                depth=s['depth'],
                n_filters=s['nfilters'],
                filter_height=s['fsize'],
                filter_width=s['fsize'],
                img_strides=None,
                filter_strides=None,
                border_mode='valid')
        if prob_spec.gflops() > 100:
            # too big...
            continue
        if prob_spec.out_height <= 0:
            continue
        yield prob_spec

def main_step(ctxt):
    _python, _cmd, wisdomfile, patience = sys.argv

    try:
        wdb, results, rng = cPickle.load(open(wisdomfile))
    except (IOError, EOFError):
        wdb, results, rng = wisdom.Wisdom(), [], numpy.random.RandomState(2)

    try:
        for iii in xrange(100):
            pgen = problem_generator(rng)
            prob_spec = pgen.next()

            prob_spec = wisdom.ProblemSpec(n_imgs=1, height=128, width=128, depth=8, n_filters=4,
                    filter_height=7, filter_width=7,
                    img_strides=None,
                    filter_strides=None,
                    border_mode='valid')

            print prob_spec
            if len(wdb._observations) > 3 + getattr(wdb, '_dtree_n_obs', 0):
                wdb.build_dtree(force=True)
            print 'n_observations', len(wdb._observations)

            #
            # The strategy for updating the training set seems to be important.
            # Currently, what seems to work best is that the speed of the
            # suggestion from plan is ALWAYS fed back as a training example, but
            # the feedback of other suggestion mechanisms (ref, random, etc.) is
            # only fed into the training set if it's an improvement over the
            # current best suggestion. Therefore only errors are corrected, and
            # the training set stays "focused?"
            #  Not sure if this is good or not.

            #
            # XXX: how does pycuda's cache interact with this idea of a walltime
            # of patience?
            #

            op_specs = dict(
                    aaa=wisdom.OpSpec(block_w=128, block_h=8, n_filter_rows=1,
                        n_output4s='all',
                        spill=False,
                        imul_fast=True,
                        pad_shared=True, use_tex1dfetch=False,
                        maxrregcount=None,
                        use_fast_math=False),
                    #ref=wisdom.reference_op_spec(),
                    #quick=prob_spec.plan(patience=-1, wisdom=None, ctxt=ctxt,
                        #rng=rng),
                    #slow=prob_spec.plan(patience=float(patience), wisdom=None,
                        #ctxt=ctxt,
                        #rng=rng),
                    #wise=prob_spec.plan(patience=float(patience),
                        #wisdom=wdb, ctxt=ctxt,
                        #rng=rng ),
                    )
            finding = {}
            for k, op_spec in sorted(op_specs.items()):
                speed = prob_spec.measure_speed(op_spec,
                        n_warmups=2, n_runs=5, wisdom=wdb, ctxt=ctxt)
                finding[k] = speed

            print 'FINDING', finding
            results.append(finding)

    finally:
        if 0:
            ofile = open(wisdomfile, 'w')
            cPickle.dump((wdb, results, rng), ofile)
            ofile.close()
        else:
            print "NOT SAVING"

def main_dtree():
    _python, _cmd, wisdomfile = sys.argv
    wdb = cPickle.load(open(wisdomfile))
    wdb.build_dtree()
    cPickle.dump(wdb, open(wisdomfile, 'w'))


def main_fig1():
    _python, _cmd, = sys.argv[:2]
    import matplotlib.pyplot as plt
    for wisdomfiletag in sys.argv[2:]:
        wisdomfile, tag, c = wisdomfiletag.split(':')
        wdb, results, rng = cPickle.load(open(wisdomfile))
        y = [r[tag] / r['ref'] for r in results if r['ref'] > 0]
        ygood = [r[tag] / r['ref'] for r in results
                if r['ref'] * r[tag] > 0 and r['ref'] > 60]
        print numpy.log(ygood).mean()
        print numpy.log(ygood).std() / numpy.sqrt(len(ygood))
        plt.axhline(numpy.exp(numpy.log(ygood).mean()), c=c)
        plt.scatter(numpy.arange(len(y)), y, c=c)
    plt.xlabel('amount of training data')
    plt.ylabel('speed of dtree / speed of reference')
    plt.show()


if __name__ == '__main__':
    print "CALL LIKE THIS: python experiment.py step foo_debug.pkl 15"
    #ctxt = init_cuda()
    import pycuda.autoinit
    ctxt = pycuda.autoinit.context

    cmd = sys.argv[1]
    main = globals()['main_' + cmd]
    sys.exit(main(ctxt))
