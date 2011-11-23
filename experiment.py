"""
For a given patience in obtaining each plan,
how many gigaflops can you get on average from a particular problem space?
"""

import sys
import cPickle

import numpy

import wisdom
from hyperopt.ht_dist2 import one_of, rSON2
import fbconv3_cuda

# XXX : should the average GFLOP/S be measured by dividing by trials or by
#       time? (Should it be more important to tune the more expensive calls? I
#       think yes)

def problem_generator():
    # TODO: sample fbcorr parameters from within LFW models
    space = rSON2(
            "nimgs" , 1, #one_of(1, 2, 4, 8, 16),
            "isize" , one_of(8, 13, 27, 28, 32, 81, 100, 121, 160, 200, 256),
            "depth" , one_of(1, 4, 8, 16, 32, 64), # XXX: 3 for rgb
            "nfilters" , one_of(1, 4, 8, 16, 32, 64), # must be 1 or 4k
            "fsize" , one_of(3, 5, 7, 9, 11),
            )
    while True:
        s = space.sample(rng=numpy.random)  # new seed on every execution
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

try:
    wdb = cPickle.load(open(sys.argv[1]))
except (IOError, EOFError):
    wdb = wisdom.Wisdom()

patience = 10  # seconds
for i, prob_spec in zip(range(5), problem_generator()):
    print prob_spec
    smart_op_spec = prob_spec.plan(patience=patience,
            wisdom=wdb,
            verbose=1)
    smart_speed = prob_spec.measure_speed(smart_op_spec,
            n_warmups=2, n_runs=3)

    ref_op_spec = wisdom.reference_op_spec()
    ref_speed = prob_spec.measure_speed(ref_op_spec,
            n_warmups=2, n_runs=3)

    # XXX: also consider taking max over N random op_specs as stiffer
    #      competition
    for ii in xrange(50):
        try:
            random_op_spec = wisdom.random_op_spec(numpy.random)
            random_speed = prob_spec.measure_speed(random_op_spec,
                    n_warmups=2, n_runs=3)
            break
        except fbconv3_cuda.InvalidConfig:
            random_speed = 0
            continue
    print 'SMART:', smart_speed, 'REFERENCE', ref_speed, 'RANDOM:', random_speed

cPickle.dump(wdb, open(sys.argv[1], 'w'))


