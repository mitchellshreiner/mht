
import motmetrics as mm
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# import cProfile

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import mht
import pandas as pd



def run_test():
    """Create plot."""

    gtFile = pd.read_csv('gt.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])
    
    detFile = pd.read_csv('det.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])

    index = 0;
    num_frames = detFile['frame'][len(detFile)-1]
    
    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    frames = []
    for i in range(1, 100):
        target = []
        while(detFile['frame'][index] == i):
            target.append(np.array([detFile['bb_left'][index], detFile['bb_top'][index]]))
            index = index + 1
        frames.append(target)

    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100, nll_limit=10, hp_limit=10),
        matching_algorithm="naive")        
    
    #the variables
    hyps = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    k = 1
    
    #for loop for the frames
    for report in frames:
        
        tracker.predict(1)
        reports = {mht.Report(
            #np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
            # t[0:2],
            t,
            np.eye(2) * 0.001,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(report)}
        
        this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
        tracker.register_scan(this_scan)
        
        if k <= 10:
           print(list(tracker.global_hypotheses()))
           
        k = k + 1
        
        #print the hypothesis
    
    # print(list(tracker.global_hypotheses()))
    # print(list(tracker.targets()))
        #append the information to the accumulator
    

def show_metrics():
    """List all the metrics for the tests"""
    mh = mm.metrics.create()
    print(mh.list_metrics_markdown())

def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    # cProfile.run('draw()', sort='tottime')
    # show_metrics()
    run_test()
    
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')

if __name__ == '__main__':
    main(*sys.argv[1:])