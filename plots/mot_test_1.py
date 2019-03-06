
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

    df = pd.read_csv('det.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])

    index = 0;
    targets = []
    max_frame = df['frame'][len(df)-1]

    # for loop for going through all the frames from data
    for i in range(1, max_frame):
        target = []
        while(df['frame'][index] == i):
            target.append(np.array([df['bb_left'][index], df['bb_top'][index]]))
            index = index + 1
        targets.append(target)

    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100, nll_limit=20, hp_limit=20),
        matching_algorithm="naive")        
    
    #the variables for storing intermediate information for printing
    hyps = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    k = 1
    
    #for loop for the frames
    for report in targets:
        k = k + 1
        tracker.predict(1)
        reports = {mht.Report(
            t,
            np.eye(2) * 0.001,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(report)}
        
        this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
        tracker.register_scan(this_scan)
        hyps = list(tracker.global_hypotheses())
        nclusters.append(len(tracker.active_clusters))
        ntargets.append(len(hyps[0].targets))
        ntargets_true.append(len(report))
        nhyps.append(len(hyps))
        mht.plot.plot_scan(this_scan)
        plt.plot([t[0] for t in report],
                 [t[1] for t in report],
                 marker='D', color='y', alpha=.5, linestyle='None')
    mht.plot.plot_hyptrace(hyps[0], covellipse=True)
    mht.plot.plt.axis([-100, 2800, -100, 1700])
    mht.plot.plt.ylabel('Tracks')
    mht.plot.plt.figure()
    mht.plot.plt.subplot(3, 1, 1)
    mht.plot.plt.plot(nclusters)
    mht.plot.plt.axis([-1, k + 1, min(nclusters) - 0.1, max(nclusters) + 0.1])
    mht.plot.plt.ylabel('# Clusters')
    mht.plot.plt.subplot(3, 1, 2)
    mht.plot.plt.plot(ntargets, label='Estimate')
    mht.plot.plt.plot(ntargets_true, label='True')
    mht.plot.plt.ylabel('# Targets')
    mht.plot.plt.legend()
    mht.plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
                       max(ntargets + ntargets_true) + 0.1])
    mht.plot.plt.subplot(3, 1, 3)
    mht.plot.plt.plot(nhyps)
    mht.plot.plt.axis([-1, k + 1, min(nhyps) - 0.1, max(nhyps) + 0.1])
    mht.plot.plt.ylabel('# Hyps')        

def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    # cProfile.run('draw()', sort='tottime')
    run_test()
    
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')

if __name__ == '__main__':
    main(*sys.argv[1:])
