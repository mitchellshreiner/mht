
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


def takeID(elem):
    return elem._id

def getXY(elem):
    return [elem.filter.x[0], elem.filter.x[1], 0, 0]


def run_test():
    """Create plot."""

    gtFile = pd.read_csv('gt.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])
    
    detFile = pd.read_csv('det.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])


    num_frames = detFile['frame'][len(detFile)-1]
        
    gtFileSorted = gtFile.sort_values(by=['frame', 'id'])

    index = 0
    gt_index = 0
    #list of indexes for getting the correct rows.
    gt_list = gtFileSorted.index.values.tolist()

    #print(detFile)

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)
    mh = mm.metrics.create()

    frames = []
    gt_frames = []
    for i in range(1, 100):
        target = []
        while(detFile['frame'][index] == i):
            target.append(np.array([detFile['bb_left'][index], detFile['bb_top'][index], 1.0, 1.0]))
            index = index + 1
        frames.append(target)

        #adding the ground truth to the it's own list
        objects = [] 
        while(gtFileSorted['frame'][gt_list[gt_index]] == i):
            objects.append([gtFileSorted['id'][gt_list[gt_index]], [gtFileSorted['bb_left'][gt_list[gt_index]], gtFileSorted['bb_top'][gt_list[gt_index]], gtFileSorted['bb_width'][gt_list[gt_index]], gtFileSorted['bb_height'][gt_list[gt_index]]]])
            gt_index = gt_index + 1

        objects.sort()
        gt_frames.append(objects)

    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100 ,nll_limit=4, hp_limit=5),
        matching_algorithm="naive")        
    
    #the variables
    hyps = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    k = 0
    
    #for loop for the frames
    for report in frames:

        #tracker setup and increment
        tracker.predict(1)
        reports = {mht.Report(
            #np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
            # t[0:2],
            t[0:2],
            np.eye(2) * 0.001,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(report)}
        
        this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
        tracker.register_scan(this_scan)
        
        #Calculate the difference between the object related points
        #and the hypotheis related points

        #how to get the bb_left and bb_top from a track filter
        #list(tracker.global_hypotheses())[0].tracks[0].filter.x[0]
        #for tr in list(tracker.global_hypotheses()):
        trs = list(list(tracker.global_hypotheses())[0].tracks)
        trs.sort(key=takeID)

        h_trs = []
        for f in trs:
            h_trs.append(getXY(f))
        h = np.array(h_trs)

        gt_ar = gt_frames
        o_trs = []
        for f in gt_frames[k]:
            o_trs.append(f[1])
        o = np.array(o_trs)

        C = mm.distances.iou_matrix(o, h, max_iou=0.5)    

        #Get the ground truth objects
        gt_objects = []
        temp = []
        for f in gt_frames[k]:
            temp.append(f[0])
        
        gt_objects = temp

        temp = []
        for f in trs:
            temp.append(f._id)
        tracker_hyp = temp
        #print(tracker_hyp)

        frameid = acc.update(
                    gt_objects, #the ground truth objects
                    tracker_hyp, 
                    C # the distance matrix
                )


        if k == 5:
            # print(list(tracker.global_hypotheses()))
            
            print(h)
            print(o)
            print(C)
            print(acc.mot_events.loc[frameid])

            # print(list(tracker.targets()))
        #print(C)
        #print("Printing the h for calculating the difference between object points")
        #print(h) 
        #print("Printing the o")
        #print(o)

        k = k + 1
        
        #print the hypothesis
    
    # print(list(tracker.global_hypotheses()))
    # print(list(tracker.targets()))
        #append the information to the accumulator
    
    #the summary of the metrics
    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]], 
        metrics=mm.metrics.motchallenge_metrics, 
        names=['full', 'part'],
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

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