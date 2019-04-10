
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
    return [elem.filter.x[0], elem.filter.x[1]]


def run_test():
    """Run through training sets"""

    # Gets list of files in directory 
    files = os.listdir('../data/train')
    # for name in files:
    #     print(name)

    metric_strs = []

    # Create accumulators that will be updated during each frame
    # accumulators = {dataset:mm.MOTAccumulator(auto_id=True) for dataset in files}
    # for key,value in accumulators.items():
    #     print(value)
    acc = mm.MOTAccumulator(auto_id = True)
    mh = mm.metrics.create()



    # Run through each of the datasets. 
    for dataset in files:
        #print('Dataset: '+dataset)

        gtFile = pd.read_csv('../data/train/'+dataset+'/gt/gt.txt', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
             'x', 'y', 'z'])
        
        detFile = pd.read_csv('../data/train/'+dataset+'/det/det.txt', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
             'x', 'y', 'z'])


        num_frames = detFile['frame'][len(detFile)-1]
            
        detFileSorted = detFile.sort_values(by=['frame'])
        gtFileSorted = gtFile.sort_values(by=['frame', 'id'])

        #print(gtFileSorted)

        index = 0
        gt_index = 0
        #list of indexes for getting the correct rows.
        det_indexes = detFileSorted.index.values.tolist()
        gt_list = gtFileSorted.index.values.tolist()
        frames = []
        gt_frames = []

        #Only going through the first N frames of video
        for i in range(1, 600):
            detections = []
            # add all the detections for a single scan
            while(detFile['frame'][det_indexes[index]] == i):
                detections.append(np.array([detFile['bb_left'][det_indexes[index]], detFile['bb_top'][det_indexes[index]]]))
                index = index + 1
            frames.append(detections)

            #adding the ground truth to the it's own list
            objects = [] 
            while(gtFileSorted['frame'][gt_list[gt_index]] == i):
                objects.append([gtFileSorted['id'][gt_list[gt_index]], [gtFileSorted['bb_left'][gt_list[gt_index]], gtFileSorted['bb_top'][gt_list[gt_index]]]])
                gt_index = gt_index + 1

            #sort the by the frame ids 
            objects.sort()
            gt_frames.append(objects)

        #setup the tracker with cluster values
        tracker = mht.MHT(
            cparams=mht.ClusterParameters(k_max=100 ,nll_limit=10, hp_limit=10),
            matching_algorithm="naive",
            dbfile='sqlitedbs/mot_test_3_db.sqlite')        
        
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
            reports = {mht.Report(
                #np.random.multivariate_normal(t, np.diag([0.1, 0.1])),  # noqa
                # t[0:2],
                t,
                np.eye(2) * 0.001,
                mht.models.position_measurement,
                i)
                for i, t in enumerate(report)}
            
            this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
            tracker.register_scan(this_scan)
            tracker.predict(1)
            #Calculate the difference between the object related points
            #and the hypotheis related points

            #how to get the bb_left and bb_top from a track filter
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

            C = mm.distances.norm2squared_matrix(o, h, max_d2=1000.0)    

            #Get the ground truth objects
            gt_objects = []
            temp = []
            for f in gt_frames[k]:
                temp.append(f[0])
            
            gt_objects = temp

            temp = []
            for f in trs:
                temp.append(f.target._id)
            tracker_hyp = temp
            #print(tracker_hyp)

            #append the ground truth objects and tracker hyp for this frame
            frameid = acc.update(
                        gt_objects, #the ground truth objects
                        tracker_hyp, 
                        C # the distance matrix
                    )

            # if k <= 5:
                #print(list(tracker.global_hypotheses()))
                #print(h)
                #print(o)
                #print(C)
                #print(acc.mot_events.loc[frameid])

            k = k + 1
        

        #the summary of the metrics
        summary = mh.compute(
            acc, 
            metrics=mm.metrics.motchallenge_metrics,
            name=dataset,
        )

        strsummary = mm.io.render_summary(
            summary, 
            formatters=mh.formatters, 
            namemap=mm.io.motchallenge_metric_names
        )

        metric_strs.append(strsummary)
        metric_strs.append("\n\n")
        print(strsummary)

    file = open("mot_test_3.out.txt", "w")
    for summary in metric_strs:
        file.write(summary)

    file.close()

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

    if args.show:
        show_metrics()
    else:
        run_test()

if __name__ == '__main__':
    main(*sys.argv[1:])