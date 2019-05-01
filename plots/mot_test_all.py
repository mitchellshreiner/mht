
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


def run_test(num_frames=100, match_alg_str="naive", debug=1):
    """Run through training sets"""

    # Gets list of files in directory 
    datasets = os.listdir('../data/train/')

    metric_strs = []

    # Create accumulators that will be updated during each frame
    # accumulators = {dataset:mm.MOTAccumulator(auto_id=True) for dataset in files}
    # for key,value in accumulators.items():
    #     print(value)
    acc = mm.MOTAccumulator(auto_id = True)
    mh = mm.metrics.create()
    
    #dataset = datasets[0]
    dataset = "MOT17-10-SDP"
    print(dataset)

    # Run through each of the datasets. 
    #for dataset in files:
        #print('Dataset: '+dataset)

    gtFile = pd.read_csv('../data/train/'+ dataset +'/gt/gt.txt', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'class', 'visibility'])
    
    detFile = pd.read_csv('../data/train/'+ dataset +'/det/det.txt', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])


    #num_frames = detFile['frame'][len(detFile)-1]
        
    detFileSorted = detFile.sort_values(by=['frame'])
    gtFileSorted = gtFile.sort_values(by=['frame', 'id'])

    center_x = 0.0 
    center_y = 0.0
    index = 0
    gt_index = 0
    #list of indexes for getting the correct rows.
    det_indexes = detFileSorted.index.values.tolist()
    gt_list = gtFileSorted.index.values.tolist()
    frames = []
    gt_frames = []

    #Only going through the first N frames of video
    for i in range(1, num_frames):
        detections = []

        # add all the detections for a single scan
        while(detFile['frame'][det_indexes[index]] == i):
            #find the center of the bounding box instead of taking just the top left.
            center_x = detFile['bb_left'][det_indexes[index]] + detFile['bb_width'][det_indexes[index]]/2
            center_y = detFile['bb_top'][det_indexes[index]] + detFile['bb_height'][det_indexes[index]]/2
            detections.append(np.array([center_x, center_y]))
            index = index + 1
        frames.append(detections)

        #adding the ground truth to the it's own list
        objects = [] 
        while(gtFileSorted['frame'][gt_list[gt_index]] == i):
            center_x = gtFileSorted['bb_left'][gt_list[gt_index]] + gtFileSorted['bb_width'][gt_list[gt_index]]/2
            center_y = gtFileSorted['bb_top'][gt_list[gt_index]] + gtFileSorted['bb_height'][gt_list[gt_index]]/2
            #[gtFileSorted['bb_left'][gt_list[gt_index]], gtFileSorted['bb_top'][gt_list[gt_index]]]            
            if( gtFileSorted['cnf'][gt_list[gt_index]] == 1): 
                objects.append([gtFileSorted['id'][gt_list[gt_index]], gtFileSorted['cnf'][gt_list[gt_index]], gtFileSorted['class'][gt_list[gt_index]], center_x, center_y])
            
            gt_index = gt_index + 1

        #sort the by the frame ids 
        objects.sort()
        gt_frames.append(objects)

    # print(frames)
    # print(gt_frames)

    # #stop loop
    # while(1):
    #     k = 1

    #setup the tracker with cluster values
    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100 ,nll_limit=10, hp_limit=5),
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
        print('Frame:',k)
        #tracker setup and increment
        reports = {mht.Report(
            #np.random.multivariate_normal(t, np.diag([0.1, 0.1])),  # noqa
            # t[0:2],
            t,
            np.eye(2) * 10.0,
            # np.ones((2,2)) * 20.0,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(report)}
        
        this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
        print('scan')
        tracker.register_scan(this_scan)
        print('predict')
        tracker.predict(1)
        #Calculate the difference between the object related points
        #and the hypotheis related points

        if debug == 1:
            print(' Number of Global Hypotheses')
            print(len(list(tracker.global_hypotheses())))
            print(' gt objects ')
            print(len(gt_frames[k]))
            # print('Targets')
            # print(list(tracker.global_hypotheses())[0].targets)
            # print('Num Targets tracker')
            # print(len(list(tracker.global_hypotheses())[0].targets))
            # print('Num detections in reports')
            # print(len(list(reports)))
            # print('Num Targets ground truth')
            # print(len(gt_frames[k]))


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
            o_trs.append(f[3:])
        o = np.array(o_trs)

        # print(o)
        # print(h)
        C = mm.distances.norm2squared_matrix(o, h, max_d2=200.0)    

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
        #     print(list(tracker.global_hypotheses()))
        #     # print(h)
        #     # print(o)
        #     print(acc.mot_events.loc[frameid])

        k = k + 1
    

        mht.plot.plot_scan(this_scan)
        plt.plot([t[0] for t in o_trs],
                 [t[1] for t in o_trs],
                 marker='D', color='y', alpha=.5, linestyle='None')
    mht.plot.plot_hyptrace(list(tracker.global_hypotheses())[0], covellipse=True)
    mht.plot.plt.axis([-1, 1920,  1080, -1])
    mht.plot.plt.ylabel('Y')
    mht.plot.plt.xlabel('X')
    # mht.plot.plt.figure()
    # mht.plot.plt.subplot(3, 1, 1)
    # mht.plot.plt.plot(nclusters)
    # mht.plot.plt.axis([-1, k + 1, min(nclusters) - 0.1, max(nclusters) + 0.1])
    # mht.plot.plt.ylabel('# Clusters')
    # mht.plot.plt.subplot(3, 1, 2)
    # mht.plot.plt.plot(ntargets, label='Estimate')
    # mht.plot.plt.plot(ntargets_true, label='True')
    # mht.plot.plt.ylabel('# Targets')
    # mht.plot.plt.legend()
    # mht.plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
    #                    max(ntargets + ntargets_true) + 0.1])
    # mht.plot.plt.subplot(3, 1, 3)
    # mht.plot.plt.plot(nhyps)
    # mht.plot.plt.axis([-1, k + 1, min(nhyps) - 0.1, max(nhyps) + 0.1])
    # mht.plot.plt.ylabel('# Hyps')

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

    # get string ready
    metric_strs.append(strsummary)
    metric_strs.append("\n\n")
    print(strsummary)

    # write the metrics to the specified file
    file = open("mot_.out.txt", "w")
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
    parser.add_argument('frames', help="type the number of frames you want to search through dataset", type=int)
    parser.add_argument('--show', action="store_true")
    #parser.add_argument('debug', type=int)
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    # cProfile.run('draw()', sort='tottime')

    if args.show:
        show_metrics()
        run_test(args.frames)
        plt.show()
    else:
        run_test(args.frames)

if __name__ == '__main__':
    main(*sys.argv[1:])