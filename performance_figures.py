import pylab
import numpy as np

def precision_and_recall_figure(precision_curve, recall_curve, model_name):
    ''' Draw Precision and Recall Figures
        Input : precision_curve - the precision values between 0-100% IOU thresholds
                recall_curve - the recall values between 0-100% IOU thresholds

        Output : precision.png - the precision figure saved in the working directory
                 recall.png - the recall figure saved in the working directory
                 f1_score.png - the f1 score figure that combines the precision and recall in one figure
                                and saves in the working directory
    '''
    # Plot the overall precision curve w.r.t iou threshold
    f_1 = pylab.figure(1)
    pylab.plot(precision_curve,label= model_name+' - AP '
               + str(round(precision_curve[50],2)) + ' - mAP '
               + str(round(np.mean(np.array(precision_curve[50:95])),2)))
    pylab.ylabel('Precision')
    pylab.xlabel('IOU Threshold')
    pylab.ylim(0, 1)
    pylab.xlim(0, 100)
    pylab.legend(loc='lower left')
    pylab.savefig('precision.png')

    # Plot the overall recall curve w.r.t iou threshold
    f_1 = pylab.figure(2)
    pylab.plot(recall_curve,label= model_name+' - AR '
               + str(round(recall_curve[50],2)) + ' - mAR '
               + str(round(np.mean(np.array(recall_curve[50:95])),2)))
    pylab.ylabel('Recall')
    pylab.xlabel('IOU Threshold')
    pylab.ylim(0, 1)
    pylab.xlim(0, 100)
    pylab.legend(loc='lower left')
    pylab.savefig('recall.png')

    # Plot the overall recall curve w.r.t iou threshold
    f_1 = pylab.figure(3)
    f1_score = precision_curve*recall_curve*2/(precision_curve+recall_curve+0.0001)
    print f1_score
    pylab.plot(f1_score,
                label= model_name+' - AF1 '
                + str(round(f1_score[50],2)) + ' - mAF1 '
                + str(round(np.mean(np.array(f1_score[50:95])),2)))
    pylab.ylabel('F1 Score')
    pylab.xlabel('IOU Threshold')
    pylab.ylim(0, 1)
    pylab.xlim(0, 100)
    pylab.legend(loc='lower left')
    pylab.savefig('f1_score.png')
