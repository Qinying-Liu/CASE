import json
import urllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge16/api.py'


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i, :], candidate_segments)

    return tiou


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib.Request(api_url)
    response = urllib.urlopen(req)
    return json.loads(response.read())


class ANETEval(object):
    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None, proposal_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 max_avg_nr_proposals=None,
                 subset='validation', verbose=False,
                 check_status=True,
                 mode=('proposal', 'prediction')):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename and 'prediction' in mode:
            raise IOError('Please input a valid prediction file.')
        if not proposal_filename and 'proposal' in mode:
            raise IOError('Please input a valid proposal file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        self.mode = mode
        self.max_avg_nr_proposals = max_avg_nr_proposals
        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        if 'proposal' in mode:
            self.proposal = self._import_proposal(proposal_filename)
        if 'prediction' in mode:
            self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            if 'prediction' in mode:
                nr_pred = len(self.prediction)
                print('\tNumber of predictions: {}'.format(nr_pred))
            if 'proposal' in mode:
                nr_pred = len(self.proposal)
                print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        # if not all([field in data.keys() for field in self.gt_fields]):
        #     raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _import_proposal(self, proposal_filename):
        """Reads proposal file, checks if it is well formatted, and returns
           the proposal instances.
        Parameters
        ----------
        proposal_filename : str
            Full path to the proposal json file.
        Outputs
        -------
        proposal : df
            Data frame containing the proposal instances.
        """
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid proposal file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                score_lst.append(result['score'])
        proposal = pd.DataFrame({'video-id': video_lst,
                                 't-start': t_start_lst,
                                 't-end': t_end_lst,
                                 'score': score_lst})
        return proposal

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # print('Warning: No predictions of label \'%s\' were provided.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate_detection(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet detection task.')
            print('\tAverage-mAP: {}'.format(self.average_mAP))
            print('\tAP values:', self.mAP)

        return self.average_mAP, self.mAP

    def evaluate_proposal(self):
        recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
            self.ground_truth, self.proposal,
            max_avg_nr_proposals=self.max_avg_nr_proposals,
            tiou_thresholds=self.tiou_thresholds)

        area_under_curve = np.trapz(avg_recall, proposals_per_video)

        auc = 100. * float(area_under_curve) / proposals_per_video[-1]

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet proposal task.')
            print('\tArea Under the AR vs AN curve: {}%'.format(auc))

        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video
        return auc

    # def plot_metric(self, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    #
    #     average_nr_proposals = self.proposals_per_video
    #     recall = self.recall
    #     average_recall = self.avg_recall
    #
    #     fn_size = 14
    #     plt.figure(num=None, figsize=(12, 8))
    #     ax = plt.subplot(1, 1, 1)
    #
    #     colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    #     area_under_curve = np.zeros_like(tiou_thresholds)
    #     for i in range(recall.shape[0]):
    #         area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)
    #
    #     for idx, tiou in enumerate(tiou_thresholds[::2]):
    #         ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
    #                 label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
    #                 linewidth=4, linestyle='--', marker=None)
    #     # Plots Average Recall vs Average number of proposals.
    #     ax.plot(average_nr_proposals, average_recall, color=colors[0],
    #             label="tiou = 0.5:0.05:0.95," + " area=" + str(
    #                 int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
    #             linewidth=4, linestyle='-', marker=None)
    #
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    #
    #     plt.ylabel('Average Recall', fontsize=fn_size)
    #     plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    #     plt.grid(b=True, which="both")
    #     plt.ylim([0, 1.0])
    #     plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    #     plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #     # plt.show()
    #     # plt.savefig('data/output.jpg')
    #     plt.savefig('/data/liuqy/ASL/outputs/thumos_ASL_2_baseline_1/output.jpg')

    def plot_metric(self, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

        average_nr_proposals = self.proposals_per_video
        # recall = self.recall
        average_recall = self.avg_recall

        fn_size = 14
        plt.figure(num=None, figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)

        colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
        # area_under_curve = np.zeros_like(tiou_thresholds)
        # for i in range(recall.shape[0]):
        #     area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)
        #
        # for idx, tiou in enumerate(tiou_thresholds[::2]):
        #     ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
        #             label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
        #             linewidth=4, linestyle='--', marker=None)
        # Plots Average Recall vs Average number of proposals.
        ax.plot(average_nr_proposals, average_recall, color=colors[0],
                label="tiou = 0.5:0.05:0.95," + " area=" + str(
                    int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
                linewidth=4, linestyle='-', marker=None)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

        plt.ylabel('Average Recall', fontsize=fn_size)
        plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
        plt.grid(b=True, which="both")
        plt.ylim([0, 1.0])
        plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
        plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
        # plt.show()
        # plt.savefig('data/output.jpg')
        plt.savefig('/data/liuqy/ASL/outputs/thumos_ASL_2_baseline_multi_1/output.jpg')


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap


def average_recall_vs_avg_nr_proposals(ground_truth, proposals,
                                       max_avg_nr_proposals=None,
                                       tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Computes the average recall given an average number
        of proposals per video.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    proposal : df
        Data frame containing the proposal instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        array with tiou thresholds.

    Outputs
    -------
    recall : 2darray
        recall[i,j] is recall at ith tiou threshold at the jth average number of average number of proposals per video.
    average_recall : 1darray
        recall averaged over a list of tiou threshold. This is equivalent to recall.mean(axis=0).
    proposals_per_video : 1darray
        average number of proposals per video.
    """

    # Get list of videos.
    video_lst = ground_truth['video-id'].unique()

    if not max_avg_nr_proposals:
        max_avg_nr_proposals = float(proposals.shape[0]) / video_lst.shape[0]

    ratio = max_avg_nr_proposals * float(video_lst.shape[0]) / proposals.shape[0]

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    proposals_gbvn = proposals.groupby('video-id')

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_nr_proposals = 0
    for videoid in tqdm(video_lst):
        # Get ground-truth instances associated to this video.
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth = ground_truth_videoid.loc[:, ['t-start', 't-end']].values

        # Get proposals for this video.
        try:
            proposals_videoid = proposals_gbvn.get_group(videoid)
        except:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        this_video_proposals = proposals_videoid.loc[:, ['t-start', 't-end']].values

        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        # Sort proposals by score.
        sort_idx = proposals_videoid['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(this_video_ground_truth, axis=0)

        nr_proposals = np.minimum(int(this_video_proposals.shape[0] * ratio), this_video_proposals.shape[0])
        total_nr_proposals += nr_proposals
        this_video_proposals = this_video_proposals[:nr_proposals, :]

        # Compute tiou scores.
        tiou = wrapper_segment_iou(this_video_proposals, this_video_ground_truth)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 101) / 100.0 * (max_avg_nr_proposals * float(video_lst.shape[0]) / total_nr_proposals)
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tqdm(tiou_thresholds)):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total re   trieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum tiou threshold.
            true_positives_tiou = score >= tiou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum((score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

            for j, nr_proposals in enumerate(pcn_proposals):
                # Compute the number of matches for each percentage of the proposals
                matches[i, j] = np.count_nonzero((true_positives_tiou[:, :nr_proposals]).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(total_nr_proposals) / video_lst.shape[0])

    return recall, avg_recall, proposals_per_video
