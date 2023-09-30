import time
import argparse
import numpy as np
from eval.eval_detection import ANETEval


def evaluate(gtfile, predfile, propfile=None, max_avg_nr_proposals=100,
             tiou_thresholds=np.linspace(0.5, 0.95, 10),
             subset='validation', verbose=False, check_status=False, plot=False):
    """
    :param gtfile: path to gt.json
    :param predfile: path to prediction.json
    :param propfile: None (if you want to calculate AUC, pass proposal_file.json and call anet_eval.evaluate_proposal)
    :param max_avg_nr_proposals:
    :param tiou_thresholds:
    :param subset: 'validation' for ANet and 'test' for Thumos
    :param verbose:
    :param check_status:
    :param plot: Not used
    :return:
    """
    start = time.time()
    mode = ['prediction']
    anet_eval = ANETEval(gtfile, predfile, propfile, tiou_thresholds=tiou_thresholds,
                         max_avg_nr_proposals=max_avg_nr_proposals, subset=subset,
                         verbose=verbose, check_status=check_status, mode=mode)

    mean_ap: float = 0.0
    class_ap = []
    mean_ap, class_ap = anet_eval.evaluate_detection()

    ap = anet_eval.ap.mean(-1)

    end = time.time()
    return mean_ap, class_ap, ap


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('--gtfile', type=str, required=True, help='Full path to json file containing the ground truth.')
    p.add_argument('--predfile', type=str, default=None, help='Full path to json file containing the predictions.')
    p.add_argument('--propfile', type=str, default=None, help='Full path to json file containing the proposals.')
    p.add_argument('--subset', default='validation', help='String indicating subset to evaluate: training, validation')
    p.add_argument('--tiou_thresholds', type=float, default=np.linspace(0.5, 0.95, 10), help='Temporal iou threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=False)
    return p.parse_args()


def main():
    args = parse_input()
    evaluate(args.gtfile, args.predfile, args.propfile, subset=args.subset)


if __name__ == '__main__':
    main()
