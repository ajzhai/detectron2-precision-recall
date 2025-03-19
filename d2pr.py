import itertools
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pycocotools.cocoeval import COCOeval
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco


class ModifiedCOCOEvaluator(COCOEvaluator):
    """
    (modified to expose COCOEval objects)
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        # we need to use official COCO API to get all the precisions
        self._use_fast_impl = False

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        self.coco_evals = {}
        
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            try:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        cocoeval_fn=COCOeval,  # use COCOeval not COCOeval_opt
                        img_ids=img_ids,
                        max_dets_per_image=self._max_dets_per_image,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            except TypeError as e:  # older versions of detectron2 has different args
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=False,  # use COCOeval not COCOeval_opt
                        img_ids=img_ids,
                        max_dets_per_image=self._max_dets_per_image,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

            # we expose the COCOEval objects here
            self.coco_evals[task] = coco_eval
   

def extract_pr_and_scores(coco_eval, iou_thr, class_id):
    """
    Extract precision, recall, and score threshold values for a single class.
    Args:
        coco_eval (COCOeval): coco evaluation object with desired class
        iou_thr (float): IoU threshold for correctness. should be a multiple of 0.05 between 0.5 and 0.95.
        class_id (int): class id
    Returns:
        tuple: (precision, recall, scores)
    """
    # extract precision, score threshold, and recall values
    precisions = coco_eval.eval["precision"]  # axes: iou thr, recall, class, area range, max dets
    scores = coco_eval.eval["scores"]
    recalls = coco_eval.eval["params"].recThrs

    # find correct indices
    iou_thr_idx = list(coco_eval.eval["params"].iouThrs).index(iou_thr)
    class_idx = list(coco_eval.eval["params"].catIds).index(class_id)
    precisions = precisions[iou_thr_idx, :, class_idx, 0, -1]
    scores = scores[iou_thr_idx, :, class_idx, 0, -1]

    return precisions, recalls, scores


def plot_pr_and_scores(coco_eval, iou_thr, class_id, 
                       xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), title=None, ax=None):
    """
    Plot Precision-Recall curve for a single class, along with corresponding score thresholds.
    Args:
        coco_eval (COCOeval): coco evaluation object with desired class
        iou_thr (float): IoU threshold for correctness. should be a multiple of 0.05 between 0.5 and 0.95.
        class_id (int): class id
        ax (matplotlib.axes.Axes): the axes to plot the PR curve. If not specified, a new axes will be created.
    Returns:
        matplotlib.axes.Axes: the axes object that contains the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    precisions, recalls, scores = extract_pr_and_scores(coco_eval, iou_thr, class_id)

    # plot precision-recall curve
    ax.plot(recalls, precisions, label="precision", color="C0")
    ax.plot(recalls, scores, label="score threshold", color="C1")

    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision/score threshold")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best")
    return ax
