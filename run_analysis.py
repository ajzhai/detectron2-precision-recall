import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader

from example_dataset import get_balloon_dicts
from d2pr import ModifiedCOCOEvaluator, extract_pr_and_scores, plot_pr_and_scores


if __name__ == '__main__':
    # register dataset in detectron2
    eval_dataset_name = 'balloon_val'
    DatasetCatalog.register(eval_dataset_name, lambda : get_balloon_dicts("balloon/val"))
    MetadataCatalog.get(eval_dataset_name).set(thing_classes=["balloon"])
    # if your dataset is in COCO format, the above can be replaced by the following three lines:
    # from detectron2.data.datasets import register_coco_instances
    # register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
    # register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

    # load the model
    cfg = get_cfg()
    cfg.merge_from_file('./output/config.yaml')
    cfg.MODEL.WEIGHTS = './output/model_final.pth'
    predictor = DefaultPredictor(cfg)

    # evaluate on the dataset
    evaluator = ModifiedCOCOEvaluator(eval_dataset_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # set IoU threshold and class id of interest
    coco_eval = evaluator.coco_evals['bbox']
    iou_thr = 0.5
    class_id = 0

    # display some precision, recall, and score threshold values
    precisions, recalls, scores = extract_pr_and_scores(coco_eval, iou_thr, class_id)

    for r in np.arange(0, 1.01, 0.05):  # by default, recalls contains 0, 0.01, 0.02, ..., 0.99, 1.0
        r_idx = np.where(recalls >= r)[0][0]
        print('recall [%.2f] achieved with precision [%.2f] using score threshold [%.2f]' % 
              (recalls[r_idx], precisions[r_idx], scores[r_idx])) 
        
    # plot precision-recall curve
    ax = plot_pr_and_scores(coco_eval, iou_thr, class_id)
    plt.show()
    