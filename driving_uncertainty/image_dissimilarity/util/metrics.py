import numpy as np

def get_metrics(flat_labels, flat_pred, num_points=50):
    # From fishycapes code
    pos = flat_labels == 1
    valid = flat_labels <= 1  # filter out void
    gt = pos[valid]
    del pos
    uncertainty = flat_pred[valid].reshape(-1).astype(np.float32, copy=False)
    del valid
    
    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices
    
    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty
    
    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs
    
    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])
    
    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall
    
    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]
    
    # Compute TPR and FPR
    tpr = tps / tps[-1]
    del tps
    fpr = fps / fps[-1]
    del fps
    
    # Compute AUROC
    auroc = np.trapz(tpr, fpr)
    
    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
    results = {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        'recall': np.array(curve_recall),
        'precision': np.array(curve_precision),
        'fpr': fpr,
        'tpr': tpr
        }
    
    return results

