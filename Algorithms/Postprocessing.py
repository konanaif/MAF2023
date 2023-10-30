from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt

import sys
sys.path.append("../")
import warnings

import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import MinMaxScaler

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric

from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification


class Calibrated_EqOdds(CalibratedEqOddsPostprocessing):
    def __init__(self, unprivileged_groups, privileged_groups, cost_constraint='weighted', seed=None):
        self.seed = seed
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.cost_constraint = cost_constraint
        if self.cost_constraint == 'fnr':
            self.fn_rate = 1
            self.fp_rate = 0
        elif self.cost_constraint == 'fpr':
            self.fn_rate = 0
            self.fp_rate = 1
        elif self.cost_constraint == 'weighted':
            self.fn_rate = 1
            self.fp_rate = 1
            
        self.base_rate_priv = 0.0
        self.base_rate_unpriv = 0.0
        
        super(CalibratedEqOddsPostprocessing, self).__init__(unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups,
                                                            cost_constraint=cost_constraint,
                                                            seed=seed)
        
class EqualizedOdds(EqOddsPostprocessing):
    def __init__(self, unprivileged_groups, privileged_groups, seed=None):
        self.seed = seed
        self.model_params = None
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        super(EqOddsPostprocessing, self).__init__(unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups,
                                                  seed=seed)
    
class RejectOption(RejectOptionClassification):
    def __init__(self, unprivileged_groups, privileged_groups, low_class_thresh=0.01, high_class_thresh=0.99, num_class_thresh=100, num_ROC_margin=50, metric_name="Statistical parity difference", metric_ub=0.05, metric_lb=-0.05):
        super(RejectOptionClassification, self).__init__(unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups,
                                                        low_class_thresh=low_class_thresh,
                                                        high_class_thresh=high_class_thresh,
                                                        num_class_thresh=num_class_thresh,
                                                        num_ROC_margin=num_ROC_margin,
                                                        metric_name=metric_name,
                                                        metric_ub=metric_ub,
                                                        metric_lb=metric_lb)
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        
        self.low_class_thresh = low_class_thresh
        self.high_class_thresh = high_class_thresh
        self.num_class_thresh = num_class_thresh
        self.num_ROC_margin = num_ROC_margin
        self.metric_name = metric_name
        self.metric_ub = metric_ub
        self.metric_lb = metric_lb
        
        self.classification_threshold = None
        self.ROC_margin = None
        
        