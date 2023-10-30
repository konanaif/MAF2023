#from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover

#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()

from aif360.algorithms.inprocessing.gerryfair.auditor import *
from aif360.algorithms import Transformer
from aif360.algorithms.inprocessing.celisMeta import FalseDiscovery
from aif360.algorithms.inprocessing.celisMeta import StatisticalRate


#class Adversarial_Debiasing(AdversarialDebiasing):
#    def __init__(self, unprivileged_groups, privileged_groups, 
#                 scope_name, sess, seed=None, 
#                 adversary_loss_weight=0.1, num_epochs=50, batch_size=128, classifier_num_hidden_units=200, debias=True):
#        super(Adversarial_Debiasing, self).__init__(unprivileged_groups=unprivileged_groups,
#                                                   privileged_groups=privileged_groups,
#                                                   scope_name=scope_name,
#                                                   sess=sess,
#                                                   seed=seed,
#                                                   adversary_loss_weight=adversary_loss_weight,
#                                                   num_epochs=num_epochs,
#                                                   batch_size=batch_size,
#                                                   classifier_num_hidden_units=classifier_num_hidden_units,
#                                                    debias=debias)
        
class Gerry_Fair_Classifier(GerryFairClassifier):
    def __init__(self, C=10, printflag=False, heatmapflag=False, 
                 heatmap_iter=10, heatmap_path='.', max_iters=10, gamma=0.01, 
                 fairness_def='FP', predictor=linear_model.LinearRegression()):
        super(GerryFairClassifier, self).__init__(C=C, printflag=printflag,
                                                 heatmapflag=heatmapflag,
                                                 heatmap_iter=heatmap_iter,
                                                 heatmap_path=heatmap_path,
                                                 max_iters=max_iters,
                                                 gamma=gamma,
                                                 fairness_def=fairness_def,
                                                 predictor=predictor)
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        self.classifiers = None
        self.errors = None
        self.fairness_violations = None
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception(
                'This metric is not yet supported for learning. Metric specified: {}.'
                .format(self.fairness_def))
        
class Meta_Fair_Classifier(MetaFairClassifier):
    def __init__(self, tau=0.8, sensitive_attr="", type="fdr", seed=None):
        super(MetaFairClassifier, self).__init__(tau=tau, sensitive_attr=sensitive_attr, type=type, seed=seed)
        self.tau = tau
        self.sensitive_attr = sensitive_attr
        if type == "fdr":
            self.obj = FalseDiscovery()
        elif type == "sr":
            self.obj = StatisticalRate()
        else:
            raise NotImplementedError("Only 'fdr' and 'sr' are supported yet.")
        self.seed = seed
        
class Prejudice_Remover(PrejudiceRemover):
    def __init__(self, eta=1.0, sensitive_attr='', class_attr=''):
        super(PrejudiceRemover, self).__init__(eta=eta, sensitive_attr=sensitive_attr, class_attr=class_attr)
        self.eta = eta
        self.sensitive_attr = sensitive_attr
        self.class_attr = class_attr