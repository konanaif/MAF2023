from aif360.metrics import DatasetMetric
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


class DataMetric:
    def __init__(self, dataset, privilege, unprivilege):
        self.dataset = dataset
        self.unprivilege = unprivilege
        self.privilege = privilege
        self.df = dataset.convert_to_dataframe()[0]
        
    def num_positive(self, privileged=None):
        labelnames = self.dataset.label_names
        
        df = self.df.copy()
        for ln in labelnames:
            df = df[ df[ln]==self.dataset.favorable_label ].copy()
        
        p_names = self.dataset.protected_attribute_names
        if privileged==True:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.privileged_protected_attributes[idx][0] ].copy()
        elif privileged==False:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.unprivileged_protected_attributes[idx][0] ].copy()
        
        return len(df)
        
    def num_negative(self, privileged=None):
        labelnames = self.dataset.label_names
        
        df = self.df.copy()
        for ln in labelnames:
            df = df[ df[ln]==self.dataset.unfavorable_label ].copy()
        
        p_names = self.dataset.protected_attribute_names
        if privileged==True:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.privileged_protected_attributes[idx][0] ].copy()
        elif privileged==False:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.unprivileged_protected_attributes[idx][0] ].copy()
                
        return len(df)
    
    def base_rate(self, privileged=None):
        df = self.df.copy()
        
        p_names = self.dataset.protected_attribute_names
        if privileged == True:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.privileged_protected_attributes[idx][0] ].copy()
        elif privileged == False:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.unprivileged_protected_attributes[idx][0] ].copy()
                
        n = len(df)
        p = self.num_positive(privileged=privileged)
        
        return p/n
    
    def disparate_impact(self):
        return self.base_rate(privileged=False) / self.base_rate(privileged=True)
    
    def statistical_parity_difference(self):
        return self.base_rate(privileged=False) - self.base_rate(privileged=True)
    
    def consistency(self, n_neighbors=5):
        r"""Individual fairness metric from [1]_ that measures how similar the
        labels are for similar instances.
        .. math::
           1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i -
           \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|
        Args:
            n_neighbors (int, optional): Number of neighbors for the knn
                computation.
        References:
            .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
               "Learning Fair Representations,"
               International Conference on Machine Learning, 2013.
        """

        X = self.dataset.features
        num_samples = X.shape[0]
        y = self.dataset.labels

        # learn a KNN on the features
        #nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
        nbrs = NearestNeighbors().fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        consistency = 1.0 - consistency/num_samples

        return consistency[0]
    
    def smoothed_base_rate(self, concentrate=1.0):
        num_classes = len(np.unique(self.dataset.labels))
        dirichlet_alpha = 1.0 / num_classes
        intersect_groups = list(self.df.groupby(self.dataset.protected_attribute_names).groups.keys())
        num_intersects = len(intersect_groups)

        # make counts total
        result = []
        for inter in intersect_groups:
            tdf = self.df.copy()

            # calculate total count
            for idx, value in enumerate(inter):
                att_name = self.dataset.protected_attribute_names[idx]
                #print(f"{att_name}: {value}")
                tdf = tdf[ tdf[att_name]==value ].copy()

            total = len(tdf)

            # calculate positive count
            for name in self.dataset.label_names:
                tdf = tdf[ tdf[name]==self.dataset.favorable_label ].copy()

            pos = len(tdf)
            result.append((pos + dirichlet_alpha) / (total + concentrate))
        
        return result
    
    def smoothed_empirical_differential_fairness(self, concentration=1.0):
        sbr = self.smoothed_base_rate(concentrate=concentration)
        
        def pos_ratio(i, j):
            return abs(np.log(sbr[i]) - np.log(sbr[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

        # overall DF of the mechanism
        return max(max(pos_ratio(i, j), neg_ratio(i, j))
                   for i in range(len(sbr)) for j in range(len(sbr)) if i != j)
    

class ClassificationMetric(DataMetric):
    def __init__(self, dataset, privilege, unprivilege, prediction_vector, target_label_name):
        super(ClassificationMetric, self).__init__(dataset, privilege, unprivilege)
        self.prediction_vector = prediction_vector
        self.target_label = target_label_name
        self.df = self.df.iloc[0:len(self.prediction_vector), :]
        self.conf_mat = self.confusion_matrix()
        self.performance = self.performance_measures()
        
    def confusion_matrix(self, privileged=None):
        # Get DataFrame
        df = self.df.copy()

        if not len(df) == len(self.prediction_vector):
            raise ValueError
        
        # Prediction Addition
        df['Prediction'] = self.prediction_vector
        
        p_names = self.dataset.protected_attribute_names
        if privileged == True:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.privileged_protected_attributes[idx][0] ].copy()
        elif privileged == False:
            for idx, pn in enumerate(p_names):
                df = df[ df[pn]==self.dataset.unprivileged_protected_attributes[idx][0] ].copy()

        # True prediction
        tdf = df[df[self.target_label] == df['Prediction']].copy()

        # True Positive
        tp = len(tdf[tdf[self.target_label] == self.dataset.favorable_label])
        # True Negative 
        tn = len(tdf[tdf[self.target_label] == self.dataset.unfavorable_label])

        # False prediction
        tdf = df[df[self.target_label] != df['Prediction']].copy()

        # False Positive
        fp = len(tdf[tdf[self.target_label] == self.dataset.unfavorable_label])
        # False Negative
        fn = len(tdf[tdf[self.target_label] == self.dataset.favorable_label])

        return dict(
            TP=tp,
            TN=tn,
            FP=fp,
            FN=fn)
    
    def performance_measures(self, privileged=None):
        conf_mat = self.confusion_matrix(privileged=privileged)
        
        tp = conf_mat['TP']
        tn = conf_mat['TN']
        fp = conf_mat['FP']
        fn = conf_mat['FN']
        
        p = self.num_positive(privileged=privileged)
        n = self.num_negative(privileged=privileged)

        return dict(
            TPR=tp / p if p > 0.0 else 0.0,
            TNR=tn / n if n > 0.0 else 0.0,
            FPR=fp / n if n > 0.0 else 0.0,
            FNR=fn / p if p > 0.0 else 0.0,
            PPV=tp / (tp+fp) if (tp+fp) > 0.0 else 0.0,
            NPV=tn / (tn+fn) if (tn+fn) > 0.0 else 0.0,
            FDR=fp / (fp+tp) if (fp+tp) > 0.0 else 0.0,
            FOR=fn / (fn+tn) if (fn+tn) > 0.0 else 0.0,
            ACC=(tp+tn) / (p+n) if (p+n) > 0.0 else 0.0)
    
    def error_rate(self):
        return 1 - self.performance['ACC']
    
    def average_odds_difference(self):
        PriPerfM = self.performance_measures(privileged=True)
        UnpriPerfM = self.performance_measures(privileged=False)
        
        diff_FPR = UnpriPerfM['FPR'] - PriPerfM['FPR']
        diff_TPR = UnpriPerfM['TPR'] - PriPerfM['TPR']
        
        return 0.5 * (diff_FPR + diff_TPR)
    
    def average_abs_odds_difference(self):
        PriPerfM = self.performance_measures(privileged=True)
        UnpriPerfM = self.performance_measures(privileged=False)
        
        diff_FPR = UnpriPerfM['FPR'] - PriPerfM['FPR']
        diff_TPR = UnpriPerfM['TPR'] - PriPerfM['TPR']
        
        return 0.5 * (np.abs(diff_FPR) + np.abs(diff_TPR))
    
    def selection_rate(self, privileged=None):
        conf_mat = self.confusion_matrix(privileged=privileged)
        
        num_pred_positives = conf_mat['TP'] + conf_mat['FP']
        num_instances = conf_mat['TP'] + conf_mat['FP'] + conf_mat['TN'] + conf_mat['FN']
        
        if num_instances == 0:
            return 0
        else:
            return num_pred_positives / num_instances
    
    def disparate_impact(self):
        denom = self.selection_rate(privileged=False)
        nom = self.selection_rate(privileged=True)
        if nom == 0:
            return 0
        else:
            return denom / nom
    
    def statistical_parity_difference(self):
        denom = self.selection_rate(privileged=False)
        nom = self.selection_rate(privileged=True)
        
        return denom - nom
    
    def generalized_entropy_index(self, alpha=2):
        pred_df = self.df.copy()
        pred_df['Prediction'] = self.prediction_vector

        y_pred = (pred_df['Prediction'] == self.dataset.favorable_label).to_numpy().astype(np.float64)
        y_true = (self.df[self.target_label] == self.dataset.favorable_label).to_numpy().astype(np.float64)
        b = 1 + y_pred - y_true

        if alpha == 1:
            # moving the b inside the log allows for 0 values
            result = np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
        elif alpha == 0:
            result = -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            result = np.mean((b / np.mean(b))**alpha - 1) / (alpha * (alpha - 1))

        return result
    
    def theil_index(self):
        r"""The Theil index is the :meth:`generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.generalized_entropy_index(alpha=1)

    def equal_opportunity_difference(self):
        r""":math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`
        """

        ## TPR_unprivileged
        perfm_unpriv = self.performance_measures(privileged=False)
        TPR_unpriv = perfm_unpriv['TPR']

        ## TPR_privileged
        perfm_priv = self.performance_measures(privileged=True)
        TPR_priv = perfm_priv['TPR']

        return TPR_unpriv - TPR_priv
