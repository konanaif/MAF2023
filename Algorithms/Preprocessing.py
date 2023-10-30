from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.preprocessing import Reweighing

class Disparate_Impact_Remover(DisparateImpactRemover):
    def __init__(self, rep_level=1.0, sensitive_attribute=''):
        super(Disparate_Impact_Remover, self).__init__(repair_level=rep_level, sensitive_attribute=sensitive_attribute)
        
        
class Learning_Fair_Representation(LFR):
    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 k=5,
                 Ax=0.01,
                 Ay=1.0,
                 Az=50.0,
                 print_interval=250,
                 verbose=0,
                 seed=None):
        super(Learning_Fair_Representation, self).__init__(unprivileged_groups=unprivileged_groups, 
                                                           privileged_groups=privileged_groups, 
                                                           k=k,
                                                           Ax=Ax, 
                                                           Ay=Ay, 
                                                           Az=Az,
                                                           print_interval=print_interval, 
                                                           verbose=verbose,
                                                           seed=seed)
        
        
class RW(Reweighing):
    def __init__(self, unprivileged_groups, privileged_groups):
        super(RW, self).__init__(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)