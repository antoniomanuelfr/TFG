from tfg_utils.MultipleLabelClassification.BaseMultipleLabelCC import BaseMultipleLabelCC
import tfg_utils.OrdinalClassification.OrdinalClassifiers as OC

class RFMultipleLabelCC(BaseMultipleLabelCC):
    def __init__(self, n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y):
        self.clf_ = OC.RandomForestOrdinalClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                           max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                           max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                                           min_impurity_decrease=self.min_impurity_decrease,
                                           bootstrap=self.bootstrap, oob_score=self.oob_score,
                                           n_jobs=self.n_jobs, random_state=self.random_state,
                                           verbose=self.verbose, warm_start=self.warm_start,
                                           class_weight=self.class_weight, ccp_alpha=self.ccp_alpha,
                                           max_samples=self.max_samples)
        super().fit(X, y)


class DTMultipleLabelCC(BaseMultipleLabelCC):
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        self.clf_ = OC.DecissionTreeOrdinalClassifier(criterion=self.criterion, splitter=self.splitter,
                                           max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                           max_features=self.max_features, random_state=self.random_state,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           min_impurity_decrease=self.min_impurity_decrease,
                                           class_weight=self.class_weight, ccp_alpha=self.ccp_alpha)
        super().fit(X, y)


class SVCMultipleLabelCC(BaseMultipleLabelCC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = True
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    def fit(self, X, y):
        self.clf_ = OC.OrdinalSVC(C=self.C, kernel=self.kernel, degree=self.degree,
                        gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking,
                        probability=self.probability, tol=self.tol, cache_size=self.cache_size,
                        class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
                        decision_function_shape=self.decision_function_shape,
                        break_ties=self.break_ties, random_state=self.random_state)
        super().fit(X, y)
