import numpy as np
from mango.tuner import Tuner
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report,roc_auc_score


class bayesianOpt:

    def __init__(self,classifier):
        self.classifier = classifier
        self.optimize_results = None
       

    def _fit(self,X_train,y_train,params):
        """
        Internal Function for bayesian Optimization
        """
        # Initialize Classifier
        clf = self.classifier

        # Extract Related Params for Classifier
        validated_params = {}
        param_cls = clf.get_params().keys()
        for key in params:
            if key in param_cls:
                validated_params[key] = params[key]
            else:
                pass
        clf.set_params(**validated_params)
        
        return clf.fit(X_train,y_train)   
    
    def optimize(self,X_train,y_train,param_grid,conf_dict):
        """
        Optimization Function for classifier with data inputs and scoring function
        """
        def objective(args_list):
            accuracies = []
            for params in args_list:
                
                # Fit the model
                clf = self._fit(X_train,y_train,params)

                # Evaluate the model with cross validation
                accuracy = cross_validate(clf, X_train, y_train, cv=params["cv"], scoring=params["scoring"],n_jobs=params["n_jobs"])
                accuracies.append(np.mean(accuracy["test_score"]))

            return accuracies

        tuner_user = Tuner(param_grid, objective, conf_dict)
        optimize_results = tuner_user.maximize()
        self.optimize_results = optimize_results
        return optimize_results
    
    
    def optimize_fit(self,X_train,y_train,param_grid,conf_dict):
        """
        Optimization and Function for classifier with data inputs and scoring function
        Returns fitted object for the classifier
        """
        optimized_results = self.optimize(X_train,y_train,param_grid,conf_dict)
        return self._fit(X_train,y_train,optimized_results["best_params"])
