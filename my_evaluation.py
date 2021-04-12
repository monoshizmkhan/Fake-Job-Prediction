import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    def __init__(self, predictions, actuals, pred_proba=None):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        self.confusion_matrix = {}
        for label in self.classes_:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i in range(len(self.actuals)):
                if self.actuals[i]==label and self.predictions[i]==label:
                    tp+=1
                elif self.actuals[i]==label and self.predictions[i]!=label:
                    fn+=1
                elif self.actuals[i]!=label and self.predictions[i]==label:
                    fp+=1
                elif self.actuals[i]!=label and self.predictions[i]!=label:
                    tn+=1
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


    def precision(self, target=None, average = "macro"):
        if self.confusion_matrix==None:
            self.confusion()
        prec = 0
        if target in self.classes_:
            TP = self.confusion_matrix[target]["TP"]
            FP = self.confusion_matrix[target]["FP"]
            if TP+FP!=0:
                prec = TP/(TP+FP)
        else:
            if average=="micro":
                TP = 0
                FP = 0
                for label in self.classes_:
                    TP+=self.confusion_matrix[label]["TP"]
                    FP+=self.confusion_matrix[label]["FP"]
                if TP+FP!=0:
                    prec = TP / (TP + FP)
            elif average=="macro":
                ratio = len(self.classes_)
                for label in self.classes_:
                    prec_for_one_label = 0
                    TP=self.confusion_matrix[label]["TP"]
                    FP=self.confusion_matrix[label]["FP"]
                    if TP+FP!=0:
                        prec_for_one_label = TP / (TP + FP)
                    prec+=(prec_for_one_label/ratio)
            elif average=="weighted":
                cnt = Counter(self.actuals)
                for label in self.classes_:
                    prec_for_one_label = 0
                    ratio = cnt[label]/len(self.actuals)
                    TP = self.confusion_matrix[label]["TP"]
                    FP = self.confusion_matrix[label]["FP"]
                    if TP + FP != 0:
                        prec_for_one_label = TP / (TP + FP)
                    prec += (prec_for_one_label * ratio)
        return prec


    def recall(self, target=None, average = "macro"):
        if self.confusion_matrix==None:
            self.confusion()
        rec = 0
        if target in self.classes_:
            TP = self.confusion_matrix[target]["TP"]
            FN = self.confusion_matrix[target]["FN"]
            if TP + FN != 0:
                rec = TP / (TP + FN)
        else:
            if average == "micro":
                TP = 0
                FN = 0
                for label in self.classes_:
                    TP += self.confusion_matrix[label]["TP"]
                    FN += self.confusion_matrix[label]["FN"]
                if TP + FN != 0:
                    rec = TP / (TP + FN)
            elif average == "macro":
                ratio = len(self.classes_)
                for label in self.classes_:
                    rec_for_one_label = 0
                    TP = self.confusion_matrix[label]["TP"]
                    FN = self.confusion_matrix[label]["FN"]
                    if TP + FN != 0:
                        rec_for_one_label = TP / (TP + FN)
                    rec += (rec_for_one_label / ratio)
            elif average == "weighted":
                cnt = Counter(self.actuals)
                for label in self.classes_:
                    rec_for_one_label = 0
                    ratio = cnt[label]/len(self.actuals)
                    TP = self.confusion_matrix[label]["TP"]
                    FN = self.confusion_matrix[label]["FN"]
                    if TP + FN != 0:
                        rec_for_one_label = TP / (TP + FN)
                    rec += (rec_for_one_label * ratio)
        return rec

    def f1(self, target=None, average = "macro"):
        prec = self.precision(target, average)
        rec = self.recall(target, average)
        f1_score = 0
        if prec+rec!=0:
            f1_score = (2*prec*rec)/(prec+rec)
        return f1_score


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        auc_target = 0
        if type(self.pred_proba) == type(None):
            return None
        return auc_target
