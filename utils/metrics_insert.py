import pymssql
import numpy as np
import pandas as pd
import math
import asyncio
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, auc, precision_recall_curve, roc_curve, fbeta_score, confusion_matrix
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, read_2D_dataset,read_ECG_dataset, get_loader, generate_synthetic_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset,rolling_window_2D, cutting_window_2D, unroll_window_3D, read_WADI_dataset, read_SWAT_dataset
from utils.metrics import calculate_average_metric, MetricsResult, zscore, create_label_based_on_zscore, create_label_based_on_quantile


class MetricsResult:
    def __init__(self, TN, FP, FN, TP, precision, recall, fbeta, pr_auc, roc_auc, cks):
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP
        self.precision = precision
        self.recall = recall
        self.fbeta = fbeta
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.cks = cks


def insert_SQL(metric_insert, model_name, pid, dataset, file_name, scenario, adjusted):
    insert_sql = """INSERT into metrics (model_name, pid, dataset, file_name, scenario, adjusted, TN, FP, FN, 
TP, precision, recall, fbeta, pr_auc, roc_auc, cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
'{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(model_name, pid, dataset, file_name, scenario, adjusted, 
    metric_insert.TN, metric_insert.FP, metric_insert.FN, metric_insert.TP, 
        metric_insert.precision, metric_insert.recall, metric_insert.fbeta, metric_insert.pr_auc, metric_insert.roc_auc, 
        metric_insert.cks)

    try:
        lines = []
        with open('MSSQL.txt') as f:
            lines = f.read().splitlines()

        connSQL = pymssql.connect(server=lines[0], user=lines[1], password=lines[2], database=lines[3])
        cursorSQL = connSQL.cursor()
        cursorSQL.execute(insert_sql)
        connSQL.commit()
        connSQL.close()
    except Exception as e:
        print("Results not inserted in MSSQL")


def regular_metrics(score, abnormal_segment, np_decision, pos_label):
    cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
    precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
    recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
    fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=1)
    pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), 
                                        probas_pred=np.nan_to_num(score), pos_label=pos_label)
    pr_auc = auc(re, pre)
    fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), 
                            y_score=np.nan_to_num(score), pos_label=pos_label)
    roc_auc = auc(fpr, tpr)       
    cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
    metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks)
    return metrics_result


def bestF1_metrics(score, abnormal_segment, pos_label, adjusted_points):
    _, _, thresholds = roc_curve(y_true=np.nan_to_num(abnormal_segment), 
                                    y_score=np.nan_to_num(score), pos_label=pos_label, drop_intermediate=True)
    
    metrics = np.array([['Thresholds', 'Precision', 'Recall', 'F1', 'CKS', 'TN', 'FP', 'FN', 'TP']])

    for threshold in thresholds[1:]:
        y_predicted = []
        for row in score:
            if row >= threshold:
                y_predicted.append(-1)
            else:
                y_predicted.append(1)   

        np_decision = pd.DataFrame(y_predicted, columns=['Score'])

        if adjusted_points:
            np_decision = adjusted_metrics(abnormal_segment, np_decision)

        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=1)
        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)

        metrics = np.concatenate((metrics, [[threshold, precision, recall, fbeta, cks, 
                                             cm[0][0], cm[0][1], cm[1][0], cm[1][1]]]), axis=0)

    top_f1 = pd.DataFrame(data=metrics[1:, 0:], columns=metrics[0,0:]).apply(pd.to_numeric).nlargest(1, ['F1'])
    metrics = pd.DataFrame(data=metrics[1:, 0:], columns=metrics[0,0:]).apply(pd.to_numeric)
    cm[0][0], cm[0][1] = top_f1['TN'].values[0], top_f1['FP'].values[0]
    cm[1][0], cm[1][1] = top_f1['FN'].values[0], top_f1['TP'].values[0]
    precision = top_f1['Precision'].values[0]
    recall = top_f1['Recall'].values[0]
    fbeta = top_f1['F1'].values[0]
    cks = top_f1['CKS'].values[0]

    tps = metrics['TP'].to_numpy()
    fps = metrics['FP'].to_numpy()

    tpsROC = np.r_[0, tps]
    fpsROC = np.r_[0, fps]

    if fpsROC[-1] <= 0:
        fpr = np.repeat(np.nan, fpsROC.shape)
    else:
        fpr = fpsROC / fpsROC[-1]

    if tpsROC[-1] <= 0:
        tpr = np.repeat(np.nan, tpsROC.shape)
    else:
        tpr = tpsROC / tpsROC[-1]

    roc_auc = auc(fpr,tpr)

    precision_SK = tps / (tps + fps)
    precision_SK[np.isnan(precision_SK)] = 0
    recall_SK = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    pr_auc = auc(np.r_[recall_SK[sl], 0], np.r_[precision_SK[sl], 1])
    
    metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks)
    
    return metrics_result


def at_k_metrics(score, abnormal_segment, pos_label, at_k, adjusted_points):
    y_predicted = []

    for row in score:
        y_predicted.append(1)

    np_decision = pd.DataFrame(y_predicted, columns=['Score']).iloc[0:abnormal_segment.shape[0],0]

    rangeTop = math.floor(-1 * len(score) * at_k / 100)
    indices = np.argpartition(score, rangeTop)[rangeTop:]
    np_decision[indices] = pos_label
    
    if adjusted_points:
        np_decision = adjusted_metrics(abnormal_segment, np_decision)

    return regular_metrics(score, abnormal_segment, np_decision, pos_label)


def adjusted_metrics(abnormal_segment, np_decision):
    label_groups = np.argwhere(np.diff(abnormal_segment.squeeze()))
    if len(label_groups)%2 != 0:
        label_groups = np.append(label_groups,len(abnormal_segment))

    anomaly_groups = label_groups.squeeze().reshape(-1, 2)

    predicted = pd.DataFrame(np_decision, columns=['Score'])

    for segment in anomaly_groups:
        predicted_segment = predicted.iloc[segment[0]+1:segment[1]+1]
        try:
            if predicted_segment['Score'].value_counts()[1] != segment[1] - segment[0]:
                predicted.iloc[segment[0]+1:segment[1]+1] = -1
        except:
            pass

    np_decision = predicted.to_numpy().squeeze()
    
    return np_decision

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def CalculateMetrics(abnormal_segment, score, pos_label, std_dev_values, quantile_values, k_values, model_name, pid, dataset, file_name):
    for adjusted_points in (False,True):
        # Z-Score
        for std_dev in std_dev_values:
            np_decision = create_label_based_on_zscore(zscore(score), std_dev, True)
            if adjusted_points:
                np_decision = adjusted_metrics(abnormal_segment, np_decision)

            zscore_metrics = regular_metrics(score, abnormal_segment, np_decision, pos_label)
            insert_SQL(zscore_metrics, model_name, pid, dataset, file_name, "Z-Score: " + str(std_dev) + " std dev", adjusted_points)
        
        # Quantile
        for quantile in quantile_values:
            np_decision = create_label_based_on_quantile(score, quantile=quantile)
            if adjusted_points:
                np_decision = adjusted_metrics(abnormal_segment, np_decision)

            quantile_metrics = regular_metrics(score, abnormal_segment, np_decision, pos_label)
            insert_SQL(quantile_metrics, model_name, pid, dataset, file_name, "Percentile: " + str(quantile), adjusted_points)
        
        # @K
        for at_k in k_values:
            k_metrics = at_k_metrics(score, abnormal_segment, pos_label, at_k, adjusted_points)
            insert_SQL(k_metrics, model_name, pid, dataset, file_name, "@K: " + str(at_k) + "%", adjusted_points)

        # Best F1      
        bestf1_metrics = bestF1_metrics(score, abnormal_segment, pos_label, adjusted_points)
        insert_SQL(bestf1_metrics, model_name, pid, dataset, file_name, "Best F1", adjusted_points)
