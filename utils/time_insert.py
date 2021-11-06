import pymssql
import numpy as np
import pandas as pd
import math
import asyncio
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, auc, precision_recall_curve, roc_curve, fbeta_score, confusion_matrix
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, read_2D_dataset,read_ECG_dataset, get_loader, generate_synthetic_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset,rolling_window_2D, cutting_window_2D, unroll_window_3D, read_WADI_dataset, read_SWAT_dataset
from utils.metrics import calculate_average_metric, MetricsResult, zscore, create_label_based_on_zscore, create_label_based_on_quantile

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def insert_time_SQL(model_name, pid, dataset, training_time, inference_time):
    insert_sql = """INSERT into metrics_time (model_name, pid, dataset, training_time, inference_time) VALUES('{}', '{}', '{}', '{}', '{}')""".format(model_name, pid, dataset, training_time, inference_time)

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