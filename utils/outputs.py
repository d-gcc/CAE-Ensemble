class OCOutput(object):
    def __init__(self, y_hat=None, rho=None, c=None, R=None, threshold=None, decision=None):
        self.y_hat = y_hat
        self.rho = rho
        self.c = c
        self.R = R
        self.threshold = threshold
        self.decision = decision


class VRAEOutput(object):
    def __init__(self, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,  best_pr_auc,
                 best_roc_auc, best_cks, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None):
        self.zs = zs
        self.z_infer_means = z_infer_means
        self.z_infer_stds = z_infer_stds
        self.decs = decs
        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class VAEOutput(object):
    def __init__(self, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,  best_pr_auc,
                 best_roc_auc, best_cks, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None):
        self.zs = zs
        self.z_infer_means = z_infer_means
        self.z_infer_stds = z_infer_stds
        self.decs = decs
        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class DAGMMOutput(object):
    def __init__(self, dec_means, energy, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.energy = energy
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class BEATGANOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class LOFOutput(object):
    def __init__(self, y_hat=None, negative_factor=None):
        self.y_hat = y_hat
        self.negative_factor = negative_factor


class ISFOutput(object):
    def __init__(self, y_hat=None, decision_function=None):
        self.y_hat = y_hat
        self.decision_function = decision_function


class OCSVMOutput(object):
    def __init__(self, y_hat=None, decision_function=None):
        self.y_hat = y_hat
        self.decision_function = decision_function


class AEOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class RAEOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class RNNVAEOutput(object):
    def __init__(self, dec_means, zs, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.zs = zs
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class CAEOutput(object):
    def __init__(self, dec_means, train_means, validation_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall,
                 best_fbeta, best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.train_means = train_means
        self.validation_means = validation_means
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class TFOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks):
        self.dec_means = dec_means
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks


class CBHOutput(object):
    def __init__(self, dec_means):
        self.dec_means = dec_means


class ARMAOutput(object):
    def __init__(self, dec_means):
        self.dec_means = dec_means

class RDAEOutput(object):
    def __init__(self, L, S, T_L, T_S):
        self.L = L
        self.S = S
        self.T_L = T_L
        self.T_S = T_S
