"""
This module provides classes and functions for evaluating the performance of a
multi-class classification model in TensorFlow. It includes a custom metric
class called FPRNormal, which computes the false positive rate for a specific
class (in this case, the "normal" class). It also includes an Evaluation
class, which can compute various evaluation metrics. The module uses `numpy`,
`TensorFlow`, `Keras`, `TensorFlow Addons`, and `scikit-learn` libraries.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from sklearn.metrics import confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            precision_recall_fscore_support, \
                            matthews_corrcoef


class FPRNormal(keras.metrics.Metric):
    """
    Computes the false positive rate (FPR) for a specific class. This custom
    metric class inherits from the Keras `Metric` class and overrides the
    `update_state` and `result` methods.

    Attributes :
        normal_idx (tf.Tensor): A tensor containing the index of the normal
            class in the `labels` tensor.
        false_positives (tf.Variable): A variable representing the number of
            false positives seen so far.
        actual_negatives (tf.Variable): A variable representing the total
            number of actual negatives seen so far.

    Methods:
        update_state(y_true, y_pred, sample_weight=None):
            Accumulates statistics for computing the FPR.
        result():
            Computes the final FPR.
    """
    def __init__(self,
                 name='FPR_normal',
                 labels=tf.constant(['batceria', 'normal', 'virus']),
                 **kwargs):
        """
        Initializes a new instance of the FPRNormal class.

        Args:
            name (str): The name of the metric. Default is "FPR_normal".
            labels (tf.Tensor): A tensor containing the label names for each
                class. Default is ['bacteria', 'normal', 'virus'].
        """
        super(FPRNormal, self).__init__(name=name, **kwargs)
        self.normal_idx = tf.where(tf.equal(labels, 'normal'))
        self.false_positives = self.add_weight(name='fn',
                                               initializer='zeros')
        self.actual_negatives = self.add_weight(name='total',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates statistics for computing the FPR.

        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor): Optional weighting of individual
                samples. Default is None.

        Returns:
            None
        """
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        false_positives = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_pred, self.normal_idx),
                                   tf.not_equal(y_true, y_pred)),
                    dtype=tf.float32)
            )
        actual_negatives = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, self.normal_idx),
                    dtype=tf.float32)
            )
        self.false_positives.assign_add(false_positives)
        self.actual_negatives.assign_add(actual_negatives)

    def result(self):
        """
        Computes the final FPR.

        Returns:
            tf.Tensor: The FPR for the normal class.
        """
        return self.false_positives / self.actual_negatives


class Evaluation():
    """
    The Evaluation class provides methods to compute evaluation metrics and
    display the results. It also provides the loss function.

    Attributes:
        labels: A list of labels for the different classes.
        full_metrics: A dictionary containing the metrics to compute.
        loss_function: The loss function used in the model.
        training_metrics: A list of training metrics.

    Methods:

    __decode_one_hot(y_true, y_pred):
        A private method that decodes one-hot encoded true and predicted
        labels.
    compute_confusion_matrix(y_true, y_pred, display=False):
        A method that computes the confusion matrix for the true and
        predicted labels.
    get_training_metrics(metrics='BASE'):
        A method that returns the training metrics.
    compute_full_metrics(y_true_oh, y_pred_oh, sample_weight=None,
    display=False, digits=2):
        A method that computes the full evaluation metrics.
    """
    def __init__(self, labels=['batceria', 'normal', 'virus']):
        """
        Initializes a new instance of the Evaluation class.

        Args:
            labels (list): A list containing the label names for each
                class. Default is ['bacteria', 'normal', 'virus'].
        """
        self.labels = labels
        self.full_metrics = {
            'Precision': keras.metrics.Precision(name='Precision'),
            'Recall': keras.metrics.Recall(name='Recall'),
            'MCC': MatthewsCorrelationCoefficient(
                num_classes=3, name='Matthews_coef'
                ),
            'FPR normal': FPRNormal(labels=tf.constant(self.labels)),
        }
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
            )
        self.training_metrics = []

    def __decode_one_hot(self, y_true, y_pred):
        """
        Private method that decodes predictions and true labels from one-hot
        encoding.

        Args:
            y_true (numpy.ndarray): A numpy array representing the true
                labels.
            y_pred (numpy.ndarray): A numpy array representing the
                predicted labels.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy
                arrays: decoded y_true and decoded y_pred.
        """
        if y_true.shape[1] == len(self.labels):
            y_true = np.argmax(y_true, axis=1)
        if y_pred.shape[1] == len(self.labels):
            y_pred = np.argmax(y_pred, axis=1)
        return y_true, y_pred

    def compute_confusion_matrix(self, y_true, y_pred, display=False):
        """
        Computes the confusion matrix between true labels and predicted
        labels, and optionally displays it.

        Args:
            y_true (numpy.ndarray): A numpy array representing the true
                labels.
            y_pred (numpy.ndarray): A numpy array representing the
                predicted labels.
            display (bool, optional): Whether to display the confusion matrix.
                Defaults to False.

        Returns:
            numpy.ndarray: The confusion matrix as a numpy array of shape
                (n_classes, n_classes).
        """
        y_true, y_pred = self.__decode_one_hot(y_true, y_pred)

        conf_mx = confusion_matrix(y_true, y_pred)

        if display:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_mx,
                display_labels=self.labels
            )
        disp.plot()

        return conf_mx

    def get_training_metrics(self, metrics='BASE'):
        """
        Returns a list of training metrics to monitor during training.

        Args:
            metrics (str or list, optional): The metrics to include. Can be
                one of 'BASE', 'FULL', or a list of specific metrics. 'BASE'
                includes only the Matthews correlation coefficient (MCC),
                which is the default. 'FULL' includes all available metrics.
                A list of specific metrics can be passed as a list of strings.
                Defaults to 'BASE'.

        Returns:
            list: A list of the selected training metrics.
        """
        if metrics == 'BASE':
            self.training_metrics.append(self.full_metrics['MCC'])

        if metrics == 'FULL':
            self.training_metrics = [
                metric for metric in self.full_metrics.values()
            ]

        if isinstance(metrics, list):
            for metric in metrics:
                self.training_metrics.append(self.full_metrics[metric])

        return self.training_metrics

    def compute_full_metrics(self,
                             y_true_oh, y_pred_oh,
                             sample_weight=None,
                             display=False,
                             digits=2):
        """
        Computes and returns a dictionary of classification metrics for each
        class and global metrics.

        Args:
            y_true_oh (numpy.ndarray): A numpy array representing the true
                labels.
            y_pred_oh (numpy.ndarray): A numpy array representing the
                predicted labels.
            sample_weight (numpy.ndarray, optional): An array of weights for
                each sample. Defaults to None.
            display (bool, optional): Whether to display the metrics report.
                Defaults to False.
            digits (int, optional): The number of decimal digits to display.
                Defaults to 2.

        Returns:
            dict: A dictionary containing classification metrics for each
                class and global metrics.
        """
        y_true, y_pred = self.__decode_one_hot(y_true_oh, y_pred_oh)

        p, r, f1, s = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            sample_weight=sample_weight
        )
        mcc = []
        for idx, label in enumerate(self.labels):
            y_true_cls = [1 if y == idx else 0 for y in y_true]
            y_pred_cls = [1 if y == idx else 0 for y in y_pred]
            mcc.append(matthews_corrcoef(y_true_cls,
                                         y_pred_cls,
                                         sample_weight=sample_weight)
                       )
        mcc = np.array(mcc)

        headers_cls = [
            metric for metric in self.full_metrics.keys()
            if metric != 'FPR normal'
        ]
        headers_cls.append('Support')

        rows = list(zip(self.labels, p, r, mcc, s))
        metrics_cls_dict = {label[0]: label[1:] for label in rows}
        for label, scores in metrics_cls_dict.items():
            metrics_cls_dict[label] = dict(zip(headers_cls,
                                               [i.item() for i in scores]))

        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='weighted',
            sample_weight=sample_weight,
        )
        avg_mcc = matthews_corrcoef(y_true,
                                    y_pred,
                                    sample_weight=sample_weight
                                    )
        fpr_normal_metric = self.full_metrics['FPR normal']
        fpr_normal_metric.update_state(y_true_oh,
                                       y_pred_oh,
                                       sample_weight=sample_weight
                                       )
        fpr_normal = fpr_normal_metric.result().numpy()

        avg = [avg_p, avg_r, avg_mcc, fpr_normal]
        metrics_avg_dict = {
            head: value for head, value in zip(self.full_metrics.keys(), avg)
            }
        metrics_avg_dict['Support'] = np.sum(s)

        metrics_dict = {
            'classes': metrics_cls_dict,
            'global': metrics_avg_dict
        }
        if display:
            metric_width = max(len(name) for name in metrics_avg_dict.keys())
            label_width = max(len(label) for label in self.labels)
            width = max(metric_width, label_width)
            global_width = width + 1 + width * len(headers_cls)
            head_fmt = "{:>{width}s} " + " {:>9}" * len(headers_cls) + "\n"
            row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
            row_avg_fmt = "{:>{width}s} " + " {:>9}" * 3 + " {:>9.{digits}f}\n"
            report = "Metrics for each class".center(global_width, " ") + "\n"
            report += "-" * global_width + "\n"
            report += head_fmt.format("", *headers_cls, width=width)
            report += "\n"
            for row in rows:
                report += row_fmt.format(*row, width=width, digits=digits)
            report += "\n\n"
            report += "Global metrics".center(global_width, " ") + "\n"
            report += "-" * global_width + "\n"
            for name, value in metrics_avg_dict.items():
                if name == 'Support':
                    row_avg_fmt = "{:>{width}s} " + " {:>9}" * 3 + " {:>9}\n"
            report += row_avg_fmt.format(name,
                                         "",
                                         "",
                                         "",
                                         value,
                                         width=width,
                                         digits=digits
                                         )

            print(report)

        return metrics_dict


if __name__ == "__main__":
    pass
