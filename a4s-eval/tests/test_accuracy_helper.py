
import numpy as np
from sklearn.metrics import confusion_matrix


def manual_accuracy_from_cm(cm):
    tp_tn = cm.trace()
    total = cm.sum()
    return tp_tn / total


def test_manual_accuracy():
    cm = np.array([[5, 2],[1, 12]])
    # manual: (5+12)/(5+2+1+12)=17/20=0.85
    assert abs(manual_accuracy_from_cm(cm) - 0.85) < 1e-6
