import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    res_sum_square =  np.sum((y_true - y_pred) ** 2) #resideual sum square
    y_mean = np.mean(y_true)
    total_sum_square = np.sum((y_true - y_mean) ** 2)
    """
        #ex y=[10,20,30] prediction=[12,18,35] 
        #resideual=> (10-12)^2  +  (20-18)^2  +  (30-35)^2
        #y_mean=> (10+20+30)/3= 20
        #total=> (10-20)^2  +  (20-20)^2  +  (30-20)^2 """

    if total_sum_square == 0:
        return 1.0 if res_sum_square == 0 else 0.0
    return 1.0 - (res_sum_square / total_sum_square)

def regression_report(y_true, y_pred):
    val_mse = mse(y_true, y_pred)
    val_rmse = rmse(y_true, y_pred)
    val_mae = mae(y_true, y_pred)
    val_r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"REGRESSİON REPORTS")
    print("=" * 50)
    print(f"  MSE : {val_mse:.4f}")
    print(f"  RMSE : {val_rmse:.4f}")
    print(f"  MAE : {val_mae:.4f}")
    print(f"  R^2 : %{val_r2 * 100:.2f} (Skor: {val_r2:.4f})")
    print("=" * 50)

if __name__ == "__main__":
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    regression_report(y_true, y_pred)

# ==================================================
# REGRESSİON REPORTS
# ==================================================
#   MSE : 11.0000
#   RMSE : 3.3166
#   MAE : 3.0000
#   R^2 : %83.50 (Skor: 0.8350)
# ==================================================

def confusion_matrix(y_true, y_pred, class_count):
    eps = 1e-15
    matrix = np.zeros((class_count, class_count), dtype=int) #dtype=int #create a confusion matrix initialized with zeros
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1 
    return matrix
"""
    Builds a confusion matrix by tallying (true, predicted) pairs.
    Rows (t) = true, Cols (p) = Predicted.
    
    Example: y_true = [0, 1, 1], y_pred = [0, 2, 1]
    - t=0, p=0 -> matrix[0, 0] += 1 (Hit)
    - t=1, p=2 -> matrix[1, 2] += 1 (Miss: Actual 1, Predicted 2)
    - t=1, p=1 -> matrix[1, 1] += 1 (Hit)
    """

def classification_report(y_true, y_pred, class_name=None):
    y_true = np.argmax(y_true, axist=1) #argmax returns the indices of the maximum values along an axis
    y_pred = np.argmax(y_pred, axis=1) # ex: 1 or 2 or 3 indices

    class_count = y_true.shape[0]
    if class_name is None: class_name = [f"Class {i}" for i in range(class_count)]
    #class1 class2 class3 class4 class5 ...
    """
                pred 0    pred 1    pred 2
true 0:    [    10,         2,          1    ]
true 1:    [     3,        20,          0    ]
true 2:    [     0,         4,         30    ]
    
    """
    conf_matrix = confusion_matrix(y_true, y_pred, class_count)
    TP = np.diag(conf_matrix) #True Positive #diagonal elements of the confusion matrix represent true positives for each class
    FP = np.sum(conf_matrix, axis=0) - TP #False Positive (2+20+4) full pred - diagonal (20)
    FN = np.sum(conf_matrix, axis=1) - TP #False Negative (10+3+0) full true - diagonal (10)
    TN = np.sum(conf_matrix) - (TP + FP + FN) #True Negative (10+20+30) full - (TP + FP + FN)

    precision = TP / (TP + FP + 1e-15)
    recall = TP / (TP + FN + 1e-15)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-15)

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_F1 = np.mean(F1)

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(f"{'Class Name':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for i, name in enumerate(class_name):
        print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {F1[i]:<12.4f}")
    print("-" * 55)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_F1:<12.4f}")
    print("=" * 55)
    print("\nCONFUSION MATRIX:")
    print(conf_matrix)
    print("=" * 55 + "\n")

    return conf_matrix, precision, recall, F1
