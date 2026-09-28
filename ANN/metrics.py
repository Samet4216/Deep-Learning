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