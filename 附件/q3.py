import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, fftfreq

# -------------------- 中文显示 --------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 读取 Excel --------------------
def read_excel_data(filename):
    df = pd.read_excel(filename)
    nu = df.iloc[:,0].values.astype(float)
    R_meas = df.iloc[:,1].values.astype(float)/100.0
    return nu, R_meas

# -------------------- 多层 TMM 稳定版 --------------------
def multilayer_tmm_stable(nu, n_list, d_list, k_list=None, decay=0.05):
    if k_list is None:
        k_list = [0.0]*len(n_list)
    nu = np.array(nu)
    lam = 1.0 / nu   # cm
    R_total = np.zeros_like(nu, dtype=float)

    for i, nu_i in enumerate(nu):
        lam_i = lam[i]
        n_layers = np.array(n_list) + 1j*np.array(k_list)
        M_s = np.eye(2, dtype=complex)
        M_p = np.eye(2, dtype=complex)

        for j in range(1, len(n_layers)-1):
            n_j = n_layers[j]
            d_j = d_list[j]
            delta = 2*np.pi*n_j*d_j/lam_i

            # s 偏振矩阵
            r01_s = (n_layers[j-1]-n_j)/(n_layers[j-1]+n_j)
            t01_s = 1 + r01_s
            P = np.array([[np.exp(-1j*delta), 0],[0, np.exp(1j*delta)]])
            M_layer_s = (1/t01_s) * np.array([[1, r01_s],[r01_s, 1]]) @ P
            M_layer_s /= np.max(np.abs(M_layer_s))  # 归一化
            M_s = M_s @ M_layer_s

            # p 偏振矩阵
            r01_p = (n_j - n_layers[j-1])/(n_j + n_layers[j-1])
            t01_p = 1 + r01_p
            M_layer_p = (1/t01_p) * np.array([[1, r01_p],[r01_p, 1]]) @ P
            M_layer_p /= np.max(np.abs(M_layer_p))
            M_p = M_p @ M_layer_p

        r_s = M_s[1,0]/M_s[0,0] * np.exp(-decay)
        r_p = M_p[1,0]/M_p[0,0] * np.exp(-decay)
        R_total[i] = (np.abs(r_s)**2 + np.abs(r_p)**2)/2

    return R_total.real

# -------------------- 拟合函数（多层） --------------------
def fit_multilayer_stable(params, nu, R_meas, n_base=3.42, decay=0.05):
    num_layers = len(params)//3
    n_list = [1.0]  # 空气
    d_list = [np.inf]
    k_list = [0.0]

    for i in range(num_layers):
        d_list.append(params[i*3])
        n_list.append(params[i*3+1])
        k_list.append(params[i*3+2])

    d_list.append(np.inf)
    n_list.append(n_base)
    k_list.append(0.0)

    R_model = multilayer_tmm_stable(nu, n_list, d_list, k_list, decay)
    return R_model - R_meas

# -------------------- FFT诊断 --------------------
def fft_diagnosis(nu, R_meas):
    R_smooth = savgol_filter(R_meas, 11, 3)
    R_zero_mean = R_smooth - np.mean(R_smooth)
    N = len(nu)
    dt = np.mean(np.diff(nu))
    freq = fftfreq(N, dt)
    fft_vals = np.abs(fft(R_zero_mean))
    plt.figure(figsize=(10,4))
    plt.plot(freq[:N//2], fft_vals[:N//2], color='purple')
    plt.xlabel("光程频率 (cm)")
    plt.ylabel("FFT幅值")
    plt.title("FFT诊断多光束干涉")
    plt.grid(True)
    plt.show()
    peaks,_ = find_peaks(fft_vals[:N//2], height=np.max(fft_vals[:N//2])*0.1)
    if len(peaks)>1:
        print(f"检测到 {len(peaks)} 个显著谐波，说明存在多光束干涉")
        return True
    else:
        print("未检测到显著多光束干涉")
        return False

# -------------------- 主程序 --------------------
if __name__=="__main__":
    filename = "附件3.xlsx"
    nu, R_meas = read_excel_data(filename)

    # FFT诊断
    multi_beam = fft_diagnosis(nu, R_meas)

    # 两层薄膜初值示例
    p0 = [0.0009, 3.42, 0.0, 0.0005, 2.8, 0.0]
    bounds_lower = [1e-6, 2.0, 0.0, 1e-6, 1.5, 0.0]
    bounds_upper = [0.01, 4.0, 50, 0.01, 4.0, 50]

    res = least_squares(fit_multilayer_stable, p0, bounds=(bounds_lower, bounds_upper),
                        args=(nu, R_meas, 3.42, 0.01))

    print("拟合结果（多层）:")
    for i in range(len(p0)//3):
        d_fit = res.x[i*3]*1e4  # μm
        n_fit = res.x[i*3+1]
        k_fit = res.x[i*3+2]
        print(f"层 {i+1}: d = {d_fit:.2f} μm, n = {n_fit:.3f}, k = {k_fit:.2f} cm^-1")

    # 绘制拟合曲线
    R_fit = multilayer_tmm_stable(nu,
                                  n_list=[1.0, res.x[1], res.x[4], 3.42],
                                  d_list=[np.inf, res.x[0], res.x[3], np.inf],
                                  k_list=[0.0, res.x[2], res.x[5], 0.0],
                                  decay=0.01)

    plt.figure(figsize=(10,5))
    plt.plot(nu, R_meas,label="实验数据")
    plt.plot(nu, R_fit,label="多层TMM拟合(稳定版)",linestyle='--')
    plt.xlabel("波数 (cm⁻¹)")
    plt.ylabel("反射率")
    plt.title("多层 TMM 干涉拟合")
    plt.grid(True)
    plt.legend()
    plt.show()
