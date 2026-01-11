import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.fft import fft, ifft, fftfreq

# ---------------- 设置中文字体 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 读取数据 -------------------
data = pd.read_excel("附件1.xlsx")
sigma = data.iloc[:, 0].values
R_exp = data.iloc[:, 1].values / 100.0

theta_deg = 10
theta = np.deg2rad(theta_deg)

# ------------------- Sellmeier 色散公式 -------------------
def n_of_sigma_sellmeier(sigma):
    lam_cm = 1.0 / sigma
    lam_um = lam_cm * 1e4
    lam2 = lam_um**2
    numerator = 5.5394 * lam2
    denominator = lam2 - 0.026945
    n2 = 1.0 + numerator / denominator
    return np.sqrt(n2)

# ------------------- FFT 检测 -------------------
def check_multibeam(sigma, R):
    R_centered = R - np.mean(R)
    N = len(R_centered)
    fft_vals = np.abs(fft(R_centered))
    freqs = fftfreq(N, d=(sigma[1]-sigma[0]))
    fft_peak = np.max(fft_vals[1:N//2])
    return fft_peak, freqs[:N//2], fft_vals[:N//2]

fft_peak, freqs, fft_vals = check_multibeam(sigma, R_exp)
print(f"FFT peak magnitude: {fft_peak:.2f}")
if fft_peak > 0.01:
    print("检测到多光束干涉，需消除干涉影响")
else:
    print("未检测到明显多光束干涉")

# ------------------- 去干涉 -------------------
R_smooth = savgol_filter(R_exp, window_length=51, polyorder=3)

R_centered = R_exp - np.mean(R_exp)
N = len(R_centered)
fft_vals_full = fft(R_centered)
freqs_full = fftfreq(N, d=(sigma[1]-sigma[0]))
fft_filtered = fft_vals_full.copy()

# 改进阈值：保留低频，避免M型问题
freq_cut = 0.0035
fft_filtered[np.abs(freqs_full) > freq_cut] = 0
R_fft_filtered = np.real(ifft(fft_filtered)) + np.mean(R_exp)

# ------------------- TMM 模型 -------------------
def tmm_model(d_cm):
    k0 = 2*np.pi*sigma
    n_air = 1.0
    n_film = n_of_sigma_sellmeier(sigma)
    n_sub = 2.6
    theta_film = np.arcsin(n_air*np.sin(theta)/n_film)
    theta_sub = np.arcsin(n_air*np.sin(theta)/n_sub)

    def r_s(n1, n2, th1, th2):
        return (n1*np.cos(th1) - n2*np.cos(th2)) / (n1*np.cos(th1) + n2*np.cos(th2))

    r01 = r_s(n_air, n_film, theta, theta_film)
    r12 = r_s(n_film, n_sub, theta_film, theta_sub)
    delta = k0 * n_film * d_cm * np.cos(theta_film)
    r_total = (r01 + r12*np.exp(2j*delta)) / (1 + r01*r12*np.exp(2j*delta))
    return np.abs(r_total)**2

# ------------------- 拟合函数 -------------------
def fit_thickness(R_input):
    def residual(d_array):
        d = d_array[0]
        return tmm_model(d) - R_input

    d0 = np.array([9.1451e-4])
    bounds = (1e-5, 1e-3)
    res = least_squares(residual, d0, bounds=bounds)
    d_fit = res.x[0]
    residual_std = np.std(res.fun)
    J = res.jac
    cov = np.linalg.inv(J.T @ J) * residual_std**2
    d_uncertainty = np.sqrt(np.diag(cov))[0]
    chi2 = np.sum(res.fun**2) / len(R_input)  # 归一化卡方
    return d_fit, d_uncertainty, chi2

# ------------------- 三种情况拟合 -------------------
d_raw, d_raw_unc, chi2_raw = fit_thickness(R_exp)
d_smooth, d_smooth_unc, chi2_smooth = fit_thickness(R_smooth)
d_fft, d_fft_unc, chi2_fft = fit_thickness(R_fft_filtered)

print("\n厚度拟合结果 (Sellmeier 色散公式, s偏振, χ²归一化):")
print(f"原始数据: d = {d_raw*1e4:.4f} ± {d_raw_unc*1e4:.4f} μm, χ² = {chi2_raw:.4f}")
print(f"平滑消干涉: d = {d_smooth*1e4:.4f} ± {d_smooth_unc*1e4:.4f} μm, χ² = {chi2_smooth:.4f}")
print(f"FFT滤波消干涉: d = {d_fft*1e4:.4f} ± {d_fft_unc*1e4:.4f} μm, χ² = {chi2_fft:.4f}")

# ------------------- 反射率谱对比 -------------------
plt.figure(figsize=(10,5))
plt.plot(sigma, R_exp, label="实验")
plt.plot(sigma, R_smooth, label="平滑消干涉")
plt.plot(sigma, R_fft_filtered, label="FFT滤波消干涉")
plt.xlabel("波数 (cm⁻¹)")
plt.ylabel("反射率")
plt.title("碳化硅反射率谱 - 干涉消除对比")
plt.legend()
plt.show()

# ------------------- FFT 频谱 -------------------
plt.figure(figsize=(10,5))
plt.plot(freqs_full[:N//2], np.abs(fft_vals_full)[:N//2], color="purple")
plt.xlabel("频率 (cm)")
plt.ylabel("FFT 幅值")
plt.title("FFT 频谱分析 (碳化硅)")
plt.grid(True)
plt.show()
