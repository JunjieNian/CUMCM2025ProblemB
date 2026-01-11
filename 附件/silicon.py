import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.fft import fft, fftfreq

# ---------------- 设置中文字体 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei']      # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号

# ------------------- 读取数据 -------------------
data = pd.read_excel("附件4.xlsx")
sigma = data.iloc[:, 0].values  # 波数 cm^-1
R_exp = data.iloc[:, 1].values / 100  # 百分比转0-1

# ------------------- 入射角 -------------------
theta_deg = 15
theta = np.deg2rad(theta_deg)

# ------------------- Sellmeier 硅晶圆经验公式 -------------------
def n_silicon(lambda_um):
    return np.sqrt(1 + 10.6684293*lambda_um**2/(lambda_um**2 - 0.301516485**2) + 
                   0.003043474*lambda_um**2/(lambda_um**2 - 1.13475115**2))

# ------------------- 多光束判断 (FFT) -------------------
def check_multibeam(sigma, R):
    R_centered = R - np.mean(R)
    N = len(R_centered)
    fft_vals = np.abs(fft(R_centered))
    freqs = fftfreq(N, d=(sigma[1]-sigma[0]))
    fft_peak = np.max(fft_vals[1:N//2])
    print(f"FFT peak magnitude: {fft_peak}")
    return fft_peak > 0.01, freqs[:N//2], fft_vals[:N//2]

multi_beam, freqs, fft_vals = check_multibeam(sigma, R_exp)
if multi_beam:
    print("检测到多光束干涉，优先使用无限反射 Airy 模型拟合")
else:
    print("未检测到明显多光束干涉，可简化模型拟合")

# ------------------- Airy 无限反射模型 -------------------
def R_model(d_cm):
    lambda_um = 1e4 / sigma
    n = n_silicon(lambda_um)
    n_complex = n
    n_2 = 3.42
    theta_r = np.arcsin(np.sin(theta) / n)
    r12 = (1 - n)/(1 + n)
    r23 = (n - 1)/(n + 1)
    delta = 4 * np.pi * n_complex * d_cm * np.cos(theta_r) * sigma
    delta = np.mod(delta, 2*np.pi)
    R = np.abs((r12 + r23 * np.exp(1j*delta)) / (1 + r12*r23*np.exp(1j*delta)))**2
    return R

# ------------------- 拟合函数 -------------------
def residual(d_array):
    d = d_array[0]
    R_fit = R_model(d)
    return R_fit - R_exp

# ------------------- 初值和边界 -------------------
d0 = np.array([5e-4])  # 初始厚度 5 μm -> 5e-4 cm
bounds = (1e-5, 1e-3)  # 厚度 0.1 μm ~ 10 μm

res = least_squares(residual, d0, bounds=bounds)
d_fit = res.x[0]

# ------------------- 不确定度计算 -------------------
residual_std = np.std(res.fun)
J = res.jac
cov = np.linalg.inv(J.T @ J) * residual_std**2
d_uncertainty = np.sqrt(np.diag(cov))[0]

print(f"拟合厚度 d = {d_fit*1e4:.4f} ± {d_uncertainty*1e4:.4f} μm")

# ------------------- 绘图 -------------------
plt.figure(figsize=(10,5))
plt.plot(sigma, R_exp, label="实验")
plt.plot(sigma, R_model(d_fit), label=f"拟合 d={d_fit*1e4:.4f} μm")
plt.xlabel("波数 (cm⁻¹)")
plt.ylabel("反射率")
plt.title("硅晶圆外延层反射率拟合")
plt.legend()
plt.show()

# ------------------- FFT 结果判定 -------------------
if multi_beam:
    print("FFT 分析结果：干涉条纹显著，多光束干涉成立。")
else:
    print("FFT 分析结果：干涉条纹不明显，可能为低阶干涉或噪声主导。")

# ------------------- 绘制 FFT 频谱 -------------------
plt.figure(figsize=(10,5))
plt.plot(freqs, fft_vals, color="purple")
plt.xlabel("频率 (cm)")
plt.ylabel("FFT 幅值")
plt.title("FFT 频谱分析")
plt.grid(True)
plt.show()

# ------------------- Bootstrap 抽样评估拟合稳定性 -------------------
n_bootstrap = 200   # 抽样次数
bootstrap_results = []

R_fit_best = R_model(d_fit)

np.random.seed(42)
for _ in range(n_bootstrap):
    # 在拟合残差上随机加噪声
    noise = np.random.choice(res.fun, size=len(res.fun), replace=True)
    R_bootstrap = R_fit_best + noise

    def residual_bootstrap(d_array):
        d = d_array[0]
        R_fit = R_model(d)
        return R_fit - R_bootstrap

    res_bs = least_squares(residual_bootstrap, d0, bounds=bounds)
    bootstrap_results.append(res_bs.x[0])

bootstrap_results = np.array(bootstrap_results)
d_mean = np.mean(bootstrap_results)
d_std = np.std(bootstrap_results)

print(f"Bootstrap 结果：平均厚度 d = {d_mean*1e4:.4f} μm，标准差 = {d_std*1e4:.4f} μm")

# 直方图展示
plt.figure(figsize=(8,4))
plt.hist(bootstrap_results*1e4, bins=20, color="skyblue", edgecolor="black")
plt.axvline(d_mean*1e4, color="red", linestyle="--", label=f"均值 {d_mean*1e4:.4f} μm")
plt.xlabel("厚度 (μm)")
plt.ylabel("频数")
plt.title("Bootstrap 拟合厚度分布")
plt.legend()
plt.show()
