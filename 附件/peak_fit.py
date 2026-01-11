# single_beam_peak_fit_sellmeier.py

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 用户设置 ---
sheet1 = "附件1波峰"    # θ = 10°
sheet2 = "附件2波峰"    # θ = 15°
peak_file = "波峰数据.xlsx"

theta1_deg = 10.0
theta2_deg = 15.0
theta1 = np.deg2rad(theta1_deg)
theta2 = np.deg2rad(theta2_deg)

m0_window = 200  # 枚举 m_ref ± 窗口
continuous_inits = [50e-4]  # 初始 d 值 (cm)，例如 50 µm

d_min = 1e-4  # 1 µm
d_max = 1e-2  # 100 µm

# --- helpers ---
def read_peak_sheet(filename, sheet):
    df = pd.read_excel(filename, sheet_name=sheet)
    if '波数' in df.columns:
        sigma = df['波数'].values.astype(float)
    else:
        sigma = df.iloc[:,0].values.astype(float)
    return np.sort(sigma)

def n_of_sigma_sellmeier(sigma):
    lam_cm = 1.0 / sigma             # 波长 (cm)
    lam_um = lam_cm * 1e4            # 转 µm
    lam2 = lam_um**2
    numerator = 5.5394 * lam2
    denominator = lam2 - 0.026945
    n2 = 1.0 + numerator / denominator
    return np.sqrt(n2)

def residuals_continuous(x, all_sigma, all_theta, all_m):
    d_cm = x[0]
    n_vals = n_of_sigma_sellmeier(all_sigma)
    pred = 2.0 * n_vals * d_cm * np.cos(all_theta) * all_sigma
    return pred - all_m

# --- main ---
if __name__ == "__main__":
    if not os.path.exists(peak_file):
        raise FileNotFoundError(f"未找到 {peak_file}")

    sigma1 = read_peak_sheet(peak_file, sheet1)
    sigma2 = read_peak_sheet(peak_file, sheet2)
    print(f"附件1峰数：{len(sigma1)}, 附件2峰数：{len(sigma2)}")

    def median_spacing(sig):
        ds = np.diff(sig)
        ds = ds[(ds>0)&(ds<np.percentile(ds,95))]
        return np.median(ds) if len(ds)>0 else np.nan

    med1, med2 = median_spacing(sigma1), median_spacing(sigma2)
    print("Δσ 中位数 (10°):", med1, " (15°):", med2)

    nd1 = 1.0/(2*med1*np.cos(theta1)) if not np.isnan(med1) else None
    print("approx nd:", nd1)

    idx_ref1, idx_ref2 = len(sigma1)//2, len(sigma2)//2
    sig_ref1, sig_ref2 = sigma1[idx_ref1], sigma2[idx_ref2]
    n_guess = n_of_sigma_sellmeier(sig_ref1)
    d_guess = nd1 / n_guess if nd1 else 50e-4
    m_ref1 = int(round(2*n_guess*d_guess*np.cos(theta1)*sig_ref1))
    m_ref2 = int(round(2*n_guess*d_guess*np.cos(theta2)*sig_ref2))
    print("估算 m_ref:", m_ref1, m_ref2)

    best = {'loss': np.inf}

    all_sigma_template = np.concatenate([sigma1, sigma2])
    all_theta_template = np.concatenate([np.full_like(sigma1, theta1), np.full_like(sigma2, theta2)])

    for m0_1 in range(m_ref1-m0_window, m_ref1+m0_window+1):
        m_seq1 = m0_1 + np.arange(len(sigma1)) - idx_ref1
        if np.any(m_seq1<=0): continue
        for m0_2 in range(m_ref2-m0_window, m_ref2+m0_window+1):
            m_seq2 = m0_2 + np.arange(len(sigma2)) - idx_ref2
            if np.any(m_seq2<=0): continue
            all_m = np.concatenate([m_seq1, m_seq2])

            for d0 in continuous_inits:
                x0 = np.array([d0])
                lb, ub = [d_min], [d_max]
                try:
                    res = least_squares(residuals_continuous, x0, bounds=(lb,ub), args=(all_sigma_template, all_theta_template, all_m))
                except:
                    continue

                loss = np.sum(res.fun**2)
                if loss < best['loss']:
                    best.update({
                        'loss':loss, 'm0_1':m0_1, 'm0_2':m0_2, 'd_cm':res.x[0],
                        'sigma':all_sigma_template, 'theta':all_theta_template, 'm':all_m
                    })

    if best['loss']==np.inf:
        raise RuntimeError("未找到拟合解，请放宽搜索条件")

    # 计算厚度不确定度
    res_best = least_squares(
        residuals_continuous,
        x0=[best['d_cm']],
        bounds=([d_min],[d_max]),
        args=(best['sigma'], best['theta'], best['m'])
    )

    J = res_best.jac
    residuals = res_best.fun
    N = len(residuals)
    p = len(res_best.x)
    cov = np.linalg.inv(J.T @ J) * np.sum(residuals**2) / (N - p)
    d_uncertainty = np.sqrt(cov[0,0])

    print(f"厚度 d = {best['d_cm']*1e4:.4f} ± {d_uncertainty*1e4:.4f} µm")


    print("\n===== 拟合结果 (Sellmeier 模型) =====")
    print(f"m0(10°)={best['m0_1']}, m0(15°)={best['m0_2']}")
    print(f"厚度 d = {best['d_cm']*1e4:.4f} µm, loss = {best['loss']:.4e}")

    pred_LHS = 2.0 * n_of_sigma_sellmeier(best['sigma']) * best['d_cm'] * np.cos(best['theta']) * best['sigma']
    plt.figure(figsize=(8,6))
    plt.scatter(best['m'], pred_LHS, s=10)
    plt.plot(best['m'], best['m'], 'k--')
    plt.xlabel('整数 m')
    plt.ylabel('2 n(σ) d cosr σ')
    plt.title('Sellmeier 模型拟合检验')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fit_sellmeier_check.png", dpi=300)
    plt.show()

    # ===== 额外绘制 n-lambda 曲线 =====
    # 波长范围：2.5 µm ~ 25 µm
    lam_um = np.linspace(2.5, 25, 500)     # 波长 (µm)
    sigma = 1e4 / lam_um                   # 转换为波数 (cm^-1)

    n_vals = n_of_sigma_sellmeier(sigma)

    plt.figure(figsize=(8,6))
    plt.plot(lam_um, n_vals, 'b-', linewidth=2)
    plt.xlabel("波长 λ (µm)")
    plt.ylabel("折射率 n(λ)")
    plt.title("碳化硅 Sellmeier 模型折射率色散曲线")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("n_lambda.png", dpi=300)
    plt.show()
