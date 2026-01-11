import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks, medfilt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ---------------- 设置中文字体 ----------------
plt.rcParams['font.sans-serif'] = ['SimHei']      # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号

# ----------- 读入数据 -----------
file1 = "附件1.xlsx"
file2 = "附件2.xlsx"

data1 = pd.read_excel(file1)
data2 = pd.read_excel(file2)
data1.columns = ["波数", "反射率"]
data2.columns = ["波数", "反射率"]

# ----------- 预处理函数 -----------
def preprocess(df, col="反射率"):
    y = df[col].values
    
    # 1. 中值滤波，去掉尖刺
    y = medfilt(y, kernel_size=5)
    
    # 2. Savitzky-Golay 滤波（轻度）
    y_smooth = savgol_filter(y, window_length=31, polyorder=3)
    
    # 3. 二级平滑（用于计算导数，抑制平稳区伪峰）
    y_smooth_strong = gaussian_filter1d(y_smooth, sigma=5)
    
    df[col + "_平滑"] = y_smooth
    df[col + "_强平滑"] = y_smooth_strong
    return df

data1 = preprocess(data1)
data2 = preprocess(data2)


# ----------- 只找波峰，含去重 -----------
def detect_peaks(x, y_smooth_strong, prominence=0.005, distance=30, width=20, merge_threshold=50):
    """
    x: 波数
    y_smooth_strong: 强平滑曲线
    prominence: 峰的显著性
    distance: 相邻峰的最小间隔
    width: 峰的最小宽度
    merge_threshold: 如果两个峰间距小于该值，合并为一个（保留高的）
    """
    peaks, properties = find_peaks(
        y_smooth_strong,
        prominence=prominence,
        distance=distance,
        width=width
    )

    # ---- 二次筛选：合并太近的峰 ----
    final_peaks = []
    if len(peaks) > 0:
        final_peaks = [peaks[0]]
        for p in peaks[1:]:
            if x[p] - x[final_peaks[-1]] < merge_threshold:
                # 两个峰太近 -> 保留更高的那个
                if y_smooth_strong[p] > y_smooth_strong[final_peaks[-1]]:
                    final_peaks[-1] = p
            else:
                final_peaks.append(p)
    return np.array(final_peaks)


# ----------- 绘图并保存函数 -----------
def plot_save_peaks(df, filename, title="光谱曲线"):
    x = df["波数"].values
    y_raw = df["反射率"].values
    y_smooth = df["反射率_平滑"].values
    y_smooth_strong = df["反射率_强平滑"].values

    peaks = detect_peaks(x, y_smooth_strong)

    # 绘图
    plt.figure(figsize=(10,6))
    plt.plot(x, y_raw, alpha=0.4, label="原始数据")
    plt.plot(x, y_smooth, label="平滑曲线", linewidth=2)
    plt.plot(x, y_smooth_strong, "--", label="强平滑曲线", alpha=0.7)
    plt.scatter(x[peaks], y_smooth_strong[peaks], color="r", marker="^", s=80, label="波峰")
    plt.xlabel("波数 (cm⁻¹)")
    plt.ylabel("反射率")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)   # 保存图片
    plt.show()

    # 保存波峰数据
    peak_data = pd.DataFrame({
        "波数": x[peaks],
        "反射率": y_smooth_strong[peaks]
    })

    # 计算峰峰间距
    peak_spacing = np.diff(x[peaks])
    median_spacing = np.median(peak_spacing) if len(peak_spacing) > 0 else np.nan

    return peak_data, peak_spacing, median_spacing


# ----------- 分别保存两组数据 -----------
peak_data1, spacing1, median_spacing1 = plot_save_peaks(data1, "附件1_光谱波峰.png", title="附件1 光谱曲线")
peak_data2, spacing2, median_spacing2 = plot_save_peaks(data2, "附件2_光谱波峰.png", title="附件2 光谱曲线")

# ----------- 保存波峰数据到 Excel --------
with pd.ExcelWriter("波峰数据.xlsx") as writer:
    peak_data1.to_excel(writer, sheet_name="附件1波峰", index=False)
    peak_data2.to_excel(writer, sheet_name="附件2波峰", index=False)

# ----------- 输出峰间距中位数 --------
print("附件1峰峰间距中位数：", median_spacing1)
print("附件2峰峰间距中位数：", median_spacing2)