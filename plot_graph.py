import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Chuẩn bị dữ liệu
data = {
    "Model": ["YOLO26s_150epochs", "RTDETR_L_150epochs"],
    "Precision": [0.9589559507351513, 0.9817375260071889],
    "Recall": [0.874977817384402, 0.9182610060661626],
    "mAP50": [0.9312802398512143, 0.9290370184153224],
    "mAP50-95": [0.6927045142170452, 0.6983343013231889]
}

df = pd.DataFrame(data)
df.set_index("Model", inplace=True)

# 2. Thiết lập cấu hình biểu đồ
ax = df.plot(kind='bar', figsize=(12, 7), rot=0, width=0.8)

# 3. Tùy chỉnh tiêu đề và nhãn
plt.title("So sánh hiệu năng: YOLO26s vs RT-DETR (150 Epochs)", fontsize=14, fontweight='bold')
plt.ylabel("Giá trị chỉ số (0.0 - 1.0)", fontsize=12)
plt.xlabel("Mô hình", fontsize=12)

# Giới hạn trục Y từ 0.6 để thấy rõ sự khác biệt giữa các cột cao
plt.ylim(0.6, 1.05) 

# Thêm lưới (grid) để dễ quan sát
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Hiển thị giá trị số trên đầu mỗi cột
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 3)), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=10)

# 5. Hiển thị chú thích
plt.legend(title="Chỉ số", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()