from ultralytics import YOLO
import numpy as np
import pandas as pd

MODEL_PATH = "runs/detect/test_v4.2/weights/best.pt"
DATA_YAML = "data_test_v4.2.yaml"

CONF_LIST = [0.01] + [round(i / 100, 2) for i in range(5, 96, 5)]
TARGET_PRECISIONS = [round(0.5 + 0.05 * i, 2) for i in range(7)]


def main():
    model = YOLO(MODEL_PATH)

    # chạy val 1 lần duy nhất, conf thấp để có đường cong đầy đủ
    res = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        conf=0.001,
        device=0,
        plots=False,
        workers=0
    )

    # curves_results: [P-R, F1-Conf, P-Conf, R-Conf]
    curves = res.curves_results
    px   = np.array(curves[1][0])            # trục confidence, 1000 điểm
    f1_c = np.array(curves[1][1]).mean(0)    # trung bình trên các class
    p_c  = np.array(curves[2][1]).mean(0)
    r_c  = np.array(curves[3][1]).mean(0)

    rows = []
    for conf in CONF_LIST:
        i = int(np.argmin(np.abs(px - conf)))
        rows.append({
            "conf": conf,
            "precision": p_c[i],
            "recall": r_c[i],
            "f1": f1_c[i],
        })

    df = pd.DataFrame(rows)
    df.to_csv("threshold_results.csv", index=False)

    print("\n===== P/R/F1 theo conf =====")
    print(df.to_string(index=False))

    # điểm F1 tối ưu chính xác trên toàn đường cong
    bi = int(f1_c.argmax())
    print(f"\nBest F1: conf={px[bi]:.3f} | P={p_c[bi]:.4f} | "
          f"R={r_c[bi]:.4f} | F1={f1_c[bi]:.4f}")

    for target in TARGET_PRECISIONS:
        mask = p_c >= target
        if mask.any():
            f1_masked = np.where(mask, f1_c, -1)
            j = int(f1_masked.argmax())
            print(f"P >= {target:.2f}: conf={px[j]:.3f} | "
                  f"P={p_c[j]:.4f} | R={r_c[j]:.4f} | F1={f1_c[j]:.4f}")
        else:
            print(f"P >= {target:.2f}: KHONG dat duoc")

    print(f"\nmAP50={res.box.map50:.4f} | mAP50-95={res.box.map:.4f}")


if __name__ == "__main__":
    main()