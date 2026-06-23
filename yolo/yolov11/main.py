from ultralytics import YOLO

DATA_CONF_PATH = r"D:\projects\graduation_paper\yolo\datasets\severstal-steel-defect-instance-segmentation.v4i.yolov11\data.yaml"
EPOCHS = 200
DEVICE = 0

def main():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=DATA_CONF_PATH,
        epochs=EPOCHS,
        device=DEVICE,

        imgsz=800,           # 原图长边就是 800，没必要再硬拉大
        rect=True,           # 800x128 这类长条图建议开启
        batch=0.70,          # AutoBatch，约使用 70% 显存
        workers=2,           # Windows + 16G RAM 更稳
        cache="disk",        # 比 cache=True 更稳，少吃内存
        amp=True,

        name="train_4060_balanced",
        patience=50,
        save=True,

        optimizer="SGD",    # 避免 auto 覆盖你手动 lr0
        lr0=0.01,
        lrf=0.1,
        cos_lr=True,

        overlap_mask=True,
        mask_ratio=4,
        close_mosaic=10,

        hsv_s=0.4,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.3,
    )

    print("训练完成")

if __name__ == "__main__":
    main()
# from ultralytics import YOLO
# import cv2

# # 加载 YOLO11n 模型
# model = YOLO("best.pt")

# # 打开摄像头（0 = 默认摄像头）
# cap = cv2.VideoCapture(0)

# # 循环读取每一帧
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 🔥 关键：只保留置信度 > 0.6 的结果
#     results = model(frame, conf=0.6)

#     # 在画面上绘制检测框
#     annotated_frame = results[0].plot()

#     # 显示窗口
#     cv2.imshow("YOLO11n 检测 (按 q 退出)", annotated_frame)

#     # ✅ 按 q 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()