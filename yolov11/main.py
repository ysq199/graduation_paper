from ultralytics import YOLO
import time

DATA_CONF16_PATH =r"D:\projects\datasets\severstal-steel-defect-detection.v1i.yolov11\data.yaml"
EPOCHS =200
DEVICE =0
BATCH =16

IMAGE_SIZE =640

CACHE = True
AMP =True
WORKERS =0

model=YOLO('yolo11n.pt')


results=model.train(
    data=DATA_CONF16_PATH,
    epochs=EPOCHS,
    device=DEVICE,
    batch=BATCH,
    name="train_optimized",
    workers=WORKERS,
    amp=AMP,
    cache=CACHE,

    imgsz=IMAGE_SIZE,
    patience=50,
    save=True,

    cos_lr=True,
    retina_masks=True,
    overlap_mask=True,
    close_mosaic=10,
    lr0=0.01,
    lrf=0.1,

)
print("训练完成")
time.sleep(10)
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