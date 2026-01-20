# fine_tune.py
from ultralytics import YOLO

model = YOLO('/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/continuing_dp3/weights/last.pt')

# # results = model.train(
# #     data='/home/24kien.dhc/AeroEyes/AeroEyes/Data/Preprocessed/data.yaml',
# #     epochs=250,
# #     imgsz=640,batch=8,                    # Giảm từ 32 → 8               # 8 x 4 = 32 → hiệu quả như batch=32
# #     device=6,       # <<<-- CHUỖI '0,1' — Ultralytics tự wrap DataParallel
# #     workers=4,
# #     name="continuing_dp",
# #     patience=0,
# #     optimizer='AdamW',
# #     box=10.0,   # mặc định 7.5 → tăng lên 10-15
# #     cls=0.5,    # giảm cls vì đã học tốt
# #     dfl=1.5,
# #     scale=0.9,
# #     translate=0.1,
# #     fliplr=0.0,
# #     lr0=0.0001,           # Giảm lr ban đầu
# #     lrf=0.01,mosaic=0.5,        # Giảm để tránh mất object nhỏ
# #     close_mosaic=40,
# #     save=True,
# #     resume = True       # Giảm để tránh treo
# )



# from ultralytics import YOLO

# model = YOLO('yolov8l.pt')  # hoặc load last.pt nếu cần

# results = model.train(
#     data='/home/24kien.dhc/AeroEyes/AeroEyes/Data/Preprocessed/data.yaml',
#     epochs=300,
#     imgsz=1280,           # Tăng lên!
#     batch=32,
#     device=6,
#     workers=8,
#     name="small_object_fix",
#     patience=50,
#     save=True,
#     resume=False,         # Tắt resume nếu last.pt không tốt

#     # Hyperparameters
#     lr0=0.001,
#     lrf=0.1,
#     optimizer='AdamW',
#     cos_lr=True,
#     warmup_epochs=5,

#     # Loss weights
#     box=12.0,             # Tăng mạnh
#     cls=0.3,              # Giảm cls
#     dfl=1.5,

#     # Augmentation
#     close_mosaic=30,      # Tắt mosaic cuối
#     mosaic=0.5,           # Giảm probability
#     scale=0.9,
#     translate=0.1,
#     fliplr=0.0,           # Tắt flip nếu vật thể có hướng

#     # Anchor
#     anchor='auto',

#     # Debug
#     plots=True,
#     val=True,
# )



results = model.train(
    data='/home/24kien.dhc/AeroEyes/AeroEyes/Data/Preprocessed/data.yaml',
    lr0 = 0.0005,
    lrf = 0.05,
    epochs = 100,
    momentum = 0.94,
    weight_decay = 0.0001,
    warmup_epochs = 0.5,
    cos_lr = True,
    batch = 16, 
    imgsz = 640,
    device=6,       # <<<-- CHUỖI '0,1' — Ultralytics tự wrap DataParallel
    workers=8,
    name="continuing_dp",
    patience=0,
    save = True
)