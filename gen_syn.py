import cv2
import os
import numpy as np
import glob

def warp_bend(img, bend_factor=0.2):
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            offset = bend_factor * np.sin(np.pi * j / w)
            map_x[i, j] = j
            map_y[i, j] = i + offset * h
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# Lấy danh sách tất cả file .avi trong thư mục train_release
video_files = glob.glob("data/casia_mfsd/train_release/*/*.avi")

num_frames = 10  # Số lượng frame muốn cắt từ mỗi video

for vid_path in video_files:
    parent_folder = os.path.basename(os.path.dirname(vid_path))
    out_dir = os.path.join("data/synthetic", parent_folder)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    frame_count = 0
    saved_count = 0
    while saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        bent = warp_bend(frame)
        base = os.path.basename(vid_path).replace('.avi', f'_bent_{saved_count+1}.jpg')
        out_path = os.path.join(out_dir, base)
        cv2.imwrite(out_path, bent)
        print(f"Đã lưu {out_path}")
        saved_count += 1
    cap.release()