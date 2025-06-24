import os
import csv

LABEL_MAP = {
    '1.avi': 'real',
    '2.avi': 'real',
    '3.avi': 'warp',
    '4.avi': 'warp',
    '5.avi': 'cut',
    '6.avi': 'cut',
    '7.avi': 'replay',
    '8.avi': 'replay',
    'HR_1.avi': 'real',
    'HR_2.avi': 'warp',
    'HR_3.avi': 'cut',
    'HR_4.avi': 'replay'
}

dataset_root = 'data/casia_mfsd/train_release'
output_csv = 'data/labels.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['subject_id', 'video_path', 'label'])

    for subject_folder in sorted(os.listdir(dataset_root)):
        subject_path = os.path.join(dataset_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        for fname in os.listdir(subject_path):
            label = LABEL_MAP.get(fname)
            if label is not None:
                rel_path = os.path.join(subject_folder, fname)
                writer.writerow([subject_folder, rel_path, label])
