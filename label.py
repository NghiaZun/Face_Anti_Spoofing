import os
import csv

synthetic_root = 'data/synthetic'
output_csv = 'data/labels.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_path', 'label'])
    for folder in sorted(os.listdir(synthetic_root)):
        folder_path = os.path.join(synthetic_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            # Quy ước: real nếu 'bent' là của x in [1,2] hoặc HR_1, HR_2, còn lại là spoof
            if fname.startswith(('1_bent', '2_bent', 'HR_1_bent', 'HR_2_bent')):
                label = 'real'
            else:
                label = 'spoof'
            rel_path = os.path.join('synthetic', folder, fname)
            writer.writerow([rel_path, label])