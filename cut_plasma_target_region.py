##get_plasma_reigon_way3
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

def get_plasma_region(img, kernel_size=(40, 40),crop_percent=0.15):
    img_raw=img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lower_lab = np.array([0, 135, 135])
    upper_lab = np.array([255, 255, 255])
    mask = cv2.inRange(lab, lower_lab, upper_lab)

    plasma_region = cv2.bitwise_and(img, img, mask=mask)

    kernel = np.ones(kernel_size, np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_morphology = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(mask_morphology)
    if coords is None:
        return None, None, mask_morphology

    x, y, w, h = cv2.boundingRect(coords)
    plasma_crop = plasma_region[y:y+h, x:x+w]
    mask_crop = mask_morphology[y:y+h, x:x+w]

    plasma_region_crop = cv2.bitwise_and(plasma_crop, plasma_crop, mask=mask_crop)
    new_x, new_y, new_w, new_h = crop_box_with_percent(x, y, w, h, crop_percent)
    plasma_region_final = img[new_y:new_y+new_h,new_x:new_x+new_w]
    return (plasma_region, plasma_region_final, mask_morphology, new_x, new_y, new_w, new_h)


def crop_box_with_percent(x, y, w, h, crop_percent):
    """
    计算裁剪后区域在原图中的坐标
    """
    new_x = x + int(w * crop_percent)
    new_y = y + int(h * crop_percent)
    new_w = int(w * (1 - 2 * crop_percent))
    new_h = int(h * (1 - 2 * crop_percent))

    return new_x, new_y, new_w, new_h


if __name__ == "__main__":
    img_files = ["/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag/batch-2025-12-04_3_bgl1120/plasma/8/8.jpg"
         ,"/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag/batch-2025-07-30_1/plasma/9/save_15_39_16_425.jpg"
         ,"/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag/batch-2025-07-30_1/plasma/16/save_15_42_46_408.jpg"
        ,"/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag/batch-2025-07-30_1/plasma/6/save_15_37_41_021.jpg"
        ,"/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag/batch-2025-07-30_1/plasma/24/save_15_46_45_591.jpg"]

    for img_path in img_files:
        #print(f"sample file:{img_path}")
        img = cv2.imread(img_path)

        plasma_region, plasma_region_final, mask_morphology, rx, ry, rw, rh = get_plasma_region(img)
 
        if plasma_region_final is None:
            print("No region detected")
            continue
    

        # 在原图上画红色矩形框
        img_marked = img.copy()
        cv2.rectangle(img_marked, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 6)
        cv2.imwrite("result1.png", plasma_region_final)
        display(Image("result1.png"))
        cv2.imwrite("result_marked.png", img_marked)
        display(Image("result_marked.png"))

        print("-" * 50)
