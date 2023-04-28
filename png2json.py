#确定版本V1.0_23.04.28
import os
import json
import numpy as np
import cv2
import labelme
import base64
from glob import glob
from tqdm import tqdm

def process_folder(folder_path):
    # 获取所有PNG文件路径
    image_paths = glob(os.path.join(folder_path, "*.png"))
    
    # 遍历所有PNG文件
    for i, image_path in enumerate(tqdm(image_paths)):
        # 缩小图像尺寸
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]
        image = cv2.resize(image, (w // 2, h // 2))
    
        # 计算缩放比例
        scale_x = w / image.shape[1]
        scale_y = h / image.shape[0]
    
        # 找到所有等于255的像素的坐标，作为标注的对象
        y, x = np.where(image == 255)
    
        # 添加标注对象
        shapes = []
        for j in range(len(x)):
            # 计算实际坐标
            px, py = int(x[j] * scale_x), int(y[j] * scale_y)
        
            # 添加标注对象
            points = [(px, py), (px + 1, py), (px + 1, py + 1), (px, py + 1)]
            shape = dict(label="center", points=points, shape_type="polygon")
            if not isinstance(shape['points'], (list, np.ndarray)):
                shape['points'] = np.array(shape['points']).tolist()
            shapes.append(shape)
    
        # 将标注数据转换为二值化的掩模图像
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))
    
        # 对掩模图像进行形态学操作，将相邻的、同种类别的区域合并成一个整体的区域
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernel)
        _, contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        merged_shapes = []
        for contour in contours:
            # 使用cv2.approxPolyDP函数对多边形进行逼近，得到优化后的多边形
                #epsilon乘的值越小，逼近的精度就越高，但是点的数量也会更多
                #在不进行缩小时乘0.001比较合适，
                # 在长宽均缩小5时建议乘0.004
                # 在长宽均缩小2时建议乘0.003
            epsilon = 0.003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
            points = approx.reshape(-1, 2)
        
            # 还原实际坐标
            points = [(int(p[0] / 1), int(p[1] / 1)) for p in points]
        
            shape = dict(label="center", points=points, shape_type="polygon")
        
            # 检查points是否为列表或数组类型
            if not isinstance(shape['points'], (list, np.ndarray)):
                shape['points'] = np.array(shape['points']).tolist()
            
            merged_shapes.append(shape)
    
        # 转换为JSON格式数据
        label_name_to_value = {"center": 1}
        img_shape = [h, w, 3]
        image_path = image_path.replace(".png", ".jpg")
        json_data = {"version": "5.2.0", "flags": {}, "shapes": merged_shapes, "imagePath": os.path.basename(image_path), "imageData": None}
    
        # 获取文件名并保存为JSON文件
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_file_path = os.path.join(os.path.dirname(image_path), base_name + ".json")
    
        with open(json_file_path, "w") as fp:
            json.dump(json_data, fp)
            
    print(f"处理完成，共处理了{i+1}张图片。")

# 输入待处理的文件夹路径
folder_path = r'D:\PATH\PATH'
process_folder(folder_path)
# 生成的JSON文件保存在与PNG文件相同的文件夹中