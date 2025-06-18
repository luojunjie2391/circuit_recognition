import cv2
import numpy as np
from skimage.morphology import skeletonize


def show(title, img, scale=0.6):
    resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, resized)
    cv2.waitKey(0)


def extract_wire_mask(hsv_img, lower, upper, name='color'):
    mask = cv2.inRange(hsv_img, lower, upper)
    # show(f"{name} - begin", mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    # show(f"{name} - process + exbound", dilated)
    return dilated


def get_skeleton(binary_mask, name='skeleton'):
    skel = skeletonize(binary_mask // 255).astype(np.uint8) * 255
    # show(f"{name} - bouns", skel)
    return skel


def find_endpoints(skel_img):
    endpoints = []
    h, w = skel_img.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel_img[y, x] == 255:
                patch = skel_img[y - 1:y + 2, x - 1:x + 2]
                if cv2.countNonZero(patch) == 2:
                    endpoints.append((x, y))
    return endpoints


def detect_wires_and_endpoints(image):
    original = image
    img = cv2.resize(original, (1000, int(original.shape[0] * 1000 / original.shape[1])))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ranges = {
        'red':  [(np.array([0, 112, 38]), np.array([8, 255, 255])),
                 (np.array([160, 70, 50]), np.array([180, 255, 255]))],
        'green': [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
        'yellow': [(np.array([19, 115, 103]), np.array([35, 255, 255]))]
    }

    result_img = img.copy()
    all_wires = []

    for color_name, hsv_ranges in ranges.items():
        # print(f"\n🟢 正在处理颜色: {color_name.upper()}")

        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in hsv_ranges:
            mask = extract_wire_mask(hsv, lower, upper, color_name)
            mask_total = cv2.bitwise_or(mask_total, mask)

        skeleton = get_skeleton(mask_total, f"{color_name}_skeleton")
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

        for i in range(1, num_labels):
            wire_mask = (labels == i).astype(np.uint8) * 255
            pixel_count = cv2.countNonZero(wire_mask)

            if pixel_count < 100:
                continue

            endpoints = find_endpoints(wire_mask)
            wire_vis = cv2.cvtColor(wire_mask, cv2.COLOR_GRAY2BGR)

            if len(endpoints) >= 2:
                start, end = endpoints[0], endpoints[-1]
                # print(f"✅ {color_name.upper()}导线 #{i}: 起点 {start}，终点 {end}，像素数 {pixel_count}")
                cv2.circle(wire_vis, start, 6, (0, 0, 255), -1)
                cv2.circle(wire_vis, end, 6, (255, 0, 0), -1)
                cv2.line(wire_vis, start, end, (0, 255, 255), 2)

                cv2.circle(result_img, start, 6, (0, 0, 255), -1)
                cv2.circle(result_img, end, 6, (255, 0, 0), -1)
                cv2.line(result_img, start, end, (0, 255, 255), 2)

                # 保存导线数据
                wire_data = {
                    "start": {"x": int(start[0]), "y": int(start[1])},
                    "end": {"x": int(end[0]), "y": int(end[1])},
                }
                all_wires.append(wire_data)

            # show(f"{color_name.upper()} 导线 #{i}", wire_vis)

    # 显示图像
    # cv2.imshow('tmp', result_img)
    # cv2.waitKey(0)
    return all_wires

# 示例调用
# detect_wires_and_endpoints("img/5.jpg")
