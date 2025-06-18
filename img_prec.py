import json
import datetime
import wires_recog
import ele_recog
import cv2
import time
import numpy as np
import base64

# 生成json数据
def generate_json(wires, components):
    vertices = []
    elements = []

    vertix_count = 0
    element_count = 0
    labels = {"微安电流表": "ammeter",
              "待测表头": "ammeter",
              "电阻箱": "RESISTOR_BOX",
              "滑动变阻器": "VARIABLE_RESISTOR",
              "单刀双掷开关": "switch",
              "电源": "battery"
              }

    for comp in components:
        bbox = comp["bbox"]
        label = labels[comp["label"]]
        elem = {
              "id": f"element_{element_count}",
              "startVertexId": "",
              "endVertexId": "",
              "type": label,
        }
        if bbox[2]-bbox[0] >= bbox[3] - bbox[1]:
            vertix = {
                "id": f"vertix_{vertix_count}",
                "x": bbox[0] + ((bbox[2] - bbox[0]) / 10),
                "y": bbox[1] + ((bbox[3] - bbox[1]) / 3),
            }
            elem["startVertexId"] = vertix["id"]
            vertices.append(vertix)
            vertix_count += 1
            vertix = {
                "id": f"vertix_{vertix_count}",
                "x": bbox[0] + ((bbox[2] - bbox[0]) * 9 / 10),
                "y": bbox[1] + ((bbox[3] - bbox[1]) / 3),
            }
            elem["endVertexId"] = vertix["id"]
            vertices.append(vertix)
            vertix_count += 1
        else:
            vertix = {
                "id": f"vertix_{vertix_count}",
                "x": bbox[0] + ((bbox[2] - bbox[0]) / 2),
                "y": bbox[1] + ((bbox[3] - bbox[1]) / 9),
            }
            elem["startVertexId"] = vertix["id"]
            vertices.append(vertix)
            vertix_count += 1
            vertix = {
                "id": f"vertix_{vertix_count}",
                "x": bbox[0] + ((bbox[2] - bbox[0]) / 2),
                "y": bbox[1] + ((bbox[3] - bbox[1]) * 8 / 9),
            }
            elem["endVertexId"] = vertix["id"]
            vertices.append(vertix)
            vertix_count += 1

        if label == "switch":
            elem["closed"] = False
        elif label == "ammeter":
            elem["resistance"] = 1
            elem["type"] = "seriesAmmeter"
            elem["customLabel"] = "电流表"
            elem["customDisplayFunction"]: "i => `${i.toFixed(2)} A`"
        elif label == "voltmeter":
            elem["resistance"] = 1
            elem["type"] = "seriesAmmeter"
            elem["customLabel"] = "电压表"
            elem["customDisplayFunction"]: "i => `${i.toFixed(2)} V`"
        elif label == "RESISTOR_BOX" or label == "VARIABLE_RESISTOR":
            elem["type"] = "resistor"
            elem["resistorType"] = label
            elem["resistance"] = 10
        elif label == "battery":
            elem["voltage"] = 9
            elem["batterType"] = "BATTERRY"
            elem["internalResistance"] = 0.01
        elements.append(elem)
        element_count += 1

    def find_nearest(point):
        min_dist = float('inf')
        nearest_vertex = None
        for vertex in vertices:
            ver = (vertex["x"], vertex["y"])
            dist = np.linalg.norm(np.array(point) - np.array(ver))
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = vertex
        return nearest_vertex
    # 加入wire
    for wire in wires:
        wire_start = (wire["start"]["x"], wire["start"]["y"])
        wire_end = (wire["end"]["x"], wire["end"]["y"])
        nearest_start = find_nearest(wire_start)
        nearest_end = find_nearest(wire_end)

        elements.append({
            "id": f"element_{element_count}",
            "startVertexId": nearest_start["id"],
            "endVertexId": nearest_end["id"],
            "type": "wire",
            "resistance": 3e-8
        })
        element_count += 1

    data = {
        "formatVersion": "1.0",
        "metadata": {
            "title": "Exported Circuit",
            "description": "Circuit exported from image",
            "created": datetime.datetime.now(datetime.UTC).isoformat() + "Z"
        },
        "vertices": vertices,
        "elements": elements,
        "displaySettings": {
            "showCurrent": True,
            "currentType": "electrons",
            "wireResistivity": 1e-10,
            "sourceResistance": 0.0001
        }
    }
    return data


def visualize_wires_and_components(image, results, components):
    original = image
    img = cv2.resize(original, (1000, int(original.shape[0] * 1000 / original.shape[1])))
    if img is None:
        raise FileNotFoundError(f"无法读取图像：")

    for p in results["vertices"]:
        point = (int(p["x"]), int(p["y"]))
        cv2.circle(img, point, 6, (0, 0, 255), -1)

    # ==== 画导线 ====
    for wire in results["elements"]:
        if wire["type"] != "wire":
            continue
        point = wire["startVertexId"]
        for v in results["vertices"]:
            if v["id"] == point:
                point = v

        start = (int(point["x"]), int(point["y"]))
        point = wire["endVertexId"]
        for v in results["vertices"]:
            if v["id"] == point:
                point = v
        end = (int(point["x"]), int(point["y"]))

        # 起点：红色，终点：蓝色
        cv2.circle(img, start, 6, (0, 0, 255), -1)
        cv2.circle(img, end, 6, (255, 0, 0), -1)
        cv2.line(img, start, end, (0, 255, 255), 2)

        cv2.putText(img, "start", (start[0]+5, start[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, "end", (end[0]+5, end[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # ==== 画元件框 ====
    labels = {"微安电流表": "seriesAmmeter",
              "待测表头": "seriesAmmeter",
              "电阻箱": "RESISTOR_BOX",
              "滑动变阻器": "VARIABLE_RESISTOR",
              "单刀双掷开关": "switch",
              "电源": "BATTERY"
              }
    for comp in components:
        label = labels[comp["label"]]
        x1, y1, x2, y2 = comp["bbox"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, encode_img = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(encode_img).decode('utf-8')
    #cv2.imwrite('output.jpg', img)
    return img_base64
    # 显示图像
    # resized = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
    # cv2.namedWindow("Wires and Components", cv2.WINDOW_NORMAL)
    # cv2.imshow("Wires and Components", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def img_recognition(img):
    wires = wires_recog.detect_wires_and_endpoints(img)
    elements = ele_recog.elements_recognition(img)
    results = generate_json(wires, elements)
    results_img = visualize_wires_and_components(img, results, elements)
    sumx = 0
    sumy = 0
    for tmp in results["vertices"]:
        sumx += tmp["x"]
        sumy += tmp["y"]
    for i in range(len(results["vertices"])):
        results["vertices"][i]["x"] -= sumx/len(results["vertices"])
        results["vertices"][i]["y"] -= sumy/len(results["vertices"])
        results["vertices"][i]["x"] *= 0.6
        results["vertices"][i]["y"] *= 0.6

    request = {
        "success": True,
        "recognizedImage": f"data:image/jpeg;base64,{results_img}",
        "circuitData": results
    }
    # with open('test.json', "w") as f:
    #     json.dump(request, f, indent=2)
    # print(f"✅ 已导出电路 JSON 至 {'result_json'}")
    return request


# if __name__ == '__main__':
#     start = time.perf_counter()
#     imgs_path = [
#
#     ]
#     for img_path in imgs_path:
#         img_recognition(img_path)
#     end = time.perf_counter()
#     print(f"处理{len(imgs_path)}张图片耗时:{end - start:.2f}s")
