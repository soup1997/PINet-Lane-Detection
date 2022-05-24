import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from parameters import Parameters
import math
import queue

p = Parameters()


class MovAvgFilter:  # 이동 평균 필터
    prevAvg = 0
    xBuf = queue.Queue()
    n = 0

    def __init__(self, _n):
        for _ in range(_n):
            self.xBuf.put(0)
        self.n = _n

    def movAvgFilter(self, x):
        front = self.xBuf.get()
        self.xBuf.put(x)

        avg = self.prevAvg + (x - front) / self.n
        self.prevAvg = avg

        return avg


def visualize_points(image, x, y):
    image = image
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 5, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)


def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size / ratio_w), int(p.y_size / ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)

    return test_image


def visualize_gt(self, gt_point, gt_instance, ground_angle, image):
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for y in range(self.p.grid_y):
        for x in range(self.p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x] * self.p.resize_ratio + self.p.resize_ratio * x)
                yy = int(gt_point[2][y][x] * self.p.resize_ratio + self.p.resize_ratio * y)
                image = cv2.circle(image, (xx, yy), 10, self.p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)


def visualize_regression(image, gt):
    image = np.rollaxis(image, axis=2, start=0)
    image = np.rollaxis(image, axis=2, start=0) * 255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):  # gt
            y_value = p.y_size - (p.regression_size - j) * (220 / p.regression_size)
            if i[j] > 0:
                x_value = int(i[j] * p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 3, p.color[color_index], -1)

    return image


###############################################################
##
## calculate
## 
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        out_x.append((np.array(i) / ratio_w).tolist())
        out_y.append((np.array(j) / ratio_h).tolist())

    return out_x, out_y


def get_closest_point_along_angle(x, y, point, angle):
    index = 0
    for i, j in zip(x, y):
        a = get_angle_two_points(point, (i, j))
        if abs(a - angle) < 0.1:
            return (i, j), index
        index += 1
    return (-1, -1), -1


def get_num_along_point(x, y, point1, point2, image=None):  # point1 : source
    x = np.array(x)
    y = np.array(y)

    x = x[y < point1[1]]
    y = y[y < point1[1]]

    dis = np.sqrt((x - point1[0]) ** 2 + (y - point1[1]) ** 2)

    count = 0
    shortest = 1000
    target_angle = get_angle_two_points(point1, point2)
    for i in range(len(dis)):
        angle = get_angle_two_points(point1, (x[i], y[i]))
        diff_angle = abs(angle - target_angle)
        distance = dis[i] * math.sin(diff_angle * math.pi * 2)
        if distance <= 12:
            count += 1
            if distance < shortest:
                shortest = distance

    return count, shortest


def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y < point[1]]
    y = y[y < point[1]]

    dis = (x - point[0]) ** 2 + (y - point[1]) ** 2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i, j))

    return points


def get_angle_two_points(p1, p2):
    del_x = p2[0] - p1[0]
    del_y = p2[1] - p1[1] + 0.000001
    if p2[0] >= p1[0] and p2[1] > p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta /= 360.0
    elif p2[0] > p1[0] and p2[1] <= p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 180
        theta /= 360.0
    elif p2[0] <= p1[0] and p2[1] < p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 180
        theta /= 360.0
    elif p2[0] < p1[0] and p2[1] >= p1[1]:
        theta = math.atan(float(del_x / del_y)) * 180 / math.pi
        theta += 360
        theta /= 360.0

    return theta


################# added ###################

steering_memory = 0  # 이전 값을 저장하기 위한 변수
avg_filter = MovAvgFilter(10)  # 실시간성이 우선 (n값을 낮게), 노이즈제거 우선(n값을 높게)


def get_angle_on_lane(x, y): 
    global steering_memory

    if len(x) == 1 or len(y) == 1:  # 레인이 한개만 인식됬을때 조향각 추출
        estimated_x = 512 - x[0][-1]
        estimated_y = y[0][-1]
        slope = (256 - estimated_y) / ((512 / 2) - estimated_x)
        if slope > 0:
            slope = -slope

        steering_angle = math.atan(slope) * 180 / math.pi

        if steering_angle > 0:
            steering_angle = -90 + steering_angle

        elif steering_angle < 0:
            steering_angle = 90 + steering_angle

    elif len(x) == 0 or len(y) == 0:  # 레인이 인식 안됬을때 조향각 유지
        return steering_memory

    else:  # 레인 2개 인식됬을때 조향각 추출
        z1 = list(np.polyfit(np.array(x[0]), np.array(y[0]), deg=1))
        z2 = list(np.polyfit(np.array(x[1]), np.array(y[1]), deg=1))

        if z1[0] > 0 or z2[0] < 0:
            left_lane, right_lane = z1, z2
        else:
            right_lane, left_lane = z1, z2

        Inter_X = (right_lane[1] - left_lane[1]) / (left_lane[0] - right_lane[0])
        Inter_Y = np.poly1d(left_lane)(Inter_X)
        slope = (256 - Inter_Y) / ((512 / 2) - Inter_X)
        steering_angle = math.atan(slope) * 180 / math.pi

        if steering_angle > 0:
            steering_angle = -90 + steering_angle

        elif steering_angle < 0:
            steering_angle = 90 + steering_angle

    steering_memory = steering_angle
    filtered_result = avg_filter.movAvgFilter(steering_angle)
    return np.round(filtered_result, 1)


def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

    return out_x, out_y


def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

    return out_x, out_y


def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)

    return out_x, out_y


def find_gt(x, y, target_lanes, target_h):
    ind = np.argsort(y, axis=0)
    min_x = np.take_along_axis(x, ind[::-1], axis=0).tolist()[0]
    min_y = np.take_along_axis(y, ind[::-1], axis=0).tolist()[0]

    index = 0
    dis = 10000
    closest_index = 0
    for i, j in zip(target_lanes, target_h):
        if i[0] > 0:
            temp = (i[0] - min_x) ** 2 + (j[0] - min_y) ** 2
            if temp < dis:
                dis = temp
                closest_index = index
            index += 1

    return closest_index
