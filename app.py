#-----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#-----------------------------------------------------------------------------------------

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return app.send_static_file("index.html")
# 下面是我生成的一段自动驾驶的代码：

# 导入必要的库
import cv2 # OpenCV库，用于图像处理
import numpy as np # NumPy库，用于数值计算
import tensorflow as tf # TensorFlow库，用于深度学习

# 定义一些常量
IMAGE_WIDTH = 320 # 图像宽度
IMAGE_HEIGHT = 240 # 图像高度
IMAGE_CHANNELS = 3 # 图像通道数
STEERING_ANGLE_MAX = 30 # 最大转向角度
STEERING_ANGLE_MIN = -30 # 最小转向角度
STEERING_ANGLE_RANGE = STEERING_ANGLE_MAX - STEERING_ANGLE_MIN # 转向角度范围

# 加载训练好的模型
model = tf.keras.models.load_model('model.h5')

# 创建摄像头对象，捕获实时图像
cap = cv2.VideoCapture(0)

# 创建串口对象，与小车通信
ser = serial.Serial('/dev/ttyUSB0', 9600)

# 循环处理图像
while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        break
    
    # 调整图像大小和颜色空间，与训练时保持一致
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 将图像转换为张量，并增加一个维度，作为模型的输入
    input = tf.convert_to_tensor(frame)
    input = tf.expand_dims(input, axis=0)
    
    # 使用模型预测转向角度，范围为[-1, 1]
    output = model.predict(input)
    
    # 将转向角度映射到实际的角度范围，单位为度
    steering_angle = output[0] * STEERING_ANGLE_RANGE + STEERING_ANGLE_MIN
    
    # 将转向角度发送给小车，控制转向电机
    ser.write(str(steering_angle).encode())
    
    # 显示图像和转向角度
    cv2.imshow('frame', frame)
    print('Steering angle: {:.2f} degrees'.format(steering_angle))
    
    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
ser.close()
cv2.destroyAllWindows()
