# import numpy as np
#
# # 加载保存的数据
# loaded_data_x = np.load("./porto_data/1_q_drop40.npy",allow_pickle=True)
# loaded_data_y = np.load("./porto_data/1_db_drop40.npy",allow_pickle=True)
# print(loaded_data_x.shape)
# print(loaded_data_x)
# # 打印数据
# # for item in loaded_data_x:
# #     for i in item:
# #         cnt = 0
# #         for j in i:
# # #             cnt += 1
# # #         print("cnt",cnt)
# # #     break
# #
# # # 查看数据的形状
# # print("数据形状:", loaded_data_x.shape,loaded_data_y.shape)
# import pandas as pd
# import csv
# import json
# # 加载pickle文件
# datas = pd.read_pickle('./axis/ct_dma_test.pkl')
# print(len(datas))
# #print(type(datas))
# # data = [row[1] for row in datas]
# traj_data = [item['traj'] for item in datas]
# #print(traj_data[0])
#
# traj = []
#
# for item in traj_data:
#     points = []
#     for point in item:
#         points.append([point[0], point[1],point[4]])
#
#     traj.append(points)
#     #print(traj)
# # df = pd.DataFrame(traj)
# # # 将数据保存为CSV文件
# # df.to_csv('./axis/traintest.csv', index=False)
# csv_file_path = "trajectories_test.csv"
# with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#
#     # 写入标题行
#     writer.writerow(["POLYLINE"])
#
#     # 将轨迹中的每个点作为一行写入CSV文件
#     for trajs in traj:
#         traj_json = json.dumps(trajs)
#         writer.writerow([trajs])
import re
import matplotlib.pyplot as plt
import csv

in_path = './dataset/data.csv'
# file = open(in_path, 'r')

longitude = []
latitude = []
source = []
longitude_list = []
latitude_list = []
source_list = []
with open(in_path, 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i == 100:
            longitude = row[1]
            longitude_values = longitude.strip('[]').split(', ')
            # 将每个字符串经度值转换为浮点数并添加到列表中
            longitude_list = [float(value) for value in longitude_values]
            latitude = row[2]
            latitude_values = latitude.strip('[]').split(', ')
            latitude_list = [float(value) for value in latitude_values]
            source = row[4]
            source_values = source.strip('[]').strip(" ").split(',')
            source_list = [value for value in source_values]
            source_list = [value.strip() for value in source_list]
            source_list = [value.replace("'", "") for value in source_list]
            print(source_list)
            break
print(len(longitude_list))
print(len(latitude_list))
print(len(source_list))
points_A_lon = []
points_A_lat = []
points_B_lon = []
points_B_lat = []
points_C_lon = []
points_C_lat = []
points_D_lon = []
points_D_lat = []
for i in range(0, len(source_list)):
    if source_list[i] == 'A':
        print('A', longitude_list[i])
        points_A_lon.append(longitude_list[i])
        points_A_lat.append(latitude_list[i])
    elif source_list[i] == 'B':
        print('B', longitude_list[i])
        points_B_lon.append(longitude_list[i])
        points_B_lat.append(latitude_list[i])
    elif source_list[i] == 'C':
        print('C', longitude_list[i])
        points_C_lon.append(longitude_list[i])
        points_C_lat.append(latitude_list[i])
    elif source_list[i] == 'D':
        print('D', longitude_list[i])
        points_D_lon.append(longitude_list[i])
        points_D_lat.append(latitude_list[i])
print(len(points_A_lat),len(points_A_lon))
plt.figure(figsize=(10, 6))
if len(points_A_lat) != 0:
    plt.plot(points_A_lon, points_A_lat, 'ro-', label='source A')  # 使用红色的圆点线连接A类点
if len(points_B_lat) != 0:
    plt.plot(points_B_lon, points_B_lat, 'bo-', label='source B')  # 使用蓝色的圆点线连接B类点
if len(points_C_lon) != 0:
    plt.plot(points_C_lon, points_C_lat, 'go-', label='source C')  # 使用绿色的圆点线连接c类点
if len(points_D_lon) != 0:
    plt.plot(points_D_lon, points_D_lat, 'yo-', label='source D')  # 使用绿色的圆点线连接D类点

plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('trajectory')
plt.legend()
plt.grid(True)
plt.savefig('trajectory_plot.png', dpi=300)
plt.show()
# with open(in_path, 'r', newline='', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     for i, row in enumerate(reader):
#         if i == 5:
#             longitude = [item for item in row[1]]
#             latitude = [item for item in row[2]]
#             source = [item for item in row[4]]
#             break
#     points_A_lon = []
#     points_A_lat = []
#     points_B_lon = []
#     points_B_lat = []
#     points_C_lon = []
#     points_C_lat = []
#     points_D_lon = []
#     points_D_lat = []
#     print(len(source))
#     for i in range(0,len(source)):
#         if source[i] == 'A':
#             points_A_lon.append(longitude[i])
#             points_A_lat.append(latitude[i])
#         elif source[i] == 'B':
#             points_B_lon.append(longitude[i])
#             points_B_lat.append(latitude[i])
#         elif source[i] == 'C':
#             points_C_lon.append(longitude[i])
#             points_C_lat.append(latitude[i])
#         elif source[i] == 'D':
#             points_D_lon.append(longitude[i])
#             points_D_lat.append(latitude[i])
#     plt.figure(figsize=(10, 6))
#     plt.plot(points_A_lon, points_A_lat, 'ro-', label='来源 A')  # 使用红色的圆点线连接A类点
#     plt.plot(points_B_lon, points_B_lat, 'bo-', label='来源 B')  # 使用蓝色的圆点线连接B类点
#     plt.plot(points_C_lon, points_C_lat, 'go-', label='来源 C')  # 使用绿色的圆点线连接c类点
#     plt.plot(points_D_lon, points_D_lat, 'yo-', label='来源 D')  # 使用绿色的圆点线连接D类点
#     plt.xlabel('经度')
#     plt.ylabel('纬度')
#     plt.title('轨迹图')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
