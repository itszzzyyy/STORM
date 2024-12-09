import csv
from datetime import datetime, timedelta
import pandas as pd
# 定义CSV文件名和要提取的列名
csv_filename = './simple_cleaned_AIS_little_region.csv'
columns_of_interest = ['MMSI','BaseDateTime', 'LAT', 'LON','IMO']

# 用于存储结果的列表
data = []

# 打开CSV文件并读取数据
with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    # 遍历每一行数据
    for row in reader:
        # 创建一个新的字典，仅包含感兴趣的列数据
        selected_data = {col: row[col] for col in columns_of_interest}
        data.append(selected_data)
points = []
minlat = 100.0
maxlat = -100.0
minlon = 100.0
maxlon = -100.0
maxslen = -1
minslen = 100000
# 输出结果
def divide_by_hour(traj):
    out = []

    middle = []
    cnt = 0
    middle.append(traj[cnt])
    for j in range(1, len(traj)):
        #print("时间戳",traj[j][2])
        time1 = datetime.fromtimestamp(traj[j][2])
        time2 = datetime.fromtimestamp(traj[cnt][2])
        time_diff = abs(time1 - time2)
        if time_diff > timedelta(hours=4):
            out.append([item for item in middle])
            middle.clear()
            middle.append(traj[j])
            cnt = j
        else:
            middle.append(traj[j])
            if j == len(traj) - 1:
                out.append([item for item in middle])
                middle.clear()
    return out

for item in data:
    # 创建一个 datetime 对象
    dt = datetime.strptime(str(item['BaseDateTime']),'%Y-%m-%dT%H:%M:%S')  # 假设要转换的时间为2024年7月28日12点0分0秒

    # 将 datetime 对象转换为时间戳
    timestamp = dt.timestamp()
    points.append((timestamp, float(item['LAT']), float(item['LON']),item['MMSI'],item['IMO']))
    minlat = min(minlat,float(item['LAT']))
    maxlat = max(maxlat,float(item['LAT']))
    minlon = min(minlon,float(item['LON']))
    maxlon = max(maxlon,float(item['LON']))
print("minlat: {}, maxlat: {}, minlon:{}, maxlon: {}".format(minlat, maxlat, minlon,maxlon))
trajs = []
traj = []
times = []
time = []
lens = 0
for i in range(0,len(points)):
    if i == 0:
        traj.append([points[i][2],points[i][1],points[i][0]])

    else:
        if points[i][3] != points[i-1][3] or points[i][4] != points[i-1][4]:
            trajs.append([item for item in traj])
            traj.clear()
        else:
            traj.append([points[i][2],points[i][1],points[i][0]])

trajs.append([item for item in traj])
divided_traj = []
print("最开始的轨迹数量",len(trajs))
for item in trajs:
    outs = divide_by_hour(item)
    for i in range(0,len(outs)):
        divided_traj.append(outs[i])
# trajs_to_save = divided_traj
trajs_to_save = []
traj_len = []
for item in divided_traj:
    temp = []
    lens += len(item)
    traj_len.append(len(item))
    maxslen = max(maxslen,len(item))
    minslen = min(minslen,len(item))
    for i in range(0,len(item)):
        # if i == 0:
        #     times.append(item[i][0])
        temp.append([item[i][0],item[i][1],item[i][2]])
    trajs_to_save.append([elem for elem in temp])
# final_trajs = []
# time_drop = 0
# mintime_drop = 1000
# maxtime_drop = 0
# cnt = 0
# for traj in trajs_to_save:
#
#     for i in range(1,len(traj)):
#         time1 = datetime.fromtimestamp(traj[i][2])
#         time2 = datetime.fromtimestamp(traj[i-1][2])
#         time_diff = abs(time1 - time2)
#         time_drop += int(time_diff.total_seconds())
#         mintime_drop = min(mintime_drop,int(time_diff.total_seconds()))
#         maxtime_drop = max(maxtime_drop,int(time_diff.total_seconds()))
#         cnt += 1
    # time_drop = time_drop/len(traj)
    # if time_drop >=15 and time_drop <=100:
    #     print("timedrop",time_drop)
    #     final_trajs.append(traj)
# print("average timedrop",time_drop/cnt)
# print("max timedrop",maxtime_drop)
# print("min timedrop",mintime_drop)
time_drops = 0

# mintime_drop = 1000
# maxtime_drop = 0
cnt = 0
for traj in trajs_to_save:
    time_drop = 0
    for i in range(1,len(traj)):
        time1 = datetime.fromtimestamp(traj[i][2])
        time2 = datetime.fromtimestamp(traj[i-1][2])
        time_diff = abs(time1 - time2)
        time_drop += int(time_diff.total_seconds())
        # mintime_drop = min(mintime_drop,int(time_diff.total_seconds()))
        # maxtime_drop = max(maxtime_drop,int(time_diff.total_seconds()))
        cnt += 1
    time_drop = time_drop/len(traj)
    time_drops += time_drop
print("average timedrop",time_drops/len(trajs_to_save))
# print("max timedrop",maxtime_drop)
# print("min timedrop",mintime_drop)


# print("AVERAGE Length:",lens/len(trajs_to_save))
# print("MAXLEN:",maxslen)
# print("MINLEN:",minslen)
# sorted_trajss = sorted(traj_len)
# print("99%之后的轨迹长度",sorted_trajss[int(len(traj_len)*0.99)])
# print("1%之前的轨迹长度",sorted_trajss[int(len(traj_len)*0.01)-1])
# idx = 0
# with open('./processed_SOUTH.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["TRIP_ID","POLYLINE"])  # 写入表头
#     for i in range(0,len(trajs_to_save)):
#         writer.writerow([idx,trajs_to_save[i]])
#         idx += 1
print("AVERAGE Length:",lens/len(trajs_to_save))
print("MAXLEN:",maxslen)
print("MINLEN:",minslen)
sorted_trajss = sorted(traj_len)
print("99%之后的轨迹长度",sorted_trajss[int(len(traj_len)*0.99)])
print("1%之前的轨迹长度",sorted_trajss[int(len(traj_len)*0.01)-1])
idx = 0
for i in range(0,len(trajs_to_save)):
    for j in range(0,len(trajs_to_save[i])):
        time = datetime.fromtimestamp(trajs_to_save[i][j][2])
        start_second = (time.hour * 3600 +
                        time.minute * 60 + time.second)
        if start_second >= 86400:
            start_second -= 86400
        minute = int(start_second / 60)
        trajs_to_save[i][j][2] = minute
with open('./processed_times.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TRIP_ID","POLYLINE"])  # 写入表头
    for i in range(0,len(divided_traj)):
        writer.writerow([idx,divided_traj[i]])
        idx += 1
idx = 0
print("保存长度",len(trajs_to_save))
with open('./processed_TIME.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["TRIP_ID","POLYLINE"])  # 写入表头
    for i in range(0,len(trajs_to_save)):
        writer.writerow([idx,trajs_to_save[i]])
        idx += 1
