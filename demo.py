import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl
# import dash_table
import numpy as np
import plotly.express as px
from dash import dash_table
from dash.dash_table.Format import Group
# 默认颜色和高亮颜色
DEFAULT_COLOR = "blue"
HIGHLIGHT_COLOR = "red"
# 示例轨迹数据处理函数
def process_traj_data(traj_db):
    traj_list = []
    first = traj_db[0][0]
    cnt = 0
    for traj in traj_db:
        traj_id = traj[0]-first
        points = traj[1]
        lon = points[:, 1]  # 经度
        lat = points[:, 0]  # 纬度
        traj_list.append({
            'id': traj_id,
            'positions': list(zip(lat, lon)),  # 轨迹完整路径
            'start_lat': float(lat[0]),
            'start_lon': float(lon[0]),
            'end_lat': float(lat[-1]),
            'end_lon': float(lon[-1]),
        })
        cnt += 1
    return traj_list

def process_mr_data(traj_mr):
    mr_list = []
    gt_list = []
    for pairs in traj_mr:
        mr_list.append(pairs[1])
        gt_list.append(pairs[2])
    return mr_list, gt_list

# 加载轨迹数据
traj_q = np.load('./1_raw_q_demo.npy', allow_pickle=True)
traj_mr = np.load('./mean_rank.npy', allow_pickle=True)
traj_mr_base = np.load('./mean_rank_baseline.npy', allow_pickle=True)
traj_dbs = np.load('./1_raw_db_demo.npy', allow_pickle=True)
traj_list = process_traj_data(traj_q)
traj_db = process_traj_data(traj_dbs)
mr_list, gt_list = process_mr_data(traj_mr)
mr_base_list,gt_base_list = process_mr_data(traj_mr_base)

print("traj_list",gt_list[86])


# 计算所有轨迹的经纬度边界
def calculate_bounds(traj_list):
    lats = []
    lons = []
    for traj in traj_list:
        for point in traj['positions']:
            lats.append(point[0])
            lons.append(point[1])
    return min(lats), max(lats), min(lons), max(lons)

# 计算经纬度的边界，方便地图显示
min_lat, max_lat, min_lon, max_lon = calculate_bounds(traj_list)

# 设置地图的中心点和缩放级别
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2
zoom_level = 10  # 根据轨迹的分布调整这个值

# 初始化表格数据
def get_table_data(selected_trajectory=None):
    if selected_trajectory is None:
        # 显示所有轨迹
        return [
            {
                "No": idx + 1,
                "Trajectory ID": traj["id"],
                "Start Latitude": traj["start_lat"],
                "Start Longitude": traj["start_lon"],
                "End Latitude": traj["end_lat"],
                "End Longitude": traj["end_lon"],
            }
            for idx, traj in enumerate(traj_list)
        ]
    else:
        # 显示选中轨迹
        traj = next(traj for traj in traj_list if traj["id"] == selected_trajectory)
        return [({
            "No": 1,
            "Trajectory ID": traj["id"],
            "Start Latitude": traj["start_lat"],
            "Start Longitude": traj["start_lon"],
            "End Latitude": traj["end_lat"],
            "End Longitude": traj["end_lon"],
        })]
def get_DB_table_data(selected_trajectory=None):

    traj_to_visulize = []
    # 显示选中轨迹
    for idx in selected_trajectory:
        traj = next(traj for traj in traj_db if traj["id"] == idx)
        traj_to_visulize.append(traj)
    return [
        {
            "No": idx + 2,
            "Trajectory ID": traj["id"],
            "Start Latitude": traj["start_lat"],
            "Start Longitude": traj["start_lon"],
            "End Latitude": traj["end_lat"],
            "End Longitude": traj["end_lon"],
        }
        for idx, traj in enumerate(traj_to_visulize)
    ]
# 创建 Dash 应用
app = dash.Dash(__name__)

# 应用布局
app.layout = html.Div([
    # 隐藏状态，用于存储当前选中的轨迹
    dcc.Store(id="selected-trajectory", data=None),

    # 整体为左右两部分布局
    html.Div([
        # 左侧：表单区域，包括柱状图
        html.Div([
            html.H2(
                "STORM Demo",
                style={
                    'textAlign': 'left',
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '30px',
                }
            ),
            html.Div([
                html.Label(
                    "Selected Trajectory:",
                    style={
                        'fontSize': '16px',
                        'fontFamily': 'Arial, sans-serif',
                        'marginBottom': '10px',
                        'display': 'block',
                        'textAlign': 'left',
                    }
                ),
                html.Div(
                    id="trajectory-info",
                    style={
                        'border': '1px solid black',
                        'padding': '10px',
                        'marginBottom': '20px',
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '14px',
                        'textAlign': 'left',
                        'width': '350px',
                        'height': '40px',  # 固定高度
                        'boxSizing': 'border-box',
                        'overflow': 'hidden',  # 防止内容溢出
                        'backgroundColor': '#fff',
                    }
                ),
                html.Label(
                    "Top K Similar Trajectories:",
                    style={
                        'fontSize': '16px',
                        'fontFamily': 'Arial, sans-serif',
                        'marginBottom': '10px',
                        'display': 'block',
                        'textAlign': 'left',
                    }
                ),
                dcc.Dropdown(
                    id='top-k-dropdown',
                    options=[{'label': str(i), 'value': i} for i in range(1, 11)],
                    value=1,
                    style={
                        'width': '350px',
                        'fontFamily': 'Arial, sans-serif',
                    }
                ),
                html.Button(
                    "Reset",
                    id="reset-button",
                    style={
                        'marginTop': '20px',
                        'fontFamily': 'Arial, sans-serif',
                        'backgroundColor': '#28a745',
                        'color': 'white',
                        'border': 'none',
                        'width': '350px',
                        'padding': '10px 20px',
                        'cursor': 'pointer',
                        'textAlign': 'center'
                    }
                ),
                html.Button(
                    "Submit",  # 新增 Submit 按钮
                    id="submit-button",
                    style={
                        'marginTop': '30px',
                        'marginBottom': '10px',
                        'width': '350px',
                        'fontFamily': 'Arial, sans-serif',
                        'backgroundColor': '#28a745',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'cursor': 'pointer',
                        'textAlign': 'center'
                    }
                ),
# 添加文献信息


            # 将柱状图放到左侧面板
            html.Div(id="mean-rank-graph", style={'marginTop': '20px', 'width': '100%', 'height': '200px'}),  # 设置柱状图容器高度
            html.Div(id="citation-text", style={
                            'marginTop': '40px',
                            'fontFamily': 'Times New Roman, serif',
                            'fontSize': '12px',
                            # 'textAlign': 'left',
                            'width': '350px',
                            'color': 'black'
                        })
                        ])
        ], style={
            'width': '30%',
            'padding': '20px',
            'borderRight': '1px solid black',
            'fontFamily': 'Arial, sans-serif',
            'height': '100vh',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'flex-start',
            'justifyContent': 'flex-start',
            'minHeight': '100vh',  # 固定左侧高度
        }),

        # 右侧：地图和表格区域
        html.Div([
            # 地图区域
            dl.Map(
                center=[center_lat, center_lon],  # 使用计算出来的中心点
                zoom=zoom_level,  # 使用计算出来的缩放级别
                children=[
                    dl.TileLayer(),
                    dl.LayerGroup(
                        id="trajectory-layer",
                    )
                ],
                style={
                    'width': '100%',
                    'height': '60vh',  # 地图占用大部分高度
                    'marginBottom': '20px',
                }
            ),
            # 表格区域
            html.Div([
                html.H3(
                    "All Trajectory Details:",
                    style={
                        'textAlign': 'left',
                        'fontFamily': 'Arial, sans-serif',
                        'marginBottom': '10px'
                    }
                ),
                dash_table.DataTable(
                    id='trajectory-table',
                    columns=[
                        {"name": "No", "id": "No"},
                        {"name": "Trajectory ID", "id": "Trajectory ID"},
                        {"name": "Start Latitude", "id": "Start Latitude"},
                        {"name": "Start Longitude", "id": "Start Longitude"},
                        {"name": "End Latitude", "id": "End Latitude"},
                        {"name": "End Longitude", "id": "End Longitude"},
                    ],
                    data=get_table_data(),
                    style_table={
                        'height': '250px',  # 调整表格高度，减少占用空间
                        'overflowY': 'auto',  # 启用滚动
                        'width': '100%'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '5px',
                        'fontSize': '14px',
                        'fontFamily': 'Arial, sans-serif'
                    },
                    style_header={
                        'backgroundColor': '#f5f5f5',
                        'fontWeight': 'bold'
                    }
                )
            ])
        ], style={
            'width': '70%',
            'padding': '20px',
            'height': '100vh',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column'
        })
    ], style={'display': 'flex'})
])
selected_id = None
@app.callback(
    [
        Output("trajectory-layer", "children"),  # 更新轨迹图层
        Output("trajectory-info", "children"),  # 更新选中轨迹信息
        Output("selected-trajectory", "data"),  # 更新选中轨迹 ID
        Output("trajectory-table", "data"),  # 更新表格数据
        Output("mean-rank-graph", "children"),  # 更新 Mean Rank 图表
        Output("citation-text", "children")  # 更新 Mean Rank 图表
    ],
    [
        Input({'type': 'trajectory', 'index': ALL}, 'n_clicks'),  # 所有轨迹点击事件
        Input("reset-button", "n_clicks"),  # Reset 按钮点击事件
        Input("submit-button", "n_clicks")  # Submit 按钮点击事件
    ],
    [
        State("selected-trajectory", "data"),  # 当前选中轨迹 ID
        State("top-k-dropdown", "value")  # 获取 top-k 的选择值
    ]
)

def update_trajectory(n_clicks_list, reset_clicked, submit_clicked, selected_trajectory, top_k):
    ctx = dash.callback_context
    print("ctx",ctx)
    global selected_id
    # 初始化返回值字典
    return_values = {
        "trajectory_layer": [],  # 默认没有轨迹显示
        "info_text": "",
        "selected_trajectory": None,
        "table_data": get_table_data(),  # 默认返回的表格数据
        "mean_rank_graph": None,
        "citation-text":None
    }

    # 如果 Reset 被点击，显示所有轨迹并清空选中状态
    if ctx.triggered and "reset-button" in ctx.triggered[0]['prop_id']:
        selected_trajectory = None  # 清空选中轨迹
        selected_id = None
        # 重置所有轨迹的颜色为默认颜色，并清除点击计数
        return_values["trajectory_layer"] = [
            dl.Polyline(
                id={'type': 'trajectory', 'index': traj['id']},
                positions=traj['positions'],
                color='blue',  # 恢复默认颜色
                weight=2,
                opacity=0.8,
                n_clicks=0  # 重置点击计数
            )
            for traj in traj_list
        ]

        return_values["info_text"] = "Reset: Displaying all trajectories."
        return_values["selected_trajectory"] = None
        return_values["mean_rank_graph"] = None  # 清空 Mean Rank 图表
        return_values["table_data"] = get_table_data()  # 清空表格数据

        # 保证 nclicks_list 长度为 200
        if len(n_clicks_list) < 200:
            n_clicks_list = [0] * (200 - len(n_clicks_list)) + n_clicks_list
        print("resetnclicks_list", n_clicks_list)
        return [
            return_values["trajectory_layer"],
            return_values["info_text"],
            return_values["selected_trajectory"],
            return_values["table_data"],
            return_values["mean_rank_graph"],
            return_values["citation-text"]
        ]


    # 处理轨迹点击事件
    elif ctx.triggered and "trajectory" in ctx.triggered[0]['prop_id']:

        for i, n_clicks in enumerate(n_clicks_list):
            if n_clicks and len(n_clicks_list)!=len(traj_list):
                clicked_trajectory = traj_list[selected_id]
                selected_trajectory = selected_id
                return_values["trajectory_layer"] = [
                    dl.Polyline(
                        id={'type': 'trajectory', 'index': selected_id},
                        positions=clicked_trajectory['positions'],
                        color='red',  # 高亮选中的轨迹
                        weight=2,
                        opacity=0.9,
                        n_clicks=0  # 重置点击计数
                    )
                ]
                # 更新轨迹信息
                return_values["info_text"] = f"Trajectory {clicked_trajectory['id']} selected.\n"
                return_values[
                    "info_text"] += f"Start: ({clicked_trajectory['start_lat']}, {clicked_trajectory['start_lon']})\n"
                return_values[
                    "info_text"] += f"End: ({clicked_trajectory['end_lat']}, {clicked_trajectory['end_lon']})"
                # 更新表格数据
                return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                return_values["selected_trajectory"] = selected_trajectory
            elif n_clicks:  # 如果某条轨迹被点击
                print("进首次点击")
                clicked_trajectory = traj_list[i]
                selected_trajectory = clicked_trajectory['id']
                selected_id = selected_trajectory
                # print("selected trajectory: ", selected_trajectory)
                # print("nlicks",n_clicks)
                # print("clicks",n_clicks_list)
                # 更新轨迹层，显示点击的轨迹
                return_values["trajectory_layer"] = [
                    dl.Polyline(
                        id={'type': 'trajectory', 'index': clicked_trajectory['id']},
                        positions=clicked_trajectory['positions'],
                        color='red',  # 高亮选中的轨迹
                        weight=2,
                        opacity=0.9,
                        n_clicks=0  # 重置点击计数
                    )
                ]


                # 更新轨迹信息
                return_values["info_text"] = f"Trajectory {clicked_trajectory['id']} selected.\n"
                # return_values[
                #     "info_text"] += f"Start: ({clicked_trajectory['start_lat']}, {clicked_trajectory['start_lon']})\n"
                # return_values[
                #     "info_text"] += f"End: ({clicked_trajectory['end_lat']}, {clicked_trajectory['end_lon']})"

                # 更新表格数据
                return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                return_values["selected_trajectory"] = selected_trajectory
                print(return_values["trajectory_layer"])
                break
        return [
            return_values["trajectory_layer"],
            return_values["info_text"],
            return_values["selected_trajectory"],
            return_values["table_data"],
            return_values["mean_rank_graph"],
            return_values["citation-text"]
        ]

    # 如果点击了 Submit 按钮
    elif ctx.triggered and "submit-button" in ctx.triggered[0]['prop_id']:
        if selected_trajectory is not None:
            selected_traj_mr = mr_list[selected_trajectory]  # 获取选中轨迹的 Mean Rank
            selected_traj_mr_base = mr_base_list[selected_trajectory]
            print("selected_mr_base",selected_traj_mr_base)
            overall_mean_rank = means # 整体的 Mean Rank
            base_mean_rank = means_base
            # 创建柱状图
            fig = px.bar(
                x=["Selected(Ours)", "Selected(SOTA[1])", "Overall(Ours)", "Overall(SOTA[1])"],
                y=[selected_traj_mr, selected_traj_mr_base, overall_mean_rank,base_mean_rank],
                labels={"y": "Mean Rank"},
                title="Mean Rank Comparison"
            )

            # 设置柱状图大小和柱子的细度
            fig.update_layout(
                autosize=True,  # 自动调整大小
                width=400,  # 设置宽度
                height=200,  # 设置高度
                margin=dict(l=40, r=40, t=40, b=40),  # 设置四周边距
                bargap=0.5,  # 设置柱子之间的间距，值越小柱子越窄
                yaxis=dict(
                    range=[0.99, 1.1],
                    tickvals=[1, 1.05] , # 只显示 1 和 1.05 这两个刻度
                    dtick=0.005  # 设置更细的刻度间距
                ),
                xaxis=dict(title=None),  # 隐藏x轴的标题
            )

            # 获取 top-k 相似轨迹
            top_k_indices = gt_list[selected_trajectory][:top_k]  # 获取前k条相似轨迹的索引
            top_k_trajectories = [traj_db[i] for i in top_k_indices]

            table = get_table_data(selected_trajectory)
            table_db = get_DB_table_data(top_k_indices)
            for item in table_db:
                table.append(item)
            return_values["table_data"] = table
            # 清空轨迹层并仅添加选中轨迹和前k条相似轨迹
            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': selected_trajectory},
                    positions=traj_list[selected_trajectory]["positions"],
                    color='blue',
                    weight=2,
                    opacity=0.9,
                    n_clicks=0  # 重置点击计数
                )
            ]
            for traj in top_k_trajectories:
                return_values["trajectory_layer"].append(
                    dl.Polyline(
                        id={'type': 'trajectory', 'index': traj['id']},
                        positions=traj['positions'],
                        color='red',  # 红色表示相似轨迹
                        weight=2,
                        opacity=0.7,
                        n_clicks=0
                    )
                )

            # 更新 Mean Rank 图表
            return_values["mean_rank_graph"] = dcc.Graph(figure=fig, style={'width': '100%', 'height': '100%'})
            return_values["citation-text"] = citation_text = "[1]. Tedjopurnomo, David Alexander, et al. \"Similar trajectory search with spatio-temporal deep representation learning.\" ACM Transactions on Intelligent Systems and Technology (TIST) 12.6 (2021): 1-26."
            print("table_data",return_values["table_data"])
        else:

            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': traj['id']},
                    positions=traj['positions'],
                    color=DEFAULT_COLOR,
                    weight=2,
                    opacity=0.8,
                    n_clicks=0  # 重置点击计数
                )
                for traj in traj_list]


    # 如果没有特定事件触发，显示所有轨迹
    elif not return_values["trajectory_layer"]:
        print("进默认了")
        return_values["trajectory_layer"] = [
            dl.Polyline(
                id={'type': 'trajectory', 'index': traj['id']},
                positions=traj['positions'],
                color=DEFAULT_COLOR,
                weight=2,
                opacity=0.8,
                n_clicks=0  # 重置点击计数
            )
            for traj in traj_list
        ]

    return [
        return_values["trajectory_layer"],
        return_values["info_text"],
        return_values["selected_trajectory"],
        return_values["table_data"],
        return_values["mean_rank_graph"],
        return_values['citation-text']
    ]




if __name__ == '__main__':
    app.run_server(debug=True)
