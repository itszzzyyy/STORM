import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl
import numpy as np
import plotly.express as px
from dash import dash_table
import json
import ast
import uuid
from dash.dash_table.Format import Group
DEFAULT_COLOR = "blue"
HIGHLIGHT_COLOR = "red"
def agent_query():
    def process_traj_data(traj_db):
        traj_list = []
        first = traj_db[0][0]
        cnt = 0
        for traj in traj_db:
            traj_id = traj[0]
            points = traj[1]
            lon = points[:, 1]
            lat = points[:, 0]
            traj_list.append({
                'id': traj_id,
                'positions': list(zip(lat, lon)),
                'start_lat': float(lat[0]),
                'start_lon': float(lon[0]),
                'end_lat': float(lat[-1]),
                'end_lon': float(lon[-1]),
            })
            cnt += 1
        return traj_list
    def process_traj_data_db(traj_db,offset=0):
        traj_list = []
        first = traj_db[0][0]
        cnt = 0
        for traj in traj_db:
            traj_id = traj[0]-200+offset
            points = traj[1]
            lon = points[:, 1]
            lat = points[:, 0]
            traj_list.append({
                'id': traj_id,
                'positions': list(zip(lat, lon)),
                'start_lat': float(lat[0]),
                'start_lon': float(lon[0]),
                'end_lat': float(lat[-1]),
                'end_lon': float(lon[-1]),
            })
            cnt += 1
        return traj_list
    def process_mr_data(traj_mr):
        gt_list = {}
        for item in traj_mr:
            if item[0] not in gt_list.keys():
                gt_list[item[0]] = item[1]
        return gt_list

    traj_agent_query = np.load('./agent_query_traj_raw.npy', allow_pickle=True)
    traj_dbs = np.load('./traj_set_raw.npy', allow_pickle=True)
    traj_mr = np.load('./similar_traj.npy', allow_pickle=True)
    traj_list = process_traj_data(traj_agent_query)
    traj_db = process_traj_data_db(traj_dbs,offset=len(traj_list)+1)
    gt_list = process_mr_data(traj_mr)


    def calculate_bounds(traj_list):
        lats = []
        lons = []
        for traj in traj_list:
            for point in traj['positions']:
                lats.append(point[0])
                lons.append(point[1])
        return min(lats), max(lats), min(lons), max(lons)

    min_lat, max_lat, min_lon, max_lon = calculate_bounds(traj_list)


    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    zoom_level = 10
    columns=[
        {"name": "No", "id": "No"},
        {"name": "Trajectory Source", "id": "Trajectory Type"},
        {"name": "Start Latitude", "id": "Start Latitude"},
        {"name": "Start Longitude", "id": "Start Longitude"},
        {"name": "End Latitude", "id": "End Latitude"},
        {"name": "End Longitude", "id": "End Longitude"},
    ]
    def get_table_data(selected_trajectory=None, type = "Trajectory"):
        if selected_trajectory is None:
            return [
                {
                    "No": idx + 1,
                    "Trajectory Type": type,
                    # "Trajectory ID": traj["id"],
                    "Start Latitude": traj["start_lat"],
                    "Start Longitude": traj["start_lon"],
                    "End Latitude": traj["end_lat"],
                    "End Longitude": traj["end_lon"],
                }
                for idx, traj in enumerate(traj_db)
            ]
        else:
            traj = next(traj for traj in traj_db if traj["id"] == selected_trajectory)
            return [({
                "No": 1,
                "Trajectory Type": type,
                "Start Latitude": traj["start_lat"],
                "Start Longitude": traj["start_lon"],
                "End Latitude": traj["end_lat"],
                "End Longitude": traj["end_lon"],
            })]


    def get_table_data_query(selected_trajectory=None, type = "Trajectory"):

        traj = next(traj for traj in traj_list if traj["id"] == selected_trajectory)
        return [({
            "No": 1,
            "Trajectory Type": type,
            "Start Latitude": traj["start_lat"],
            "Start Longitude": traj["start_lon"],
            "End Latitude": traj["end_lat"],
            "End Longitude": traj["end_lon"],
        })]

    def get_DB_table_data(selected_trajectory=None,type= "Query",ids=0):
        traj_to_visulize = []
        for idx in selected_trajectory:

            traj = next(traj for traj in traj_db if traj["id"]-len(traj_list) == idx)
            traj_to_visulize.append(traj)
        return [
            {
                "No": ids+idx,
                "Trajectory Type": 'Top {i} Similar Trajectory'.format(i=idx+1),
                "Start Latitude": traj["start_lat"],
                "Start Longitude": traj["start_lon"],
                "End Latitude": traj["end_lat"],
                "End Longitude": traj["end_lon"],
            }
            for idx, traj in enumerate(traj_to_visulize)
        ]
    app = dash.Dash(__name__)


    app.layout = html.Div([
        dcc.Store(id="selected-trajectory", data=None),

        html.Div([
            html.Div([
                html.H2(
                    "Automatic Similar Trajectory Cluster Detection",
                    style={
                        'textAlign': 'left',
                        'fontFamily': 'Arial, sans-serif',
                        'marginBottom': '30px',
                    }
                ),
                html.Div([
             
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
                    html.Div(
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
                                'display': 'block'
                            }
                        ),
                        style={'textAlign': 'left'}
                    ),


                    html.Div(
                        html.Button(
                            "Similar Trajectory Search",
                            id="submit-button",
                            style={
                                'marginTop': '15px',
                                'marginBottom': '10px',
                                'width': '350px',
                                'fontFamily': 'Arial, sans-serif',
                                'backgroundColor': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'cursor': 'pointer',
                                'display': 'block'
                            }
                        ),
                        style={'textAlign': 'left'}
                    ),
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
                'minHeight': '100vh',
            }),

            html.Div([
                dl.Map(
                    id='map',
                    center=[center_lat, center_lon],
                    zoom=zoom_level,
                        children=[
                                    dl.TileLayer(),
                                    dl.LayerGroup(id="trajectory-layer"),
                                    html.Div(
                                            id="legend",
                                            style={
                                                'position': 'absolute',
                                                'bottom': '10px',
                                                'right': '10px',
                                                'backgroundColor': 'white',
                                                'padding': '10px',
                                                'border': '1px solid black',
                                                'fontFamily': 'Arial, sans-serif',
                                                'fontSize': '14px',
                                                'zIndex': '1000',
                                            }
                                                        )
                                    ],
                    style={
                        'width': '100%',
                        'height': '60vh',
                        'marginBottom': '20px',
                    }
                ),

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
                            {"name": "Trajectory Source", "id": "Trajectory Type"},
                            {"name": "Start Latitude", "id": "Start Latitude"},
                            {"name": "Start Longitude", "id": "Start Longitude"},
                            {"name": "End Latitude", "id": "End Latitude"},
                            {"name": "End Longitude", "id": "End Longitude"},
                        ],
                        data=get_table_data(),
                        style_table={
                            'height': '250px',
                            'overflowY': 'auto',
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
                ]),

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
    @app.callback(
        [
            Output("trajectory-layer", "children"),
            Output("selected-trajectory", "data"),
            Output("trajectory-table", "data"),
            Output("legend", "children"),
            Output('map', 'zoom'),
            Output('map', 'center')
        ],
        [
            Input({'type': 'trajectory', 'index': ALL}, 'n_clicks'),
            Input("reset-button", "n_clicks"),
            Input("submit-button", "n_clicks")
        ],
        [
            State("selected-trajectory", "data"),
            State("top-k-dropdown", "value")
        ]
    )

    def update_trajectory(n_clicks_list, reset_clicked, submit_clicked, selected_trajectory, top_k):
        ctx = dash.callback_context
        return_values = {
            "trajectory_layer": [],
            "selected_trajectory": None,
            "table_data": get_table_data(),
            "legend": None,
            'zoom': zoom_level,
            'center':[center_lat,center_lon]
        }

        if ctx.triggered and "reset-button" in ctx.triggered[0]['prop_id']:
            selected_trajectory = None
            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': traj['id']},
                    positions=traj['positions'],
                    color= 'blue',
                    weight=1,
                    opacity=0.8,
                    n_clicks=0
                )
                for traj in traj_db
            ]

            return_values["selected_trajectory"] = None
            return_values["table_data"] = get_table_data()
            return_values["legend"] = None

            return [
                return_values["trajectory_layer"],
                return_values["selected_trajectory"],
                return_values["table_data"],
                return_values["legend"],
                return_values["zoom"],
                return_values["center"]
            ]

        elif ctx.triggered and "trajectory" in ctx.triggered[0]['prop_id']:

            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks:
                    clicked_trajectory = traj_db[i]
                    return_values["trajectory_layer"] = [
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': clicked_trajectory['id']},
                            positions=clicked_trajectory['positions'],
                            color='blue',
                            weight=2,
                            opacity=0.9,
                            n_clicks=0
                        )
                    ]
                    return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                    return_values["selected_trajectory"] = selected_trajectory
                    return_values["legend"]= None
                    return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                    return_values["selected_trajectory"] = selected_trajectory
                    break
            return [
                return_values["trajectory_layer"],
                return_values["selected_trajectory"],
                return_values["table_data"],
                return_values["legend"],
                return_values["zoom"],
                return_values["center"]
            ]
        elif ctx.triggered and "submit-button" in ctx.triggered[0]['prop_id']:
            return_values["trajectory_layer"] = []

            selected_trajectory = traj_list[0]['id']
            if selected_trajectory is not None:
                top_k_indices = gt_list[selected_trajectory][:top_k]
                top_k_trajectories = []
                data = next(traj for traj in traj_list if traj["id"] == selected_trajectory)
                top_k_trajectories.append(data)
                for ids in top_k_indices:
                    for item in traj_db:
                        if ids == item['id']-len(traj_list):
                            top_k_trajectories.append(item)
                table = get_table_data_query(selected_trajectory,"Query Trajectory")

                table_db = get_DB_table_data(top_k_indices,"Similar Trajectory of STORM",2)

                for item in table_db:
                    table.append(item)
                return_values["table_data"] = table

                for idx, traj in enumerate(top_k_trajectories):

                    color = 'red' if idx == 0 else 'blue'
                    weights = 2 if idx == 0 else 2
                    return_values["trajectory_layer"].append(
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': traj['id']},
                            positions=traj['positions'],
                            color=color,
                            weight=weights,
                            opacity=0.7,
                            n_clicks=0
                        )
                    )


                return_values["legend"] = html.Div([
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '10px', 'backgroundColor': 'red',
                                        'display': 'inline-block'}),
                        html.Span("Query Trajectory", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Div(
                            style={'width': '20px', 'height': '10px', 'backgroundColor': 'blue', 'display': 'inline-block'}),
                        html.Span("Similar Trajectory (STORM)", style={'marginLeft': '5px'})
                    ]),
                ])
                min_lat, max_lat, min_lon, max_lon = calculate_bounds([data])
                center_lat_new = (min_lat + max_lat) / 2
                center_lon_new = (min_lon + max_lon) / 2
                return_values["center"] = [center_lat_new, center_lon_new]
                return_values["zoom"] = 15
        elif not return_values["trajectory_layer"]:
            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': traj['id']},
                    positions=traj['positions'],
                    color=DEFAULT_COLOR,
                    weight=2,
                    opacity=0.8,
                    n_clicks=0
                )
                for traj in traj_db
            ]

        return [
            return_values["trajectory_layer"],
            return_values["selected_trajectory"],
            return_values["table_data"],
            return_values["legend"],
            return_values["zoom"],
            return_values["center"]
        ]
    return app



