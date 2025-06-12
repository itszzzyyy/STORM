import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl
# import dash_table
import numpy as np
import plotly.express as px
from dash import dash_table
from dash.dash_table.Format import Group
DEFAULT_COLOR = "blue"
HIGHLIGHT_COLOR = "red"
def query_search(port):
    def process_traj_data(traj_db, offset=0):
        traj_list = []
        first = traj_db[0][0]
        print("first", first)
        cnt = 0
        for traj in traj_db:
            traj_id = traj[0] - first + offset
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
        mr_list = []
        gt_list = []
        for pairs in traj_mr:
            mr_list.append(pairs[1])
            gt_list.append(pairs[2])
        return mr_list, gt_list

    traj_q = np.load('./1_raw_q_demodis50.npy', allow_pickle=True)
    traj_mr = np.load('./mean_rank.npy', allow_pickle=True)
    traj_mr_base = np.load('./mean_rank_b1.npy', allow_pickle=True)
    traj_mr_base2 = np.load('./mean_rank_b2.npy', allow_pickle=True)
    traj_mr_base3 = np.load('./mean_rank_t2vec.npy', allow_pickle=True)
    traj_dbs = np.load('./1_raw_db_demodis50.npy', allow_pickle=True)
    traj_list = process_traj_data(traj_q)
    traj_db = process_traj_data(traj_dbs,len(traj_list))
    mr_list, gt_list = process_mr_data(traj_mr)
    mr_base_list,gt_base_list = process_mr_data(traj_mr_base)
    mr_base2_list,gt_base2_list = process_mr_data(traj_mr_base2)
    mr_base3_list,gt_base3_list = process_mr_data(traj_mr_base3)
    means_base = mean(mr_list)
    means_1 = mean(mr_base_list)
    means_2 = mean(mr_base2_list)
    means_3 = mean(mr_base3_list)

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

    def get_table_data(selected_trajectory=None, type = "Query"):
        if selected_trajectory is None:

            return [
                {
                    "No": idx + 1,
                    "Trajectory Type": type,
                    "Start Latitude": traj["start_lat"],
                    "Start Longitude": traj["start_lon"],
                    "End Latitude": traj["end_lat"],
                    "End Longitude": traj["end_lon"],
                }
                for idx, traj in enumerate(traj_list)
            ]
        else:

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
                "Trajectory Type": type,
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
                    "Similar Trajectory Search",
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
                            'height': '40px',
                            'boxSizing': 'border-box',
                            'overflow': 'hidden',
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



                html.Div(id="mean-rank-graph", style={'marginTop': '20px', 'width': '100%', 'height': '400px'}),
                html.Div(id="overall-mean-rank-graph", style={'marginTop': '20px', 'width': '100%', 'height': '200px'}),
                html.Div(id="citation-text", style={
                                'marginTop': '40px',
                                'fontFamily': 'Times New Roman, serif',
                                'fontSize': '12px',
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
                'minHeight': '100vh',
            }),

            html.Div([

                dl.Map(
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
    selected_id = None
    @app.callback(
        [
            Output("trajectory-layer", "children"),
            Output("trajectory-info", "children"),
            Output("selected-trajectory", "data"),
            Output("trajectory-table", "data"),
            Output("mean-rank-graph", "children"),
            Output("overall-mean-rank-graph", "children"),
            Output("citation-text", "children"),
            Output("legend", "children")
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
        global selected_id
        return_values = {
            "trajectory_layer": [],
            "info_text": "",
            "selected_trajectory": None,
            "table_data": get_table_data(),
            "mean_rank_graph": None,
            "overall_mean_rank":None,
            "citation-text":None,
            "legend": None
        }

        if ctx.triggered and "reset-button" in ctx.triggered[0]['prop_id']:
            selected_trajectory = None
            selected_id = None

            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': traj['id']},
                    positions=traj['positions'],
                    color= 'blue',
                    weight=2,
                    opacity=0.8,
                    n_clicks=0
                )
                for traj in traj_list
            ]

            return_values["info_text"] = "Reset: Displaying all trajectories."
            return_values["selected_trajectory"] = None
            return_values["mean_rank_graph"] = None
            return_values["overall_mean_rank"] = None
            return_values["table_data"] = get_table_data()
            return_values["legend"] = None

            if len(n_clicks_list) < 200:
                n_clicks_list = [0] * (200 - len(n_clicks_list)) + n_clicks_list

            return [
                return_values["trajectory_layer"],
                return_values["info_text"],
                return_values["selected_trajectory"],
                return_values["table_data"],
                return_values["mean_rank_graph"],
                return_values["overall_mean_rank"],
                return_values["citation-text"],
                return_values["legend"]
            ]


        elif ctx.triggered and "trajectory" in ctx.triggered[0]['prop_id']:

            for i, n_clicks in enumerate(n_clicks_list):
                if n_clicks and len(n_clicks_list)!=len(traj_list):
                    clicked_trajectory = traj_list[selected_id]
                    selected_trajectory = selected_id
                    return_values["trajectory_layer"] = [
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': selected_id},
                            positions=clicked_trajectory['positions'],
                            color='red',
                            weight=2,
                            opacity=0.9,
                            n_clicks=0
                        )
                    ]
                    return_values["info_text"] = f"Trajectory {clicked_trajectory['id']} selected.\n"
                    return_values[
                        "info_text"] += f"Start: ({clicked_trajectory['start_lat']}, {clicked_trajectory['start_lon']})\n"
                    return_values[
                        "info_text"] += f"End: ({clicked_trajectory['end_lat']}, {clicked_trajectory['end_lon']})"

                    return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                    return_values["selected_trajectory"] = selected_trajectory
                    return_values["legend"]= None
                elif n_clicks:
                    clicked_trajectory = traj_list[i]
                    selected_trajectory = clicked_trajectory['id']
                    selected_id = selected_trajectory

                    return_values["trajectory_layer"] = [
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': clicked_trajectory['id']},
                            positions=clicked_trajectory['positions'],
                            color='red',
                            weight=2,
                            opacity=0.9,
                            n_clicks=0
                        )
                    ]
                    return_values["legend"] = None



                    return_values["info_text"] = f"Trajectory {clicked_trajectory['id']} selected.\n"



                    return_values["table_data"] = get_table_data(clicked_trajectory['id'])
                    return_values["selected_trajectory"] = selected_trajectory
                    break
            return [
                return_values["trajectory_layer"],
                return_values["info_text"],
                return_values["selected_trajectory"],
                return_values["table_data"],
                return_values["mean_rank_graph"],
                return_values["overall_mean_rank"],
                return_values["citation-text"],
                return_values["legend"]
            ]


        elif ctx.triggered and "submit-button" in ctx.triggered[0]['prop_id']:
            if selected_trajectory is not None:
                selected_traj_mr = mr_list[selected_trajectory]
                selected_traj_mr_base = mr_base_list[selected_trajectory]
                selected_traj_mr_base2 = mr_base2_list[selected_trajectory]
                selected_traj_mr_base3 = mr_base3_list[selected_trajectory]
                overall_mean_rank = means
                base_mean_rank = means_base

                fig = px.bar(
                    x=["STORM", "Autoencoder[1]", "RSTS[2]", "t2vec[3]"],
                    y=[selected_traj_mr, selected_traj_mr_base, selected_traj_mr_base2,selected_traj_mr_base3],
                    labels={"y": "Mean Rank"},
                    title="Selected Trajectory Mean Rank Comparison"
                )


                fig.update_layout(
                    autosize=True,
                    width=400,
                    height=400,
                    margin=dict(l=60, r=60, t=60, b=60),
                    bargap=0.5,
                    yaxis=dict(
                        range=[0, 10],
                        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ,
                        dtick=5
                    ),
                    xaxis=dict(title=None),
                )
                fig2 = px.bar(
                    x=["STORM", "Autoencoder[1]", "RSTS[2]", "t2vec[3]"],
                    y=[means, means_1, means_2, means_3],
                    labels={"y": "Mean Rank"},
                    title="Overall Mean Rank Comparison"
                )


                fig2.update_layout(
                    autosize=True,
                    width=400,
                    height=200,
                    margin=dict(l=40, r=40, t=40, b=40),
                    bargap=0.5,
                    yaxis=dict(
                        range=[0, 4],
                        tickvals=[1,2,3,4],
                        dtick=5
                    ),
                    xaxis=dict(title=None),
                )


                top_k_indices = gt_list[selected_trajectory][:top_k]
                top_k_trajectories = [traj_db[i] for i in top_k_indices]
                top_k_indices_b1 = gt_base_list[selected_trajectory][:top_k]
                top_k_indices_b2 = gt_base2_list[selected_trajectory][:top_k]
                top_k_indices_b3 = gt_base3_list[selected_trajectory][:top_k]
                top_k_trajectories_b1 = [traj_db[i] for i in top_k_indices_b1]
                top_k_trajectories_b2 = [traj_db[i] for i in top_k_indices_b2]
                top_k_trajectories_b3 = [traj_db[i] for i in top_k_indices_b3]

                table = get_table_data(selected_trajectory,"Query Trajectory")
                table_db = get_DB_table_data(top_k_indices,"Similar Trajectory of STORM",2)
                table_db_b1 = get_DB_table_data(top_k_indices_b1, "Similar Trajectory of Autoencoder",3)
                table_db_b2 = get_DB_table_data(top_k_indices_b2, "Similar Trajectory of RSTS",4)
                table_db_b3 = get_DB_table_data(top_k_indices_b3, "Similar Trajectory of t2vec",5)

                for item in table_db:
                    table.append(item)
                for item in table_db_b1:
                    table.append(item)
                for item in table_db_b2:
                    table.append(item)
                for item in table_db_b3:
                    table.append(item)
                return_values["table_data"] = table

                return_values["trajectory_layer"] = [
                    dl.Polyline(
                        id={'type': 'trajectory', 'index': selected_trajectory},
                        positions=traj_list[selected_trajectory]["positions"],
                        color='blue',
                        weight=2,
                        opacity=0.9,
                        n_clicks=0
                    )
                ]
                for traj in top_k_trajectories:

                    return_values["trajectory_layer"].append(
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': traj['id']},
                            positions=traj['positions'],
                            color='red',
                            weight=3,
                            opacity=0.7,
                            n_clicks=0
                        )
                    )

                for traj in top_k_trajectories_b1:
                    return_values["trajectory_layer"].append(
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': traj['id']},
                            positions=traj['positions'],
                            color='green',
                            weight=3,
                            opacity=0.7,
                            n_clicks=0
                        )
                    )

                for traj in top_k_trajectories_b2:
                    return_values["trajectory_layer"].append(
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': traj['id']},
                            positions=traj['positions'],
                            color='black',
                            weight=3,
                            opacity=0.7,
                            n_clicks=0
                        )
                    )

                for traj in top_k_trajectories_b3:
                    return_values["trajectory_layer"].append(
                        dl.Polyline(
                            id={'type': 'trajectory', 'index': traj['id']},
                            positions=traj['positions'],
                            color='purple',
                            weight=3,
                            opacity=0.7,
                            n_clicks=0
                        )
                    )



                return_values["mean_rank_graph"] = dcc.Graph(figure=fig, style={'width': '100%', 'height': '100%'})
                return_values["overall_mean_rank"] = dcc.Graph(figure=fig2, style={'width': '100%', 'height': '100%'})
                return_values["citation-text"] =None

                return_values["legend"] = html.Div([
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '10px', 'backgroundColor': 'blue',
                                        'display': 'inline-block'}),
                        html.Span("Query Trajectory", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Div(
                            style={'width': '20px', 'height': '10px', 'backgroundColor': 'red', 'display': 'inline-block'}),
                        html.Span("Similar Trajectory (STORM)", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '10px', 'backgroundColor': 'green',
                                        'display': 'inline-block'}),
                        html.Span("Similar Trajectory (Autoencoder)", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '10px', 'backgroundColor': 'black',
                                        'display': 'inline-block'}),
                        html.Span("Similar Trajectory (RSTS)", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '10px', 'backgroundColor': 'purple',
                                        'display': 'inline-block'}),
                        html.Span("Similar Trajectory (t2vec)", style={'marginLeft': '5px'})
                    ]),

                ])
            else:

                return_values["trajectory_layer"] = [
                    dl.Polyline(
                        id={'type': 'trajectory', 'index': traj['id']},
                        positions=traj['positions'],
                        color=DEFAULT_COLOR,
                        weight=2,
                        opacity=0.8,
                        n_clicks=0
                    )
                    for traj in traj_list]


        elif not return_values["trajectory_layer"]:

            return_values["trajectory_layer"] = [
                dl.Polyline(
                    id={'type': 'trajectory', 'index': traj['id']},
                    positions=traj['positions'],
                    color=DEFAULT_COLOR if traj['id'] != 126 else 'red',
                    weight=2,
                    opacity=0.8,
                    n_clicks=0
                )
                for traj in traj_list
            ]

        return [
            return_values["trajectory_layer"],
            return_values["info_text"],
            return_values["selected_trajectory"],
            return_values["table_data"],
            return_values["mean_rank_graph"],
            return_values["overall_mean_rank"],
            return_values['citation-text'],
            return_values["legend"]
        ]

    return app


