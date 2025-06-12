import agent_query
import query_search
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl
# import dash_table
import numpy as np
import plotly.express as px
from dash import dash_table
import dash
from dash import html, dcc, Input, Output
import multiprocessing
import webbrowser
import time
import os
import signal
import pandas as pd
class AppManager:
    def __init__(self):
        self.active_apps = {}
        self.app_counter = 0
        self.lock = multiprocessing.Lock()

    def get_app_info(self, app_id):
        if app_id in self.active_apps:
            info = self.active_apps[app_id].copy()
            info['status'] = 'running' if info['process'].is_alive() else 'stopped'
            info['uptime'] = int(time.time() - info['start_time'])
            return info
        return None

    def start_sub_app(self, app_type):
        with self.lock:
            self.app_counter += 1
            port = 8050 + self.app_counter
            process = multiprocessing.Process(
                target=run_sub_app,
                args=(app_type, port, self.app_counter)
            )
            process.start()

            self.active_apps[self.app_counter] = {
                'process': process,
                'port': port,
                'type': app_type
            }

            time.sleep(2)
            webbrowser.open(f'http://localhost:{port}')
            return self.app_counter

    def stop_sub_app(self, app_id):
        with self.lock:
            if app_id in self.active_apps:
                try:
                    os.kill(self.active_apps[app_id]['process'].pid, signal.SIGTERM)
                except ProcessLookupError:
                    print("kill failed")
                    pass
                del self.active_apps[app_id]


app_manager = AppManager()

main_app = dash.Dash(__name__)
main_app.layout = html.Div([
    html.H1("STORM Demo", style={'textAlign': 'center'}),
    html.Div([
html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Button("Activate Similar Trajectory Search",
                                  id="btn-storm",
                                  style={
                                      'fontSize': '20px',
                                      'padding': '15px 30px',
                                      'width': '500px',
                                      'marginLeft': '0',
                                      'marginRight': 'auto'
                                  })
                    ], style={'margin': '15px 0'}),
                    html.Div([
                        html.Button("Activate Automatic Similar Trajectory Cluster Detection",
                                  id="btn-behavior",
                                  style={
                                      'fontSize': '20px',
                                      'padding': '15px 30px',
                                      'width': '500px',
                                      'marginLeft': '0',
                                      'marginRight': 'auto'
                                  })
                    ], style={'margin': '15px 0'}),

                    html.Div([
                        html.Button("Deactivate Similar Trajectory Search",
                                  id="btn-stop-storm",
                                  style={
                                      'fontSize': '20px',
                                      'padding': '15px 30px',
                                      'width': '500px',
                                      'marginLeft': '0',
                                      'marginRight': 'auto'
                                  })
                    ], style={'margin': '15px 0'}),

                    html.Div([
                        html.Button("Deactivate Automatic Similar Trajectory Cluster Detection",
                                  id="btn-stop-behavior",
                                  style={
                                      'fontSize': '20px',
                                      'padding': '15px 30px',
                                      'width': '500px',
                                      'marginLeft': '0',
                                      'marginRight': 'auto'
                                  })
                    ], style={'margin': '15px 0'}),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'flex-start',
                    'padding': '0',
                    'margin': '0'
                }),
                dcc.Upload(
                    id='uploader',
                    children=html.Div([
                        'Click to select files or drag files here',
                        html.Br(),
                    ]),
                    style={
                        'width': '500px',
                        'height': '100px',
                        'lineHeight': '100px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '20px 0',
                        'cursor': 'pointer',
                        'backgroundColor': '#f8f9fa',
                        'color': '#6c757d',
                        'borderColor': '#ced4da',
                        ':hover': {
                            'borderColor': '#28a745',
                            'backgroundColor': '#e2f0e9'
                        }
                    },
                    multiple=False
                ),
                html.Div(id='file-info', style={'marginBottom': '20px'}),

                dcc.Interval(id='status-check', interval=1000),
                html.Div(id='active-apps-list',
                       style={'width': '80%', 'margin': '40px auto'}),
            ], style={
                'width': '100%',
                'padding': '0',
                'margin': '0'
            })
        ], style={
            'width': '40%',
            'padding': '0 20px 0 0',
            'marginLeft': '0'
        }),
        html.Div([
            dl.Map(
                id='map',
                center=[29, -89],
                zoom=10,
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
                    'height': '75vh',
                    'margin': '20px 0',
                    'position': 'relative'
                }
            )
        ], style={'width': '60%', 'padding': '0 20px'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'margin': '0 auto',
        'maxWidth': '1600px'
    }),

    dcc.Location(id='main-url', refresh=True)
])
@main_app.callback(
    Output('btn-stop-storm', 'n_clicks'),
    Input('btn-stop-storm', 'n_clicks')
)
def stop_storm(n):
    if n:
        for app_id in list(app_manager.active_apps.keys()):
            if app_manager.active_apps[app_id]['type'] == 'storm':
                app_manager.stop_sub_app(app_id)
    return None

@main_app.callback(
    Output('btn-stop-behavior', 'n_clicks'),
    Input('btn-stop-behavior', 'n_clicks')
)
def stop_behavior(n):
    if n:
        for app_id in list(app_manager.active_apps.keys()):
            if app_manager.active_apps[app_id]['type'] == 'agent_query':
                app_manager.stop_sub_app(app_id)
    return None

@main_app.callback(
    Output('active-apps-list', 'children'),
    Input('status-check', 'n_intervals')
)
def update_active_apps(_):
    items = []
    for app_id, info in app_manager.active_apps.items():
        status = "EXECUTING" if info['process'].is_alive() else "Terminated"
        if info['type'] == "storm":
            items.append(html.Li([
                html.Span(f" Trajectory Similarity Search Application (port: {info['port']}) "),
                html.Span(status,
                          style={'color': 'green' if 'EXECUTING' in status else 'red'})
            ], style={'margin': '10px'}))
        else:
            items.append(html.Li([
                html.Span(f" Macro-Scale Spatiotemporal Behavior Discovery System (port: {info['port']}) "),
                html.Span(status,
                          style={'color': 'green' if 'EXECUTING' in status else 'red'})
            ], style={'margin': '10px'}))
    return html.Ul(items)


@main_app.callback(
    Output('btn-storm', 'n_clicks'),
    Input('btn-storm', 'n_clicks')
)
def launch_storm(n):
    if n:
        app_manager.start_sub_app('storm')
    return None


@main_app.callback(
    Output('btn-behavior', 'n_clicks'),
    Input('btn-behavior', 'n_clicks')
)
def launch_behavior(n):
    if n:
        app_manager.start_sub_app('agent_query')
    return None


@main_app.callback(
    Output('file-info', 'children'),
    Input('uploader', 'contents'),
    [State('uploader', 'filename'),
     State('uploader', 'last_modified')]
)
def get_file_info(content, filename, last_modified):
    if content is not None:
        return html.Div([
            html.P(f"Filename：{filename}"),
        ])

    return "Awaiting File Transfer Completion..."
def run_sub_app(app_type, port, app_id):
    try:
        if app_type == 'agent_query':
            sub_app = agent_query.agent_query()
            sub_app.run(port=port, debug=False)

        elif app_type == 'storm':
            # 类似地创建STORM应用
            sub_app = query_search.query_search(port=port)
            sub_app.run(port=port, debug=False)

    except Exception as e:
        print(f"Subsystem Activation Aborted: {str(e)}")
    finally:
        app_manager.stop_sub_app(app_id)


if __name__ == '__main__':
    main_app.run(port=8049)
