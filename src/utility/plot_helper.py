from dash import html, dcc, Dash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from src.utility.constants import mapping, mapping_title, box_plots, \
    selected_on_others, other_plots, subplot_titles


def visualize_v3(cols, group=True, group_col=None, other_plots=['3dsurface'], vesrion='v1'):
    if group_col is not None:
        x = group_col
    figs = []
    traces_all = []
    evals = defaultdict(list)
    evals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, values in cols.items():
        for k, v in values.items():
            for k_, v_ in v.items():
                evals[key][k_][k].append(v_)

    for eval_key, eval_values in evals.items():

        for eval_values_key, eval_values_value in eval_values.items():
            traces = defaultdict(list)
            for eval_values_value_k, eval_values_value_v in eval_values_value.items():
                trace = go.Box(
                    y=eval_values_value_v[0],
                    name=eval_values_value_k.split('=')[1].split('-')[0],
                    # name=eval_values_value_k,
                )
                traces[f'{str(eval_key)}-{str(eval_values_key)}'].append(trace)
                # traces[f'{str(eval_key)}'].append(trace)

            traces_all.append(traces)

    # fig = make_subplots(rows=len(cols), cols=3,
    #                     row_titles=[key.split('*')[1][:-7] for item in traces_all for key, _ in item.items()],
    #                     shared_xaxes=False,
    #                     shared_yaxes=False
    #                     )
    row_titles=[key.split('*')[1][:-7] for item in traces_all for key, _ in item.items()]
    fig = make_subplots(rows=len(cols), cols=3,
                        row_titles=row_titles,
                        shared_xaxes=False,
                        shared_yaxes=False,
                        start_cell='top-left'
                        )
    
    # fig = make_subplots(rows=len(cols), cols=3,
    #                     shared_xaxes=False,
    #                     shared_yaxes=False
    #                     )

    row = 1
    col = 1
    for index, item in enumerate(traces_all):
        if (index > 0 and index % 3 == 0):
            row += 1
            col = 1
        for i, t in item.items():
            for f in t:
                fig.append_trace(f, row=row, col=col)
        col += 1

    fig['layout'].update(height=len(cols) * 350)
    # fig.for_each_annotation(lambda a:  a.update(x = 0.9) if a.text in row_titles else())

    # xaxis titles

    lst_xaxis = [item for item in fig['layout'] if 'xax' in item]
    for item in lst_xaxis:
        fig['layout'][item]['title'] = 'cycle'

    # yaxis titles
    # Sort yaxis titles
    items = [item[5:] for item in fig['layout'] if 'yax' in item]
    num_items = sorted([int(num) for num in items if num != ''])
    lst_yaxis = ['yaxis'] + [f'yaxis{item}' for item in num_items]
    # Split them into equal parts
    lst_yaxis = np.array_split(lst_yaxis, len(cols))

    for items in lst_yaxis:
        for index, item in enumerate(items):
            if index == 0:
                fig['layout'][item]['title'] = 'Energy Consumption (c)'
            elif index == 1:
                fig['layout'][item]['title'] = 'Packet Loss (%)'
            else:
                fig['layout'][item]['title'] = 'Latency (%)'

    figs.append(fig)

    return figs


def visualize_v2(cols, group=True, group_col=None, other_plots=['3dsurface'], vesrion='v1'):
    if group_col is not None:
        x = group_col
    figs = []

    traces_all = []
    for k_, v_ in cols.items():

        traces = []
        for key, values in v_.items():

            trace = go.Box(
                y=values,
                name=key,
            )
            traces.append(trace)

        traces_all.append(traces)

    fig = make_subplots(rows=len(cols), cols=3,
                        # subplot_titles=subp_titles,
                        shared_xaxes=False,
                        shared_yaxes=False
                        )
    row = 1
    col = 1
    for index, item in enumerate(traces_all):
        if (index > 0 and index % 3 == 0):
            row += 1
            col = 1
        for i, t in enumerate(item):
            fig.append_trace(t, row=row, col=col)
        col += 1

    fig['layout'].update(height=900)

    figs.append(fig)

    return figs


def visualize(cols, group=True, group_col=None, other_plots=['3dsurface'], vesrion='v1'):
    if group_col is not None:
        x = group_col
    figs = []

    for k_, v_ in box_plots.items():
        if (k_ == 'single_obj'):
            for key, values in cols.items():
                fig = go.Figure()
                if key in box_plots[k_]:
                    for k, v in values.items():
                        fig.add_trace(go.Box(
                            y=v,
                            # x=x,
                            name=mapping[k],
                        ))

                        if group:
                            fig.update_layout(
                                title={
                                    'text': mapping_title[key],
                                    'y': 0.9,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},
                                yaxis_title=key,
                                # boxmode='group',  # group together boxes of the different traces for each value of x
                            )

                    figs.append(fig)
        elif (k_ == 'multi_obj'):
            if vesrion == 'v1':
                subp_titles = subplot_titles['version1']
            else:
                subp_titles = subplot_titles['version2']

            traces_all = []
            for key, values in cols.items():
                for n, b in box_plots[k_].items():
                    if key in b:
                        traces = []
                        for k, v in values.items():
                            trace = go.Box(
                                y=v,
                                name=k,
                            )
                            traces.append(trace)

                        traces_all.append(traces)

            fig = make_subplots(rows=2, cols=3,
                                subplot_titles=subp_titles,
                                shared_xaxes=True,
                                shared_yaxes=False
                                )
            row = 1
            col = 1
            for index, item in enumerate(traces_all):
                if index == 3:
                    row = 2
                    col = 1
                for i, t in enumerate(item):
                    fig.append_trace(t, row=row, col=col)
                col += 1

            # fig['layout'].update(height=900)
            fig['layout']['yaxis']['title'] = 'Energy Consumption (c)'
            fig['layout']['yaxis2']['title'] = 'Packet Loss (%)'
            fig['layout']['yaxis3']['title'] = 'Latency (%)'
            fig['layout']['yaxis4']['title'] = 'Energy Consumption (C)'
            fig['layout']['yaxis5']['title'] = 'Packet Loss (%)'
            fig['layout']['yaxis6']['title'] = 'Latency (%)'

            figs.append(fig)

            #     fig.update_xaxes(title_text="cycle", row=1, col=index+1)
            #     fig.update_layout(height=500, width=1500)

            # else:
            # df = pd.DataFrame(values)

            # df = pd.DataFrame(dict(
            #     value=[values['selected_item_latency'].tolist(),
            #            values['selected_item_energy_for_latency'].tolist(),
            #            values['selected_item_packet_for_latency'].tolist()],
            #     variable=['V1', 'V2', 'V3']
            # ))
            # cols = df.columns.tolist()
            # fig = px.line_polar(df, r = 'value', theta = 'variable', line_close = True,
            #         line_dash_sequence = ['dash'])

            # df = pd.DataFrame(dict(
            #     value = [67, 8, 3,68, 9, 5],

            #     variable = ['energy consumption', 'packet loss', 'latency','energy consumption', 'packet loss', 'latency'],
            #     # group = ['A', 'A', 'A','B', 'B', 'B']
            #     ))

            # fig = px.line_polar(df, r = 'value', theta = 'variable', line_close = True,
            #                     # color = 'group',
            #                     # color_discrete_map = {'A': 'dodgerblue', 'B': 'gold'}
            #                     )
            # fig.update_traces(fill = 'toself')
            # figs.append(fig)
            #     fig.show()

            #     for item in other_plots:

            #         if item == 'scatter3d':
            #             fig.add_trace(go.Scatter3d(
            #                 x=df[cols[0]], y=df[cols[1]], z=df[cols[2]], mode='markers'))
            #             # fig.add_trace(go.Scatter3d(
            #             #     x=df[cols[0]], y=df[cols[1]], z=df[cols[2]], mode='lines'))

            #             fig.update_layout(
            #                 title=k,
            #                 # autosize=False,
            #                 # width=500,
            #                 # height=500,
            #                 margin=dict(l=65, r=50, b=65, t=90),
            #                 scene=dict(
            #                     xaxis_title=selected_on_others[cols[0]],
            #                     yaxis_title=selected_on_others[cols[1]],
            #                     zaxis_title=selected_on_others[cols[2]],
            #                 ),
            #             )
            #         elif item == '3dsurface':
            #             fig.add_trace(go.Surface(
            #                 x=df[cols[0]], y=df[cols[1]], z=df[cols[2]]))

    return figs


def visualize_data(cols, normalize=True, group=True, group_col=None):

    if normalize:
        scaled_cols = {}
        scaler = MinMaxScaler()
        for col in cols:
            scaled_cols[col.name] = pd.DataFrame(
                scaler.fit_transform(col.to_numpy().reshape(-1, 1)))

        cols = scaled_cols

    app = Dash(__name__)

    figs = visualize_v3(cols=cols, group=group, group_col=group_col)

    divs = []
    divs.append(html.H3("DeltaIoT Case"))

    for fig in figs:
        divs.append(dcc.Graph(id='visual', figure=fig))

    app.layout = html.Div(
        divs
    )

    app.run_server()
