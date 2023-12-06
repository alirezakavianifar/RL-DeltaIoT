import plotly
import json
import re
import plotly.graph_objects as go
from src.utility.plot_helper import visualize_data
import seaborn as sns
from src.utility.utils import load_data, \
    utility, return_next_item, scale_data, get_tts_qs, get_tt_qs, create_dummy
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import sys
import plotly.express as px
from plotly.subplots import make_subplots
import itertools

sys.path.append(r'D:\projects\tensorflow_gpu\experiments')
sys.path.append(r'D:\projects\tensorflow_gpu\src\experiments')

VERSION = 'DeltaIoTv1'
PATH = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\DLASeR_plus_online_material\dlaser_plus\raw\%s' % VERSION
# if VERSION == 'DeltaIoTv1':
# PATH = r'D:\projects\papers\Lifelong\generated_data_by_deltaiot_simulation\no_drift_scanerio'
# else:
# PATH = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\DLASeR_plus_online_material\dlaser_plus\raw\DeltaIoTv2'

PATH_TRACE = r'D:\projects\tensorflow_gpu\src\experiments\dqn\results\selected'
Q_EVAL_DIR = r'D:\projects\tensorflow_gpu\experiments\DQN\results\models\DQN_v7DeltaIOT_DeltaIOTAgent_q_eval'

# plot_type could be either scatter or line
PLOT_TYPE = 'scatter'
# for_type could be either quality or uncertainty
FOR_TYPE = 'uncertainty'
# if shuffle true, then data will be shuffled
SHUFFLE = False
# if return_train_test true, split into train and test
RETURN_TRAIN_TEST = False


def load_models(path):
    return tf.keras.models.load_model(path)


def evaluate_models(models):
    data = load_data(path=PATH, load_all=False, version='')
    all_data = return_next_item(data, normalize=False)
    # evals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    evals = defaultdict(lambda: defaultdict(list))
    while (True):
        try:
            df = next(all_data)
            features = df['features'][0][:17]
            predicted_multi = np.argsort(
                models([features]).numpy()[0])[-1:].item()
            evals['energy'].append(
                df.iloc[predicted_multi]['energyconsumption'])
            evals['packet'].append(
                df.iloc[predicted_multi]['packetloss'])
            evals['latency'].append(
                df.iloc[predicted_multi]['latency'])
        except Exception as e:
            print(e)
            break

    return evals


def predict_action(features, model, model_type='1'):
    if model_type == '1':
        predicted_multi = np.argsort(
            model(features)).flatten()[-1:].item()
    elif model_type == '2':
        predicted_multi = model.predict(features)[0].flatten().item()
    return predicted_multi


def test_phase(data, models, energy_coef=None,
               packet_coef=None, latency_coef=None,
               energy_thresh=None, packet_thresh=None, latency_thresh=None,
               num_features=17, cmp=True, algo_name=None,
               quality_type=None, model_type=None, cmp_dir=None, *args, **kwargs):
    log_dir = os.path.join(PATH_TRACE, VERSION)

    if data is None:
        data = load_data(path=PATH, load_all=False, version='')

    all_data = return_next_item(data, normalize=False)
    evals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Compare res with other methods
    i = 0
    res = read_from_html(os.path.join(
        os.getcwd(), 'fig', cmp_dir['tto']))

    while (True):
        try:
            df = next(all_data)

            for keys, values in models.items():
                for key, value in values.items():
                    features = df[['energyconsumption', 'packetloss',
                                   'latency']].iloc[1:2, :].to_numpy()
                    predicted_multi = predict_action(
                        features, value[0], model_type=model_type)
                    evals[keys][key]['energy'].append(
                        df.iloc[predicted_multi]['energyconsumption'])
                    evals[keys][key]['packet'].append(
                        df.iloc[predicted_multi]['packetloss'])
                    evals[keys][key]['latency'].append(
                        df.iloc[predicted_multi]['latency'])
                for k, v in res.items():
                    evals[keys][k]['energy'].append(
                        res[k]['energyconsumption'][0][i])
                    evals[keys][k]['packet'].append(
                        res[k]['packetloss'][0][i])
                    evals[keys][k]['latency'].append(
                        res[k]['latency'][0][i])
                    evals[keys]['Random']['energy'].append(
                        df['energyconsumption'].sample().item())
                    evals[keys]['Random']['packet'].append(
                        df['packetloss'].sample().item())
                    evals[keys]['Random']['latency'].append(
                        df['latency'].sample().item())

            i += 1
        except Exception as e:
            break
    visualize_data(evals,
                   normalize=False,
                   group=True,
                   cmp=cmp,
                   algo_name=algo_name,
                   quality_type=quality_type,
                   model_type=model_type,
                   cmd_dir=cmp_dir
                   )


def read_from_html(file_path):
    with open(file_path) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2**16:])[0]
    lst = json.loads(f'[{call_arg_str}]')[1]
    lst_names = list(dict.fromkeys([ls['name'] for ls in lst]))
    new_lst = []
    for index, item in enumerate(lst_names):
        new_lst.append(itertools.islice(lst, index, None, len(lst_names)))

    res = list(itertools.chain(*new_lst))
    res_indeices = np.split(np.arange(len(lst)), len(lst_names))
    final_res = defaultdict(lambda: defaultdict(list))
    for ind, item in enumerate(res_indeices):
        final_res[res[item[0]]['name']]['latency'].append(res[item[0]]['y'])
        final_res[res[item[0]]['name']]['packetloss'].append(res[item[1]]['y'])
        final_res[res[item[0]]['name']]['energyconsumption'].append(
            res[item[2]]['y'])
    return final_res


# def plot_quality_properties(plot_type=PLOT_TYPE, for_type=PLOT_TYPE, path=PATH):
#     data, _ = load_data(path=path, load_all=False, shuffle=SHUFFLE,
#                         version='', return_train_test=RETURN_TRAIN_TEST)
#     items = return_next_item(data, normalize=False)
#     df = next(items)
#     # plotting the scatter chart
#     if plot_type == 'scatter':
#         fig = px.scatter(df, x="packetloss", y="latency")
#         fig.add_shape(type='line',
#                       x0=0,
#                       y0=5,
#                       x1=8,
#                       y1=5,
#                       line=dict(color='Red',),
#                       xref='x',
#                       yref='y')

#         fig.add_shape(type='line',
#                       x0=8,
#                       y0=0,
#                       x1=8,
#                       y1=5,
#                       line=dict(color='Red',),
#                       xref='x',
#                       yref='y')

#         fig.update_layout(showlegend=True)

#     elif plot_type == 'line':

#         if for_type == 'quality':
#             lst_packet = []
#             lst_energy = []
#             lst_latency = []

#             try:
#                 while (True):
#                     lst_packet.append(df['packetloss'].iloc[23:24].item())
#                     lst_energy.append(
#                         df['energyconsumption'].iloc[23:24].item())
#                     lst_latency.append(df['latency'].iloc[23:24].item())
#                     df = next(items)
#             except Exception as e:
#                 final_df = pd.DataFrame({'Packet loss': lst_packet,
#                                         'Energy consumption': lst_energy,
#                                          'Latency': lst_latency})

#                 final_df = final_df.sample(80)
#                 cols = final_df.columns.tolist()
#                 fig = make_subplots(rows=1, cols=3,
#                                     subplot_titles=("Packet Loss", "Energy Consumption", "Latency"))

#                 for index, value in enumerate(cols):

#                     fig.add_trace(go.Line(
#                         y=final_df[value],
#                         name=value,
#                     ), row=1, col=index+1)

#                     fig.update_xaxes(title_text="cycle", row=1, col=index+1)
#                     fig.update_layout(height=500, width=1500)

#                 fig['layout']['xaxis']['title'] = 'cycle'
#                 fig['layout']['xaxis2']['title'] = 'cycle'
#                 fig['layout']['xaxis2']['title'] = 'cycle'
#                 fig['layout']['yaxis']['title'] = 'Traffic Load'
#                 fig['layout']['yaxis2']['title'] = 'SNR'
#                 fig['layout']['yaxis2']['title'] = 'SNR'

#         elif for_type == 'uncertainty':

#             lst_trafficload = []
#             lst_snr = []

#             try:
#                 while (True):
#                     lst_snr.append(df['features'][0][12:13][0])
#                     lst_trafficload.append(
#                         df['features'][0][27:28][0])
#                     df = next(items)
#             except Exception as e:
#                 final_df = pd.DataFrame({'traffic load': lst_trafficload,
#                                         'snr': lst_snr
#                                          })

#                 # final_df = final_df.sample(100)
#                 final_df = final_df.iloc[:300, :]
#                 cols = final_df.columns.tolist()
#                 fig = make_subplots(rows=1, cols=2,
#                                     subplot_titles=("Traffic Load", "SNR"),
#                                     )

#                 for index, value in enumerate(cols):

#                     fig.add_trace(go.Line(
#                         y=final_df[value],
#                         name=value,
#                     ), row=1, col=index+1)

#                     fig.update_xaxes(title_text="cycle", row=1, col=index+1)
#                     fig.update_layout(height=500, width=1500)

#                 fig['layout']['xaxis']['title'] = 'cycle'
#                 fig['layout']['xaxis2']['title'] = 'cycle'
#                 fig['layout']['yaxis']['title'] = 'Traffic Load'
#                 fig['layout']['yaxis2']['title'] = 'SNR'


# # showing the plot
#     with open("final.html", 'a') as f:
#         f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#     fig.show()

if __name__ == '__main__':
    # plot_quality_properties(plot_type=PLOT_TYPE, for_type=FOR_TYPE, path=PATH)
    res = read_from_html(os.path.join(os.getcwd(), 'fig', 'Fig15-a.htm'))
    print('f')
