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


def test_phase(data, models, energy_coef=None, packet_coef=None, latency_coef=None,
               energy_thresh=None, packet_thresh=None, latency_thresh=None, num_features=17):
    log_dir = os.path.join(PATH_TRACE, VERSION)

    if data is None:
        data = load_data(path=PATH, load_all=False, version='')

    all_data = return_next_item(data, normalize=False)
    evals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    while (True):
        try:
            df = next(all_data)
            for keys, values in models.items():
                for key, value in values.items():
                    features = df[['energyconsumption', 'packetloss',
                                   'latency']].iloc[1:2, :].to_numpy()
                    predicted_multi = np.argsort(value[0](features)).flatten()[-1:].item()
                    evals[keys][key]['energy'].append(
                        df.iloc[predicted_multi]['energyconsumption'])
                    evals[keys][key]['packet'].append(
                        df.iloc[predicted_multi]['packetloss'])
                    evals[keys][key]['latency'].append(
                        df.iloc[predicted_multi]['latency'])
                df_truth=get_tts_qs(df, packet_thresh=15, latency_thresh=10, energy_thresh=13.2)[['energyconsumption','packetloss','latency']]
                evals[keys]['DLASER']['energy'].append(create_dummy(df_truth['energyconsumption'].item()))
                evals[keys]['DLASER']['packet'].append(create_dummy(df_truth['packetloss'].item()))
                evals[keys]['DLASER']['latency'].append(create_dummy(df_truth['latency'].item()))
                evals[keys]['Reference']['energy'].append(df_truth['energyconsumption'].item())
                evals[keys]['Reference']['packet'].append(df_truth['packetloss'].item())
                evals[keys]['Reference']['latency'].append(df_truth['latency'].item())
                evals[keys]['Random']['energy'].append(df['energyconsumption'].sample().item())
                evals[keys]['Random']['packet'].append(df['packetloss'].sample().item())
                evals[keys]['Random']['latency'].append(df['latency'].sample().item())
        except Exception as e:
            print(e)
            break
    visualize_data(evals,
                   normalize=False,
                   group=True,
                   )

#     q_eval_energy = models[0]
#     q_eval_packet = models[1]
#     q_eval_latency = models[2]
#     q_eval_multi = models[3]
#     q_eval_multi_tt = models[4]

#     # lists for calculating precision
#     lst_packet_precision = []
#     lst_energy_precision = []
#     lst_latency_precision = []

#     lst = []
#     lst_selected_energy = []
#     lst_selected_item_packet_for_energy = []
#     lst_selected_item_latency_for_energy = []
#     lst_selected_packet = []
#     lst_selected_item_energy_for_packet = []
#     lst_selected_item_latency_for_packet = []
#     lst_selected_latency = []
#     lst_selected_item_energy_for_latency = []
#     lst_selected_item_packet_for_latency = []
#     # Predicted quality properties arrays for TTS setting
#     lst_selected_energy_multi = []
#     lst_selected_packet_multi = []
#     lst_selected_latency_multi = []
#     # Real quality properties arrays for TTS setting
#     lst_TTS_multi_energy = []
#     lst_TTS_multi_packet = []
#     lst_TTS_multi_latency = []
#     # Predicted quality properties arrays for TT setting
#     lst_selected_energy_multi_tt = []
#     lst_selected_packet_multi_tt = []
#     lst_selected_latency_multi_tt = []
#     # Real quality properties arrays for TT setting
#     lst_TT_multi_tt_energy = []
#     lst_TT_multi_tt_packet = []
#     lst_TT_multi_tt_latency = []

#     lst_min_energy = []
#     lst_max_energy = []
#     lst_min_packet = []
#     lst_max_packet = []
#     lst_min_latency = []
#     lst_max_latency = []
#     lst_rand_energy = []
#     lst_rand_packet = []
#     lst_rand_latency = []

#     all_data = return_next_item(data, normalize=False)
#     for index, item in enumerate(data):
#         df = next(all_data)
#         df['scaled_packet'] = scale_data(df['packetloss'])
#         df['scaled_energy'] = scale_data(df['energyconsumption'])
#         df['scaled_latency'] = scale_data(df['latency'])
#         df['util_energy'] = utility(
#             1.0, 0.0, 0.0, df['energyconsumption'], df['packetloss'], df['latency'])
#         df['util_packet'] = utility(
#             0.0, 1.0, 0.0, df['energyconsumption'], df['packetloss'], df['latency'])
#         df['util_latency'] = utility(
#             0.0, 0.0, 1.0, df['energyconsumption'], df['packetloss'], df['latency'])
#         features = df['features'][0][:num_features]
#         # Utilize the quality models  for predicting the index of selected quality property
#         predicted_energy = np.argsort(
#             q_eval_energy([features]).numpy()[0])[-1:].item()
#         predicted_packet = np.argsort(
#             q_eval_packet([features]).numpy()[0])[-1:].item()
#         predicted_latency = np.argsort(
#             q_eval_latency([features]).numpy()[0])[-1:].item()
#         predicted_multi = np.argsort(
#             q_eval_multi([features]).numpy()[0])[-1:].item()
#         predicted_multi_tt = np.argsort(
#             q_eval_multi_tt([features]).numpy()[0])[-1:].item()
#         # Get the actual value of selected quality proeperty
#         selected_item_energy = df.iloc[predicted_energy]['util_energy']
#         selected_item_packet_for_energy = df.iloc[predicted_energy]['util_packet']
#         selected_item_latency_for_energy = df.iloc[predicted_energy]['util_latency']

#         selected_item_packet = df.iloc[predicted_packet]['util_packet']
#         selected_item_energy_for_packet = df.iloc[predicted_packet]['util_energy']
#         selected_item_latency_for_packet = df.iloc[predicted_packet]['util_latency']

#         selected_item_latency = df.iloc[predicted_latency]['util_latency']
#         selected_item_energy_for_latency = df.iloc[predicted_latency]['util_energy']
#         selected_item_packet_for_latency = df.iloc[predicted_latency]['util_packet']

#         selected_item_energy_multi = df.iloc[predicted_multi]['util_energy']
#         selected_item_packet_multi = df.iloc[predicted_multi]['util_packet']
#         selected_item_latency_multi = df.iloc[predicted_multi]['util_latency']

#         selected_item_energy_multi_tt = df.iloc[predicted_multi_tt]['util_energy']
#         selected_item_packet_multi_tt = df.iloc[predicted_multi_tt]['util_packet']
#         selected_item_latency_multi_tt = df.iloc[predicted_multi_tt]['util_latency']

#         # Real quality properties for the TTS setting
#         df_tts_final = get_tts_qs(
#             df, packet_thresh, latency_thresh, energy_thresh)
#         lst_TTS_multi_energy.append(df_tts_final['energyconsumption'].item())
#         lst_TTS_multi_packet.append(df_tts_final['packetloss'].item())
#         lst_TTS_multi_latency.append(df_tts_final['latency'].item())
#         # Real quality properties for the TT setting
#         df_tt_final = get_tt_qs(
#             df, packet_thresh, latency_thresh, energy_thresh)
#         lst_TT_multi_tt_energy.append(df_tt_final['energyconsumption'].item())
#         lst_TT_multi_tt_packet.append(df_tt_final['packetloss'].item())
#         lst_TT_multi_tt_latency.append(df_tt_final['latency'].item())

#         # Selected Energy
#         lst_selected_energy.append(selected_item_energy)
#         lst_selected_item_packet_for_energy.append(
#             selected_item_packet_for_energy)
#         lst_selected_item_latency_for_energy.append(
#             selected_item_latency_for_energy)

#         # Selected Packet
#         lst_selected_packet.append(
#             selected_item_packet)
#         lst_selected_item_energy_for_packet.append(
#             selected_item_energy_for_packet)
#         lst_selected_item_latency_for_packet.append(
#             selected_item_latency_for_packet)

#         # Selected Latency
#         lst_selected_latency.append(
#             selected_item_latency)
#         lst_selected_item_energy_for_latency.append(
#             selected_item_energy_for_latency)
#         lst_selected_item_packet_for_latency.append(
#             selected_item_packet_for_latency)

#         # Selected TTS
#         lst_selected_energy_multi.append(selected_item_energy_multi)
#         lst_selected_packet_multi.append(selected_item_packet_multi)
#         lst_selected_latency_multi.append(selected_item_latency_multi)

#         # Selected TT
#         lst_selected_energy_multi_tt.append(selected_item_energy_multi_tt)
#         lst_selected_packet_multi_tt.append(selected_item_packet_multi_tt)
#         lst_selected_latency_multi_tt.append(selected_item_latency_multi_tt)

#         # Min Reference qs
#         real_min_item_energy = df['util_energy'].min()
#         real_min_item_packet = df['util_packet'].min()
#         real_min_item_latency = df['util_latency'].min()
#         lst_min_energy.append(real_min_item_energy)
#         lst_min_packet.append(real_min_item_packet)
#         lst_min_latency.append(real_min_item_latency)

#         real_max_item_energy = df['util_energy'].max()
#         real_max_item_packet = df['util_packet'].max()
#         real_max_item_latency = df['util_latency'].max()
#         lst_max_energy.append(real_max_item_energy)
#         lst_max_packet.append(real_max_item_packet)
#         lst_max_latency.append(real_max_item_latency)

#         # Random Reference qs
#         random_energy_util = df['util_energy'].sample().item()
#         random_packet_util = df['util_packet'].sample().item()
#         random_latency_util = df['util_latency'].sample().item()
#         lst_rand_energy.append(random_energy_util)
#         lst_rand_packet.append(random_packet_util)
#         lst_rand_latency.append(random_latency_util)

#         # Calculate precision for energy
#         if (((selected_item_energy < energy_thresh) and (real_min_item_energy <= energy_thresh))
#                 or ((selected_item_energy > energy_thresh) and (real_min_item_energy >= energy_thresh))):
#             lst_energy_precision.append(1)
#         else:
#             lst_energy_precision.append(0)

#         # Calculate precision for packet
#         if (((selected_item_packet < packet_thresh) and (real_min_item_packet <= packet_thresh))
#                 or ((selected_item_packet > packet_thresh) and (real_min_item_packet >= packet_thresh))):
#             lst_packet_precision.append(1)
#         else:
#             lst_packet_precision.append(0)

#         # Calculate precision for latency
#         if (((selected_item_latency < latency_thresh) and (real_min_item_latency <= latency_thresh))
#                 or ((selected_item_latency > latency_thresh) and (real_min_item_latency >= latency_thresh))):
#             lst_latency_precision.append(1)
#         else:
#             lst_latency_precision.append(0)

#         df_final = pd.DataFrame({'selected_item_energy': [selected_item_energy],
#                                  'selected_item_packet_for_energy': [selected_item_packet_for_energy],
#                                  'selected_item_latency_for_energy':  [selected_item_latency_for_energy],
#                                  'selected_item_packet': [selected_item_packet],
#                                  'selected_item_energy_for_packet': [selected_item_energy_for_packet],
#                                  'selected_item_latency_for_packet': [selected_item_latency_for_packet],
#                                  'selected_item_latency': [selected_item_latency],
#                                  'selected_item_energy_for_latency': [selected_item_energy_for_latency],
#                                  'selected_item_packet_for_latency': [selected_item_packet_for_latency],
#                                  'selected_item_energy_multi': [selected_item_energy_multi],
#                                  'selected_item_packet_multi': [selected_item_packet_multi],
#                                  'selected_item_latency_multi': [selected_item_latency_multi],
#                                  'selected_item_energy_multi_tt': [selected_item_energy_multi_tt],
#                                  'selected_item_packet_multi_tt': [selected_item_packet_multi_tt],
#                                  'selected_item_latency_multi_tt': [selected_item_latency_multi_tt],
#                                  'TTS_energy_multi': [df_tts_final['energyconsumption'].item()],
#                                  'TTS_packet_multi': [df_tts_final['packetloss'].item()],
#                                  'TTS_latency_multi': [df_tts_final['latency'].item()],
#                                  'TT_energy_multi': [df_tt_final['energyconsumption'].item()],
#                                  'TT_packet_multi': [df_tt_final['packetloss'].item()],
#                                  'TT_latency_multi': [df_tt_final['latency'].item()],
#                                  'real_min_item_energy': [real_min_item_energy],
#                                  'real_min_item_packet': [real_min_item_packet],
#                                  'real_min_item_latency': [real_min_item_latency],
#                                  'real_max_item_energy': [real_max_item_energy],
#                                  'real_max_item_packet': [real_max_item_packet],
#                                  'real_max_item_latency': [real_max_item_latency],
#                                  'random_energy_util': [random_energy_util],
#                                  'random_packet_util': [random_packet_util],
#                                  'random_latency_util': [random_latency_util],
#                                  'is_drift': ['Selected Item by RL vs Best Item vs Worst Item vs random']})
#         lst.append(df_final)

#     df = pd.concat(lst)



#     df['energy_DLASeR'] = df['TTS_energy_multi'].apply(
#         lambda x: create_dummy(x))
#     df['packet_DLASeR'] = df['TTS_packet_multi'].apply(
#         lambda x: create_dummy(x))
#     df['latency_DLASeR'] = df['TTS_latency_multi'].apply(
#         lambda x: create_dummy(x))

#     #

#     monitored = {'selected_item_energy': lst_selected_energy,
#                  'real_min_item_energy': lst_min_energy,
#                  'real_max_item_energy': lst_max_energy,
#                  'random_energy_util': lst_rand_energy,
#                  'selected_item_packet': lst_selected_packet,
#                  'real_min_item_packet': lst_min_packet,
#                  'real_max_item_packet': lst_max_packet,
#                  'random_packet_util': lst_rand_packet,
#                  'selected_item_latency': lst_selected_latency,
#                  'real_min_item_latency': lst_min_latency,
#                  'real_max_item_latency': lst_max_latency,
#                  'random_latency_util': lst_rand_latency,
#                  'selected_item_energy_multi': lst_selected_energy_multi,
#                  'selected_item_packet_multi': lst_selected_packet_multi,
#                  'selected_item_latency_multi': lst_selected_latency_multi,
#                  'selected_item_energy_multi': lst_selected_energy_multi,
#                  'selected_item_packet_multi': lst_selected_packet_multi,
#                  'selected_item_latency_multi': lst_selected_latency_multi,
#                  'TTS_multi_energy': lst_TT_multi_tt_energy,
#                  'TTS_multi_packet': lst_TT_multi_tt_packet,
#                  'TTS_multi_latency': lst_TT_multi_tt_latency,
#                  'TT_multi_energy': lst_TT_multi_tt_energy,
#                  'TT_multi_packet': lst_TT_multi_tt_packet,
#                  'TT_multi_latency': lst_TT_multi_tt_latency,
#                  }

#     # for key, value in monitored.items():
#     #     train_summary_writer = set_log_dir(path=log_dir, name=key)

#     #     for index, item in tqdm(enumerate(value)):
#     #         with train_summary_writer.as_default():
#     #             tf.summary.scalar('key', item, step=index)

#     visualize_data({'EnergyConsumption': {df['selected_item_energy'].name: df['selected_item_energy'],
#                                           df['real_min_item_energy'].name: df['real_min_item_energy'],
#                                           df['real_max_item_energy'].name: df['real_max_item_energy'],
#                                           df['random_energy_util'].name: df['random_energy_util']},
#                     'PacketLoss': {df['selected_item_packet'].name: df['selected_item_packet'],
#                                    df['real_min_item_packet'].name: df['real_min_item_packet'],
#                                    df['real_max_item_packet'].name: df['real_max_item_packet'],
#                                    df['random_packet_util'].name: df['random_packet_util']},
#                     'Latency': {df['selected_item_latency'].name: df['selected_item_latency'],
#                                 df['real_min_item_latency'].name: df['real_min_item_latency'],
#                                 df['real_max_item_latency'].name: df['real_max_item_latency'],
#                                 df['random_latency_util'].name: df['random_latency_util']},

#                     'EnergyConsumptionTTS': {'DRL4SAO': df['selected_item_energy_multi'],
#                                              'DLASeR': df['energy_DLASeR'],
#                                              'Reference': df['TTS_energy_multi'],
#                                              'max': df['real_max_item_energy'],
#                                              'random': df['random_energy_util']},
#                     'PacketLossTTS': {'DRL4SAO': df['selected_item_packet_multi'],
#                                       'DLASeR': df['packet_DLASeR'],
#                                       'Reference': df['TTS_packet_multi'],
#                                       'max': df['real_max_item_packet'],
#                                       'random': df['random_packet_util']},
#                     'LatencyTTS': {'DRL4SAO': df['selected_item_latency_multi'],
#                                    'DLASeR': df['latency_DLASeR'],
#                                    'Reference': df['TTS_latency_multi'],
#                                    'max': df['real_max_item_latency'],
#                                    'random': df['random_latency_util']},
#                     'EnergyConsumptionTT': {'DRL4SAO': df['selected_item_energy_multi_tt'],
#                                             'DLASeR': df['energy_DLASeR'],
#                                             'Reference': df['TT_energy_multi'],
#                                             'max': df['real_max_item_energy'],
#                                             'random': df['random_energy_util']},
#                     'PacketLossTT': {'DRL4SAO': df['selected_item_packet_multi_tt'],
#                                      'DLASeR': df['packet_DLASeR'],
#                                      'Reference': df['TT_packet_multi'],
#                                      'max': df['real_max_item_packet'],
#                                      'random': df['random_packet_util']},
#                     'LatencyTT': {'DRL4SAO': df['selected_item_latency_multi_tt'],
#                                   'DLASeR': df['latency_DLASeR'],
#                                   'Reference': df['TT_latency_multi'],
#                                   'max': df['real_max_item_latency'],
#                                   'random': df['random_latency_util']},
#                     'selected_energy_on_others': {df['selected_item_energy'].name: df['selected_item_energy'],
#                                                   df['selected_item_packet_for_energy'].name: df['selected_item_packet_for_energy'],
#                                                   df['selected_item_latency_for_energy'].name: df['selected_item_latency_for_energy']},
#                     'selected_packet_on_others': {df['selected_item_packet'].name: df['selected_item_packet'],
#                                                   df['selected_item_energy_for_packet'].name: df['selected_item_energy_for_packet'],
#                                                   df['selected_item_latency_for_packet'].name: df['selected_item_latency_for_packet']},
#                     'selected_latency_on_others': {df['selected_item_latency'].name: df['selected_item_latency'],
#                                                    df['selected_item_energy_for_latency'].name: df['selected_item_energy_for_latency'],
#                                                    df['selected_item_packet_for_latency'].name: df['selected_item_packet_for_latency']},
#                     },

#                    normalize=False,
#                    group=True,
#                    group_col=df['is_drift'])

#     print('... models loaded successfully ...')


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


# if __name__ == '__main__':
#     plot_quality_properties(plot_type=PLOT_TYPE, for_type=FOR_TYPE, path=PATH)
