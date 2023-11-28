
mapping = {'selected_item_energy': 'DRL4SAO',
           'selected_item_packet': 'DRL4SAO',
           'selected_item_latency': 'DRL4SAO',
           'selected_item_energy_multi': 'DRL4SAO',
           'selected_item_packet_multi': 'DRL4SAO',
           'selected_item_latency_multi': 'DRL4SAO',
           'TTS_energy_multi': 'Reference',
           'TTS_packet_multi': 'Reference',
           'TTS_latency_multi': 'Reference',
           'selected_item_energy_multi_tt': 'DRL4SAO',
           'selected_item_packet_multi_tt': 'DRL4SAO',
           'selected_item_latency_multi_tt': 'DRL4SAO',
           'TT_energy_multi': 'Reference',
           'TT_packet_multi': 'Reference',
           'TT_latency_multi': 'Reference',
           'real_min_item_energy': 'Reference',
           'real_min_item_packet': 'Reference',
           'real_min_item_latency': 'Reference',
           'real_max_item_energy': 'Reference-max',
           'real_max_item_packet': 'Reference-max',
           'real_max_item_latency': 'Reference-max',
           'random_energy_util': 'Random',
           'random_packet_util': 'Random',
           'random_latency_util': 'Random',
           'energy_DLASeR+': 'DLASeR',
           'packet_DLASeR+': 'DLASeR',
           'latency_DLASeR+': 'DLASeR',
           }

cmp_methods = ["DLASER", "DLASER+", 'Reference', 'Random']

mapping_title = {'EnergyConsumption': 'DRL4SAO',
                 'PacketLoss': 'DRL4SAO',
                 'Latency': 'DRL4SAO',
                 'EnergyConsumptionTTS': 'Set-point: EC in [12.90 +- 0.1]',
                 'PacketLossTTS': 'Threshold: PL < 15%',
                 'LatencyTTS': 'Threshold: LA < 10%',
                 'selected_energy_on_others': 'Selected Energy Consumption impact on other properties',
                 'selected_packet_on_others': 'Selected Packet loss impact on other properties',
                 'selected_latency_on_others': 'Selected Latency impact on other properties',
                 }

box_plots = {'single_obj': {'EnergyConsumption': 'DRL4SAO',
                            'PacketLoss': 'DRL4SAO',
                            'Latency': 'DRL4SAO'},
             'multi_obj': {'tts': {'EnergyConsumptionTTS': 'Set-point: EC in [12.90 +- 0.1]',
                           'PacketLossTTS': 'Threshold: PL < 15%',
                                   'LatencyTTS': 'Threshold: LA < 10%'},
                           'tto': {'EnergyConsumptionTT': 'Minimize EC',
                                   'PacketLossTT': 'Threshold: PL < 15%',
                                   'LatencyTT': 'Threshold: LA < 10%'}}
             }

other_plots = {'selected_energy_on_others': 'energy consumption',
               'selected_packet_on_others': 'packet loss',
               'selected_latency_on_others': 'latency'}

selected_on_others = {'selected_item_energy': 'energy consumption',
                      'selected_item_packet_for_energy': 'packet loss',
                      'selected_item_latency_for_energy': 'latency',
                      'selected_item_packet': 'packet loss',
                      'selected_item_energy_for_packet': 'energy consumption',
                      'selected_item_latency_for_packet': 'latency',
                      'selected_item_latency': 'latency',
                      'selected_item_energy_for_latency': 'energy consumption',
                      'selected_item_packet_for_latency': 'packet loss',
                      }

# subplot_titles = {'version1': ("Set-point: EC in [13.2 ± 0.1]", "PL < 15%", "Threshold: LA < 10%",
#                                "Minimize EC", "PL < 15%", "Threshold: LA < 10%"),
#                   'version2': ("Set-point: EC in [67 ± 0.3]", "PL < 15%", "Threshold: LA < 10%",
#                                "Minimize EC", "PL < 15%", "Threshold: LA < 10%")}

dict_subplot_titles = {
    'DQN_v1':
    {
        'multi': ["Set-point: EC in [12.90 ± 0.1]", "PL < 15%", "Threshold: LA < 10%"],
        'multi_tto': ["Minimize EC", "PL < 15%", "Threshold: LA < 10%"]
    },
    'DQN_v2':
    {
        'multi': ["Set-point: EC in [67 ± 0.3]", "PL < 15%", "Threshold: LA < 10%"],
        'multi_tto': ["Minimize EC", "PL < 15%", "Threshold: LA < 10%"],
    }

}
