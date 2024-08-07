from src.utility.utils import plot_adaptation_spaces, plot_latency_vs_packet_loss, load_and_prepare_data, plot_metrics_vs_configurations

def load_and_plot_data(st, data_dir, from_cycles, to_cycles):
    LST_PACKET, LST_ENERGY, LST_LATENCY, df = load_and_prepare_data(data_dir, from_cycles=from_cycles, to_cycles=to_cycles)
    plot_adaptation_spaces(st, df, from_cycles=from_cycles, to_cycles=to_cycles)
    plot_metrics_vs_configurations(st, LST_LATENCY, LST_PACKET, LST_ENERGY)
