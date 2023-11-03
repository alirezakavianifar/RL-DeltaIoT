import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def utility(energy_coef, packet_coef, latency_coef, energy_consumption, packet_loss, latency):
    
    return (energy_coef * energy_consumption + packet_coef * packet_loss + latency_coef * latency)

def scale_data(data):
    '''
    This function scales the data
    '''
    data = data.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data

def return_next_item(lst, normalize=True, normalize_cols=['energyconsumption', 'packetloss', 'latency']):
    '''
    A generator function which returns the next data frame from given repository
    '''
    for item in lst:
        df = pd.read_json(item)
        if normalize:
            scaler = MinMaxScaler()
            for item in normalize_cols:
                df[item] = scale_data(df[item])
        yield df

