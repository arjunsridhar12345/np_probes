import np_session
import pathlib
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
from np_probes.utils import get_probe_metrics_path
import uuid

def clean_region(region:str) -> str:
    if pd.isna(region):
        return 'No Area'
    else:
        return region

def get_annotation_volume() -> np.ndarray:
    ccf_annotation_array = sitk.GetArrayFromImage(sitk.ReadImage(pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/tissuecyte/', 
                                                                              'field_reference', 'ccf_ano.mhd')))

    return ccf_annotation_array

def get_day(session:np_session.Session) -> str:
    session_directory = session.npexp_path.parent
    sessions_mouse = sorted(list(session_directory.glob('*{}*'.format(session.mouse))))
    day = [i + 1 for i in range(len(sessions_mouse)) if str(session.id) in str(sessions_mouse[i])][0]
    return str(day)

def get_channels_info_for_probe(current_probe:str, probe_id:int, session:np_session.Session, id_json_dict:dict[str, list]) -> list:
    channels = []

    if len(id_json_dict['channel_ids']) > 0:
        channel_id = id_json_dict['channel_ids'][-1] + 1
    else:
        channel_id = probe_id

    # get channel dataframe
    mouse_id = str(session.mouse)
    probe = current_probe[-1]
    day = get_day(session)

    ccf_alignment_path = pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/tissuecyte', mouse_id, 
                                             'Probe_{}_channels_{}_warped.csv'.format(probe+day, mouse_id))

    if ccf_alignment_path.exists():
        df_ccf_coords = pd.read_csv(ccf_alignment_path)

        ccf_annotation_array = get_annotation_volume()

        vertical_position = 20
        horizontal_position_even = [43, 59]
        horizontal_position = horizontal_position_even[0]
        horizontal_position_even_index = 0
        horizontal_position_odd = [11, 27]
        horizontal_position_odd_index = 0

        for index, row in df_ccf_coords.iterrows():
            if index != 0 and index % 2 == 0:
                vertical_position += 20
            
            if index == 0:
                horizontal_position = horizontal_position_even[0]
            elif index == 1:
                horizontal_position = horizontal_position_odd[0]
            elif index != 0 and index % 2 == 0:
                if horizontal_position_even_index == 0:
                    horizontal_position_even_index = 1
                    horizontal_position = horizontal_position_even[horizontal_position_even_index]
                else:
                    horizontal_position_even_index = 0
                    horizontal_position = horizontal_position_even[horizontal_position_even_index]
            elif index != 1 and index % 1 == 0:
                if horizontal_position_odd_index == 0:
                    horizontal_position_odd_index = 1
                    horizontal_position = horizontal_position_odd[horizontal_position_odd_index]
                else:
                    horizontal_position_odd_index = 0
                    horizontal_position = horizontal_position_odd[horizontal_position_odd_index]

            channel_dict = {
                'probe_id': probe_id,
                'probe_channel_number': row.channel,
                'structure_id': int(ccf_annotation_array[row.AP, row.DV, row.ML]),
                'structure_acronym': clean_region(row.region),
                'anterior_posterior_ccf_coordinate': float(row.AP*25),
                'dorsal_ventral_ccf_coordinate': float(row.DV*25),
                'left_right_ccf_coordinate': float(row.ML*25),
                'probe_horizontal_position': horizontal_position,
                'probe_vertical_position': vertical_position,
                'id': channel_id,
                'valid_data': True
            }

            channels.append(channel_dict)
            id_json_dict['channel_ids'].append(channel_id)

            channel_id += 1
    else:
        for i in range(384):
            channel_dict = {
                'probe_id': probe_id,
                'probe_channel_number': i,
                'structure_id': -1,
                'structure_acronym': 'No Area',
                'anterior_posterior_ccf_coordinate': -1.0,
                'dorsal_ventral_ccf_coordinate': -1.0,
                'left_right_ccf_coordinate': -1.0,
                'probe_horizontal_position': -1,
                'probe_vertical_position': -1,
                'id': channel_id,
                'valid_data': True
            }
            
            channels.append(channel_dict)
            id_json_dict['channel_ids'].append(channel_id)

            channel_id += 1

    return channels

def get_units_info_for_probe(current_probe:str, probe_metrics_path:dict, session:np_session.Session, channels:list[dict], id_json_dict:dict):
    probe_metrics_csv_file = probe_metrics_path[current_probe[-1]]

    if '_test' in str(probe_metrics_csv_file):
        df_metrics = pd.read_csv(probe_metrics_csv_file)
        df_waveforms = pd.read_csv(pathlib.Path(probe_metrics_csv_file.parent, 'waveform_metrics.csv'))

        df_metrics = df_metrics.merge(df_waveforms, on='cluster_id')
    else:
        df_metrics = pd.read_csv(probe_metrics_csv_file)

    df_metrics.fillna(0, inplace=True)

    local_index = 0
    
    if len(id_json_dict['unit_ids']) > 0:
        unit_id = id_json_dict['unit_ids'][-1] + 1
    else:
        unit_id = 0

    units = []
    
    for index, row in df_metrics.iterrows():
        unit_dict = {
            'peak_channel_id': channels[row.peak_channel]['id'],
            'cluster_id': row.cluster_id,
            'quality': row.quality if 'quality' in df_metrics.columns else 'good',
            'snr': row.snr if not pd.isna(row.snr) and not np.isinf(row.snr) else 0,
            'firing_rate': row.firing_rate if not pd.isna(row.firing_rate) else 0,
            'isi_violations': row.isi_viol if not pd.isna(row.isi_viol) else 0,
            'presence_ratio': row.presence_ratio if not pd.isna(row.presence_ratio) else 0,
            'amplitude_cutoff': row.amplitude_cutoff if not pd.isna(row.amplitude_cutoff) else 0,
            'isolation_distance': row.isolation_distance if not pd.isna(row.isolation_distance) else 0,
            'l_ratio': row.l_ratio if not pd.isna(row.l_ratio) else 0,
            'd_prime': row.d_prime if not pd.isna(row.d_prime) else 0,
            'nn_hit_rate': row.nn_hit_rate if not pd.isna(row.nn_hit_rate) else 0,
            'nn_miss_rate': row.nn_miss_rate if not pd.isna(row.nn_miss_rate) else 0,
            'silhouette_score': row.silhouette_score if not pd.isna(row.silhouette_score) else 0,
            'max_drift': row.max_drift if not pd.isna(row.max_drift) else 0,
            'cumulative_drift': row.cumulative_drift if not pd.isna(row.cumulative_drift) else 0,
            'waveform_duration': row.duration if not pd.isna(row.duration) else 0,
            'waveform_halfwidth': row.halfwidth if not pd.isna(row.halfwidth) else 0,
            'PT_ratio': row.PT_ratio if not pd.isna(row.PT_ratio) else 0,
            'repolarization_slope': row.repolarization_slope if not pd.isna(row.repolarization_slope) else 0,
            'recovery_slope': row.recovery_slope if not pd.isna(row.recovery_slope) else 0,
            'amplitude': row.amplitude if not pd.isna(row.amplitude) else 0,
            'spread': row.spread if not pd.isna(row.spread) else 0,
            'velocity_above': row.velocity_above if not pd.isna(row.velocity_above) else 0,
            'velocity_below': row.velocity_below if not pd.isna(row.velocity_below) else 0,
            'local_index': local_index,
            'id': unit_id
        }

        id_json_dict['unit_ids'].append(unit_id)
        units.append(unit_dict)
        local_index += 1
        unit_id += 1

    return units