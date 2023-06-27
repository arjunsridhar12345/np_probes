import np_session
import pathlib
import re
from np_probes.utils import get_probe_metrics_path
import json
import itertools
from np_probes.align_barcode_timestamps import get_align_timestamps_output_dictionary
from typing import Union

def create_lfp_json(session: np_session.Session) -> Union[dict, None]:
    if len(list(session.npexp_path.glob('*.h5'))) == 0:
        return None
    
    #get_align_timestamps_output_dictionary(session)

    lfp_dict:dict = {
        'lfp_subsampling':
            {'temporal_subsampling_factor': 2}, 
        'probes': []}
    probe_metrics_path = get_probe_metrics_path(session)

    if not pathlib.Path(session.npexp_path, 'SDK_outputs').exists():
        pathlib.Path(session.npexp_path, 'SDK_outputs').mkdir()
    
    for probe in probe_metrics_path:
        spike_path = probe_metrics_path[probe].parent
        lfp_path = spike_path.parent
        lfp_path = list(session.npexp_path.glob('*/*/*/*/continuous/*{}-AP'.format(probe)))[0].parent

        probe_dict = {
            "name": 'probe{}'.format(probe),
            "lfp_sampling_rate": 2500.0,
            "lfp_input_file_path": list(lfp_path.glob('*{}-LFP/continuous.dat'.format(probe)))[0].as_posix()[1:],
            "lfp_timestamps_input_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'lfp_times_{}_aligned.npy'.format(probe)).as_posix()[1:],
            "lfp_data_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'probe{}_lfp.dat'.format(probe)).as_posix()[1:],
            "lfp_timestamps_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'probe{}_lfp_timestamps.npy'.format(probe)).as_posix()[1:],
            "lfp_channel_info_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'probe{}_lfp_channels.npy'.format(probe)).as_posix()[1:],
            "surface_channel": 384.0,
            "reference_channels": [
                191
            ] 
        }

        lfp_dict['probes'].append(probe_dict)

    with open(pathlib.Path(session.npexp_path, 'SDK_outputs', 'lfp_subsampling_input.json'), 'w') as f:
        json.dump(lfp_dict, f, indent=2)

    return lfp_dict

if __name__ == '__main__':
    sessions = list(itertools.chain(*(np_session.sessions(root=dir) for dir in np_session.DRPilotSession.storage_dirs)))
    for session in sessions:
       print(pathlib.Path(session.npexp_path, session.id))