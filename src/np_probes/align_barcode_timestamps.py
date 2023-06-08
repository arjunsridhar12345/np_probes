from allensdk.brain_observatory.ecephys.align_timestamps.__main__ import align_timestamps
import np_session
import pathlib
import re
from np_probes.utils import get_probe_metrics_path

def get_align_timestamps_input_dictionary(session:np_session.Session) -> dict:
    probe_metrics_path = get_probe_metrics_path(session)

    align_timestamps_input_dictionary: dict = {'probes': []}
    if not pathlib.Path(session.npexp_path, 'SDK_outputs').exists():
        pathlib.Path(session.npexp_path, 'SDK_outputs').mkdir()

    for probe in probe_metrics_path:
        spike_path = probe_metrics_path[probe].parent
        event_path = spike_path.parent.parent / 'events'
        lfp_path = spike_path.parent
        probe_dict = {
            "name": 'probe{}'.format(probe),
            "sampling_rate": 30000.0,
            "lfp_sampling_rate": 2500.0,
            "barcode_channel_states_path": list(event_path.glob('*{}-AP/*/states.npy'.format(probe)))[0].as_posix(),
            "barcode_timestamps_path": list(event_path.glob('*{}-AP/*/sample_numbers.npy'.format(probe)))[0].as_posix(),
            "mappable_timestamp_files": [
                {
                    "name": "spike_timestamps",
                    "input_path": pathlib.Path(spike_path, 'spike_times.npy').as_posix(),
                    "output_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'spike_times_{}_aligned.npy'.format(probe)).as_posix()
                },
                {
                    "name": "lfp_timestamps",
                    "input_path": list(lfp_path.glob('*{}-LFP/timestamps.npy'.format(probe)))[0].as_posix(),
                    "output_path": pathlib.Path(session.npexp_path, 'SDK_outputs', 'lfp_times_{}_aligned.npy'.format(probe)).as_posix()
                }
            ],
            'start_index': 0
        }

        align_timestamps_input_dictionary['probes'].append(probe_dict)
    
    sync_file = list(session.npexp_path.glob('*.h5'))[0].as_posix()
    align_timestamps_input_dictionary['sync_h5_path'] = sync_file

    return align_timestamps_input_dictionary

def get_align_timestamps_output_dictionary(session:np_session.Session) -> dict:
    align_timestamps_input_dictionary = get_align_timestamps_input_dictionary(session)
    return align_timestamps(align_timestamps_input_dictionary)

if __name__ == '__main__':
    session = np_session.Session('DRpilot_626791_20220817')
    input_dictionary = get_align_timestamps_input_dictionary(session)
    output_dictionary = align_timestamps(input_dictionary)