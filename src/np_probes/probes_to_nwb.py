import np_session
import pathlib
import json
from np_probes.probe_channel_units import get_channels_info_for_probe
from np_probes.probe_channel_units import get_units_info_for_probe
from np_probes.utils import get_probe_metrics_path
from np_probes.align_barcode_timestamps import get_align_timestamps_output_dictionary
from allensdk.brain_observatory.ecephys.probes import Probes
import pynwb
import uuid
from typing import Union, Optional
from np_probes.utils import init_nwb, load_nwb, save_nwb

def generate_probe_dictionary(session:np_session.Session, current_probe:str, probe_metrics_path:dict[str, pathlib.Path],
                              align_timestamps_probe_outputs:dict) -> dict:
    ap_path = probe_metrics_path[current_probe[-1]].parent
    id_json_dict = None
    
    # unique ids for probe, channel, and units
    id_json_path = pathlib.Path('//allen/programs/mindscope/workgroups/dynamicrouting', 'dynamic_routing_unique_ids.json')
    if id_json_path.exists():
        with open(id_json_path, 'r') as f:
            id_json_dict = json.load(f)
            probe_id = id_json_dict['probe_ids'][-1] + 1
    else:
        probe_id = 0

    if id_json_dict is None:
        id_json_dict = {'probe_ids': [], 'channel_ids': [], 'unit_ids': []}

    probe_information = [probe_info for probe_info in align_timestamps_probe_outputs if probe_info['name'] == current_probe][0]
    probe_dict = {
        'name': current_probe,
        'sampling_rate': probe_information['global_probe_sampling_rate'][0],
        'temporal_subsampling_factor': 2,
        'lfp_sampling_rate': probe_information['global_probe_lfp_sampling_rate'][0],

        'csd_path': None,

        'lfp': None,

        'id': probe_id,

        'inverse_whitening_matrix_path': pathlib.Path(ap_path, 'whitening_mat_inv.npy').as_posix(),
        'mean_waveforms_path': pathlib.Path(ap_path, 'mean_waveforms.npy').as_posix(),
        'spike_amplitudes_path': pathlib.Path(ap_path, 'amplitudes.npy').as_posix(),
        'spike_clusters_file': pathlib.Path(ap_path, 'spike_clusters.npy').as_posix(),
        'spike_templates_path': pathlib.Path(ap_path, 'spike_templates.npy').as_posix(),
        'templates_path': pathlib.Path(ap_path, 'templates.npy').as_posix(),
        'spike_times_path': pathlib.Path(session.npexp_path, 'SDK_outputs', 'spike_times_{}_aligned.npy'.format(current_probe[-1])).as_posix()
    }

    id_json_dict['probe_ids'].append(probe_id)

    channels = get_channels_info_for_probe(current_probe, probe_id, id_json_dict, session)
    probe_dict['channels'] = channels
    units = get_units_info_for_probe(current_probe, probe_metrics_path, session, channels, id_json_dict)
    probe_dict['units'] = units

    with open(id_json_path, 'w') as f:
        json.dump(id_json_dict, f, indent=2)
    
    return probe_dict

def generate_probes_dictionary(session:np_session.Session) -> dict:
    align_timestamps_output_dictionary = get_align_timestamps_output_dictionary(session)
    align_timestamps_probe_outputs = align_timestamps_output_dictionary['probe_outputs']

    probes_dictionary: dict[str, list] = {'probes': []}
    probe_metrics_path = get_probe_metrics_path(session)

    for probe in probe_metrics_path:
        current_probe = 'probe' + probe
        probe_dictionary = generate_probe_dictionary(session, current_probe, probe_metrics_path, align_timestamps_probe_outputs)
        probes_dictionary['probes'].append(probe_dictionary)
    
    return probes_dictionary

def add_to_nwb(session_folder: Union[str, pathlib.Path], nwb_file: Optional[Union[str, pathlib.Path, pynwb.NWBFile]]=None,
                output_file: Optional[Union[str, pathlib.Path]] = None,) -> pynwb.NWBFile:
    session_folder = pathlib.Path(session_folder)
    session = np_session.Session(session_folder)

    if not isinstance(nwb_file, pynwb.NWBFile) and nwb_file is not None:
        nwb_file = load_nwb(nwb_file) 
    
    if nwb_file is None:
        nwb_file = init_nwb(session)

    probes_dictionary = generate_probes_dictionary(session)
    probes_object = Probes.from_json(probes_dictionary['probes'])

    nwb_file = probes_object.to_nwb(nwb_file)[0]

    if output_file is not None:
        save_nwb(nwb_file, output_file)

    return nwb_file

if __name__ == '__main__':
    session = np_session.Session('DRpilot_626791_20220817')
    nwb_file = pynwb.NWBFile(session_description="DR Pilot experiment with probe data", identifier=str(uuid.uuid4()),
                             session_start_time=session.start)
    
    #nwb_file = add_to_nwb(session, nwb_file)

    with pynwb.NWBHDF5IO('DRpilot_626791_20220817_probes.nwb', mode='r') as io:
        nwb_file = io.read()
        #probes = Probes.from_nwb(nwb_file)
        print()