import np_session
import pathlib
import json
import numpy as np
from np_probes.probe_channel_units import get_channels_info_for_probe
from np_probes.probe_channel_units import get_units_info_for_probe
from np_probes.utils import get_probe_metrics_path
from np_probes.align_barcode_timestamps import get_align_timestamps_output_dictionary
from allensdk.brain_observatory.ecephys.probes import Probes
from allensdk.brain_observatory.ecephys._probe import Probe
from allensdk.brain_observatory.ecephys._lfp import LFP
import pynwb
import uuid
from typing import Union, Optional
from np_probes.utils import init_nwb, load_nwb, save_nwb
import datetime
from np_probes.lfp_subsampling_json import create_lfp_json
from allensdk.brain_observatory.ecephys.nwb_util import add_ragged_data_to_dynamic_table

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

    channels = get_channels_info_for_probe(current_probe, probe_id, session=session, id_json_dict=id_json_dict)
    probe_dict['channels'] = channels
    units = get_units_info_for_probe(current_probe, probe_metrics_path, session, channels, id_json_dict)
    probe_dict['units'] = units
    
    with open(id_json_path, 'w') as f:
        json.dump(id_json_dict, f, indent=2)
    
    return probe_dict

def generate_probes_dictionary(session:np_session.Session) -> tuple[dict, dict]:
    align_timestamps_output_dictionary = get_align_timestamps_output_dictionary(session)
    align_timestamps_probe_outputs = align_timestamps_output_dictionary['probe_outputs']

    probes_dictionary: dict[str, list] = {'probes': []}
    probe_metrics_path = get_probe_metrics_path(session)

    for probe in probe_metrics_path:
        current_probe = 'probe' + probe
        probe_dictionary = generate_probe_dictionary(session, current_probe, probe_metrics_path, align_timestamps_probe_outputs)
        probes_dictionary['probes'].append(probe_dictionary)
    
    return probes_dictionary, align_timestamps_output_dictionary

def add_lfp_to_object(session:np_session.Session, probes_object:Probes, align_timestamps_probe_outputs:dict) -> Probes:
    for probe_object in probes_object.probes:
        current_probe = probe_object.name
        probe_information = [probe_info for probe_info in align_timestamps_probe_outputs if probe_info['name'] == current_probe][0]
        probe_meta = {
            'lfp': {
                'input_data_path': pathlib.Path(session.npexp_path, 'SDK_outputs', '{}_lfp.dat'.format(current_probe)).as_posix(),
                'input_timestamps_path': pathlib.Path(session.npexp_path, 'SDK_outputs', '{}_lfp_timestamps.npy'.format(current_probe)).as_posix(),
                'input_channels_path': pathlib.Path(session.npexp_path, 'SDK_outputs', '{}_lfp_channels.npy'.format(current_probe)),
                'output_path': pathlib.Path(session.npexp_path, 'SDK_outputs', '{}_{}_{}_lfp.nwb'.format(str(session.id), current_probe, '061123'))
            },
            'lfp_sampling_rate': probe_information['global_probe_lfp_sampling_rate'][0],
            'temporal_subsampling_factor': 2.0
        }
        probe_object._lfp = LFP.from_json(probe_meta)
    
    return probes_object

def add_lfp_to_nwb(probe: Probe, session_id: str, session_start_time) -> pynwb.NWBFile:
    nwbfile = pynwb.NWBFile(
        session_description='LFP data and associated info for one probe',
        identifier=f"{probe._id}",
        session_id=f"{session_id}",
        session_start_time=session_start_time,
        institution="Allen Institute for Brain Science"
    )

    nwbfile = probe._add_probe_to_nwb(
        nwbfile=nwbfile,
        add_only_lfp_channels=True
    )
    lfp_nwb = pynwb.ecephys.LFP(name=f"{probe._name}_lfp")

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=np.arange(len(nwbfile.electrodes)).tolist(),
        name='electrodes',
        description=f"lfp channels on probe {probe._name}"
    )

    electrial_series = lfp_nwb.create_electrical_series(
        name=f"session{session_id}_{probe._name}_lfp_data",
        data=probe._lfp.data,
        timestamps=probe._lfp.timestamps,
        electrodes=electrode_table_region
    )

    ecephys_module = nwbfile.create_processing_module(
        name="lfp_data", description="processed lfp data"
    )
    ecephys_module.add(lfp_nwb)
    """
    nwbfile.add_acquisition(lfp_nwb.create_electrical_series(
        name=f"session{session_id}_probe_{probe._name}_lfp_data",
        data=probe._lfp.data,
        timestamps=probe._lfp.timestamps,
        electrodes=electrode_table_region
    ))
    nwbfile.add_acquisition(lfp_nwb)
    """

    if probe._current_source_density is not None:
        nwbfile = probe._add_csd_to_nwb(nwbfile=nwbfile)

    return nwbfile

def add_to_nwb(session_folder: Union[str, pathlib.Path], nwb_file: Optional[Union[str, pathlib.Path, pynwb.NWBFile]]=None,
                output_file: Optional[Union[str, pathlib.Path]] = None,) -> tuple[pynwb.NWBFile, dict[str, pynwb.NWBFile]]:
    session_folder = pathlib.Path(session_folder)
    session = np_session.Session(session_folder)

    if not isinstance(nwb_file, pynwb.NWBFile) and nwb_file is not None:
        nwb_file = load_nwb(nwb_file) 
    
    if nwb_file is None:
        nwb_file = init_nwb(session)


    probes_dictionary, align_timestamps_output_dictionary = generate_probes_dictionary(session)
    create_lfp_json(session)
    probes_object = Probes.from_json(probes_dictionary['probes'])
    for probe_object in probes_object:
        nwb_file = probe_object.to_nwb(nwb_file)[0]

    units = probes_object.get_units_table()
    electrodes = nwb_file.electrodes.to_dataframe()
    units_channels = units.merge(electrodes[['location', 'x', 'y', 'z']], left_on='peak_channel_id', right_index=True)
    nwb_file.units = pynwb.misc.Units.from_dataframe(
            units_channels,
            name='units')
    
    add_ragged_data_to_dynamic_table(
            table=nwb_file.units,
            data=probes_object.spike_times,
            column_name="spike_times",
            column_description="times (s) of detected spiking events",
    )

    add_ragged_data_to_dynamic_table(
            table=nwb_file.units,
            data=probes_object.spike_amplitudes,
            column_name="spike_amplitudes",
            column_description="amplitude (s) of detected spiking events"
    )

    add_ragged_data_to_dynamic_table(
            table=nwb_file.units,
            data=probes_object.mean_waveforms,
            column_name="waveform_mean",
            column_description="mean waveforms on peak channels (over "
                               "samples)",
    )

    #nwb_file = probes_object.to_nwb(nwb_file)[0]

    probes_object_with_lfp = add_lfp_to_object(session, probes_object, align_timestamps_output_dictionary['probe_outputs'])
    lfp_nwbs:dict = {}

    for probe in probes_object_with_lfp:
        lfp_nwbs[probe.name] = add_lfp_to_nwb(probe, str(session.id), session.start)

    if output_file is not None:
        save_nwb(nwb_file, output_file)

    return nwb_file, lfp_nwbs

if __name__ == '__main__':
    session = np_session.Session('DRpilot_626791_20220817')
    nwb_file = pynwb.NWBFile(session_description="DR Pilot experiment with probe data", identifier=str(uuid.uuid4()),
                             session_start_time=session.start)
    
    """
    nwb_file, probe_lfp_map = add_to_nwb(str(session.id), nwb_file)
    
    with pynwb.NWBHDF5IO('DRpilot_626791_20220817_probes_061623.nwb', mode='w') as io:
        io.write(nwb_file)
    
    for probe_lfp in probe_lfp_map:
        with pynwb.NWBHDF5IO('DRpilot_626791_20220817_{}_061623.nwb'.format(probe_lfp), mode='w') as io:
            io.write(probe_lfp_map[probe_lfp])
    """
    with pynwb.NWBHDF5IO('DRpilot_626791_20220817_probeA_061623.nwb', mode='r') as io:
        nwb_file = io.read()
        print()
    
    
    
    