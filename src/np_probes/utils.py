import re
import np_session
import pathlib
import np_logging
import pynwb
import uuid
from typing import Union, Optional
import tempfile

logger = np_logging.getLogger(__name__)


def init_nwb(
    session: np_session.Session,
    description: str = 'Data and metadata for a Neuropixels ecephys session',
) -> pynwb.NWBFile:
    """
    Init `NWBFile` with minimum required arguments from an
    `np_session.Session` instance.
    """
    return pynwb.NWBFile(
        session_description=description,
        identifier=str(uuid.uuid4()),  # globally unique for this nwb - not human-readable
        session_start_time=session.start,
    )
    

def load_nwb(
    nwb_path: Union[str, pathlib.Path],
    ) -> pynwb.NWBFile:
    """Load `pynb.NWBFile` instance from path."""
    logger.info(f'Loading .nwb file at {nwb_path}')
    with pynwb.NWBHDF5IO(nwb_path, mode='r') as f:
        return f.read()


def save_nwb(
    nwb_file: pynwb.NWBFile,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> Union[str, pathlib.Path]:
    """
    Write `pynb.NWBFile` instance to disk.
    
    Temp dir is used if `output_path` isn't provided.
    """
    if output_path is None:
        output_path = pathlib.Path(tempfile.mkdtemp()) / f'{nwb_file.session_id}.nwb'
    
    nwb_file.set_modified()
    # not clear if this is necessary, but suggested by docs:
    # https://pynwb.readthedocs.io/en/stable/_modules/pynwb.html

    logger.info(f'Writing .nwb file `{nwb_file.session_id!r}` to {output_path}')
    with pynwb.NWBHDF5IO(output_path, mode='w') as f:
        f.write(nwb_file, cache_spec=True)
    logger.debug(f'Writing complete for nwb file `{nwb_file.session_id!r}`')
    return output_path

def get_probes_from_metrics(metrics_path):
    if not metrics_path:
        return {}

    def letter(x):
        return re.findall('(?<=.Probe)[A-F]', str(x))

    probe_letters = [_[-1] for _ in map(letter, metrics_path) if _]

    if probe_letters:
        return dict(zip(probe_letters, metrics_path))
    return {}

def get_probe_metrics_path(session: np_session.Session) -> dict[str, pathlib.Path]:
    if len(list(session.metrics_csv)) == 0:
        metrics_path = list(session.npexp_path.rglob('metrics_test.csv'))
    else:
        metrics_path = list(session.metrics_csv)
    
    if len(session.probe_letter_to_metrics_csv_path) == 0:
        probe_metrics_path = get_probes_from_metrics(metrics_path)
    else:
        probe_metrics_path = session.probe_letter_to_metrics_csv_path
    
    return probe_metrics_path