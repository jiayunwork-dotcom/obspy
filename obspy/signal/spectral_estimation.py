#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Filename: spectral_estimation.py
#  Purpose: Various Routines Related to Spectral Estimation
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 Tobias Megies
# -----------------------------------------------------------------------------
"""
Various Routines Related to Spectral Estimation

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import bisect
import glob
import math
from pathlib import Path
import warnings

import numpy as np
from matplotlib import mlab
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patheffects import withStroke

from obspy import Stream, Trace, UTCDateTime, __version__
from obspy.core import Stats
from obspy.imaging.scripts.scan import compress_start_end
from obspy.core.inventory import Inventory
from obspy.core.util import AttribDict
from obspy.core.util.base import MATPLOTLIB_VERSION
from obspy.core.util.obspy_types import ObsPyException
from obspy.imaging.cm import obspy_sequential
from obspy.imaging.util import _set_xaxis_obspy_dates
from obspy.io.xseed import Parser
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import prev_pow_2
from obspy.signal.invsim import paz_to_freq_resp, evalresp


dtiny = np.finfo(0.0).tiny

NOISE_MODEL_FILE = Path(__file__).parent / "data" / "noise_models.npz"


# Noise models for special_handling="infrasound"
NOISE_MODEL_FILE_INF = Path(__file__).parent / "data" / "idc_noise_models.npz"


class PPSDRecalibrationError(ObsPyException):
    """
    Raised when recalibration of archived PPSD spectra with updated response
    information fails (see :meth:`PPSD.recalibrate`).
    """
    pass


earthquake_models = {
    (1.5, 10): [[7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                 1.1300000e+01, 2.2600000e+01],
                [1.1940722e-06, 5.9652811e-06, 4.2022150e-05, 2.1857010e-04,
                 6.4011669e-04, 1.0630361e-03]],
    (1.5, 100): [[1.4140000e+00, 2.8280000e+00, 5.6600000e+00, 1.1300000e+01,
                  2.2600000e+01],
                 [2.0636508e-07, 1.0591860e-06, 4.4487964e-06, 1.1678490e-05,
                  1.2427714e-05]],
    (2.5, 10): [[3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [3.1301161e-06, 1.9775227e-05, 1.5482427e-04, 8.3811450e-04,
                 3.7537242e-03, 7.3529155e-03, 6.5422494e-03]],
    (2.5, 100): [[7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                  1.1300000e+01, 2.2600000e+01],
                 [7.4170515e-07, 4.9107516e-06, 2.5050832e-05, 8.4981163e-05,
                  9.5891456e-05, 6.5125171e-05]],
    (3.5, 10): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                 2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [1.4096634e-05, 8.5860094e-05, 5.4403050e-04, 3.4576905e-03,
                 1.3476040e-02, 3.4295149e-02, 4.7010730e-02, 2.3188069e-02]],
    (3.5, 100): [[3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                  5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [2.0677442e-06, 1.4237587e-05, 8.5915636e-05, 3.8997357e-04,
                  1.0533844e-03, 8.7243737e-04, 2.8224063e-04]],
    (4.5, 10): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                 2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                [4.0994260e-05, 2.9194046e-04, 1.9989323e-03, 1.0610332e-02,
                 3.1188683e-02, 6.5872809e-02, 7.7301339e-02, 2.7431279e-02]],
    (4.5, 100): [[1.7670000e-01, 3.5350000e-01, 7.0700000e-01, 1.4140000e+00,
                  2.8280000e+00, 5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [8.3385376e-06, 4.8130301e-05, 3.1461832e-04, 1.1054180e-03,
                  3.2015584e-03, 5.9504166e-03, 3.4106672e-03, 1.0389085e-03]],
    (5.5, 10): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
                 7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                 1.1300000e+01, 2.2600000e+01],
                [5.1597812e-05, 2.4350754e-04, 1.5861857e-03, 9.1333683e-03,
                 5.4321168e-02, 3.0210031e-01, 4.4505525e-01, 3.3287588e-01,
                 3.0775399e-01, 1.2501407e-01]],
    (5.5, 100): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
                  7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
                  1.1300000e+01, 2.2600000e+01],
                 [7.6042044e-06, 5.0568931e-05, 2.6926792e-04, 9.7744507e-04,
                  3.4530700e-03, 9.2832164e-03, 1.5054122e-02, 1.3421025e-02,
                  8.0524871e-03, 2.0691071e-03]],
    (6.5, 10): [[8.8375000e-02, 1.7670000e-01, 3.5350000e-01, 7.0700000e-01,
                 1.4140000e+00, 2.8280000e+00, 5.6600000e+00, 1.1300000e+01,
                 2.2600000e+01],
                [1.7468292e-02, 6.5551361e-02, 1.8302926e-01, 3.9065640e-01,
                 7.1420714e-01, 1.0447672e+00, 1.2770160e+00, 1.0372500e+00,
                 5.0465056e-01]],
    (6.5, 100): [[2.2097000e-02, 4.4187500e-02, 8.8375000e-02, 1.7670000e-01,
                  3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                  5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
                 [1.4561822e-05, 1.4224456e-04, 8.7870935e-04, 2.4575511e-03,
                  8.0615599e-03, 2.3743599e-02, 4.8707533e-02, 7.0969056e-02,
                  7.6622487e-02, 2.6998756e-02, 4.6235290e-03]],
    (7, 10): [[4.4187500e-02, 8.8375000e-02, 1.7670000e-01, 3.5350000e-01,
               7.0700000e-01, 1.4140000e+00, 2.8280000e+00, 5.6600000e+00,
               1.1300000e+01, 2.2600000e+01, 4.5200000e+01, 7.0700000e+01],
              [9.1004029e-03, 4.7514014e-02, 1.5787725e-01, 3.1426661e-01,
               6.3241827e-01, 1.0667690e+00, 1.2675411e+00, 1.1326387e+00,
               8.3962478e-01, 4.1229082e-01, 1.6311586e-01, 5.6992659e-02]],
    (7, 100): [[2.2097000e-02, 4.4187500e-02, 8.8375000e-02, 1.7670000e-01,
                3.5350000e-01, 7.0700000e-01, 1.4140000e+00, 2.8280000e+00,
                5.6600000e+00, 1.1300000e+01, 2.2600000e+01],
               [5.8736062e-04, 3.2124167e-03, 1.5314560e-02, 3.3624616e-02,
                6.4082408e-02, 1.3541913e-01, 1.8690009e-01, 1.8862548e-01,
                1.1841223e-01, 4.8641263e-02, 1.3723779e-02]],
    (6, 3000): [[5.0000000e-03, 4.5000000e-02, 3.0000000e-01, 1.3000000e+00,
                 6.0000000e+00],
                [9.8100000e-10, 7.8480000e-07, 4.9050000e-06, 4.9050000e-06,
                 7.8480000e-07]],
    (7, 3000): [[1.5000000e-03, 2.1000000e-02, 4.5000000e-02, 9.0000000e-02,
                 8.0000000e-01, 6.0000000e+00],
                [1.4715000e-09, 7.8480000e-06, 2.9430000e-05, 3.9240000e-05,
                 1.4715000e-05, 9.8100000e-07]],
    (8, 3000): [[1.5000000e-03, 1.0000000e-02, 4.5000000e-02, 9.0000000e-02,
                 3.0000000e+00, 6.0000000e+00],
                [9.8100000e-09, 3.9240000e-06, 1.1772000e-04, 1.9620000e-04,
                 4.9050000e-06, 1.9620000e-06]]}


def fft_taper(data):
    """
    Cosine taper, 10 percent at each end (like done by [McNamara2004]_).

    .. warning::
        Inplace operation, so data should be float.
    """
    data *= cosine_taper(len(data), 0.2)
    return data


def welch_taper(data):
    """
    Applies a welch window to data. See
    :func:`~obspy.signal.spectral_estimation.welch_window`.

    .. warning::
        Inplace operation, so data should be float.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to apply the taper to. Inplace operation, but also
        returns data for convenience.
    :returns: Tapered data.
    """
    data *= welch_window(len(data))
    return data


def welch_window(n):
    """
    Return a welch window for data of length n.

    Routine is checked against PITSA for both even and odd values, but not for
    strange values like n<5.

    .. note::
        See e.g.:
        http://www.cescg.org/CESCG99/TTheussl/node7.html

    :type n: int
    :param n: Length of window function.
    :rtype: :class:`~numpy.ndarray`
    :returns: Window function for tapering data.
    """
    n = math.ceil(n / 2.0)
    taper_left = np.arange(n, dtype=np.float64)
    taper_left = 1 - np.power(taper_left / n, 2)
    # first/last sample is zero by definition
    if n % 2 == 0:
        # even number of samples: two ones in the middle, perfectly symmetric
        taper_right = taper_left
    else:
        # odd number of samples: still two ones in the middle, however, not
        # perfectly symmetric anymore. right side is shorter by one sample
        nn = n - 1
        taper_right = np.arange(nn, dtype=np.float64)
        taper_right = 1 - np.power(taper_right / nn, 2)
    taper_left = taper_left[::-1]
    # first/last sample is zero by definition
    taper_left[0] = 0.0
    taper_right[-1] = 0.0
    taper = np.concatenate((taper_left, taper_right))
    return taper


class PPSD(object):
    """
    Class to compile probabilistic power spectral densities for one combination
    of network/station/location/channel/sampling_rate.

    Calculations are based on the routine used by [McNamara2004]_.
    For information on New High/Low Noise Model see [Peterson1993]_.

    .. rubric:: Basic Usage

    >>> from obspy import read
    >>> from obspy.signal import PPSD

    >>> st = read()
    >>> tr = st.select(channel="EHZ")[0]
    >>> paz = {'gain': 60077000.0,
    ...        'poles': [-0.037004+0.037016j, -0.037004-0.037016j,
    ...                  -251.33+0j, -131.04-467.29j, -131.04+467.29j],
    ...        'sensitivity': 2516778400.0,
    ...        'zeros': [0j, 0j]}

    >>> ppsd = PPSD(tr.stats, paz)
    >>> print(ppsd.id)
    BW.RJOB..EHZ
    >>> print(ppsd.times_processed)
    []

    Now we could add data to the probabilistic psd (all processing like
    demeaning, tapering and so on is done internally) and plot it like ...

    >>> ppsd.add(st) # doctest: +SKIP
    >>> print(ppsd.times_processed) # doctest: +SKIP
    [UTCDateTime(...), UTCDateTime(...), ..., UTCDateTime(...)]
    >>> ppsd.plot() # doctest: +SKIP

    ... but the example stream is too short and does not contain enough data.

    .. note::

        For a real world example see the `ObsPy Tutorial`_.

    .. rubric:: Saving and Loading

    The PPSD object supports saving to a numpy npz compressed binary file:

    >>> ppsd.save_npz("myfile.npz") # doctest: +SKIP

    The saved PPSD can then be loaded again using the static method
    :func:`~obspy.signal.spectral_estimation.PPSD.load_npz`, e.g. to plot the
    results again. If additional data is to be processed (note that another
    option is to combine multiple npz files using
    :meth:`~obspy.signal.spectral_estimation.PPSD.add_npz`), metadata must be
    provided again, since they are not stored in the numpy npz file:

    >>> ppsd = PPSD.load_npz("myfile.npz")  # doctest: +SKIP

    .. note::

        When using metadata from an
        :class:`~obspy.core.inventory.inventory.Inventory`,
        a :class:`~obspy.io.xseed.parser.Parser` instance or from a RESP file,
        information on metadata will be correctly picked for the respective
        starttime of the data trace. This means that instrument changes are
        correctly taken into account during response removal.
        This is obviously not the case for a static PAZ dictionary!

    .. rubric:: Archiving Spectra for Later Recalibration

    Optionally, the PPSD can archive for every processed time window the
    spectral information needed to recompute the window's (smoothed and
    binned) psd values with updated instrument response information at a
    later time, e.g. when a corrected StationXML file becomes available
    after the raw waveform data has been cleaned up. This is switched on
    with ``archive_spectra=True`` together with a non-empty
    ``metadata_version`` label that identifies the version of the metadata
    in use (currently only supported for regular seismic channels with
    metadata given as an
    :class:`~obspy.core.inventory.inventory.Inventory`):

    >>> ppsd = PPSD(tr.stats, inventory, archive_spectra=True,
    ...             metadata_version="v1")  # doctest: +SKIP

    For every processed time window, the one-sided, raw (i.e. **not**
    instrument-corrected) power spectrum is stored as float64 values
    together with a calibration record (metadata version label, channel SEED
    ID, the actual time used for the response lookup and the start/end times
    of the channel epoch that was used). No time-domain data samples are
    stored. The archive grows with the number of retained time windows and
    the number of spectral points (``nfft // 2`` float64 values, i.e. 8
    bytes each, per window), so the npz files get considerably larger than
    for a plain PPSD (which only stores the much smaller smoothed and
    binned psd arrays as float32). The float64 representation was chosen
    because it makes spectra recomputed from the archive bit-exact
    identical to a fresh computation on the original waveforms (absolute
    difference of 0.0 dB, far below the required 0.0001 dB); a float32
    archive would halve the archive size at the cost of a small recompute
    error (on the order of 1e-6 dB).

    Windows can then be recalibrated with updated metadata (e.g. a corrected
    StationXML file read into a new
    :class:`~obspy.core.inventory.inventory.Inventory`) using
    :meth:`recalibrate`, without changing window times, data coverage or
    gap statistics, and the histogram is recounted from the updated values.
    The calibration record of every archived window can be queried with
    :meth:`get_archive_info`, both in memory and after reloading a saved
    npz file.

    Regarding npz read-back compatibility: the archive is stored in
    additional arrays inside the same npz file and the ``ppsd_version``
    number is left unchanged. Files written with spectra archiving can
    still be opened with :meth:`load_npz` by ObsPy versions without
    archiving support (the additional arrays are simply ignored and the
    PPSD behaves like a plain one), and plain npz files without archive
    information are loaded as before (with spectra archiving switched off).
    When combining npz files with :meth:`add_npz`, a PPSD with spectra
    archiving enabled only accepts npz files that also contain archive
    information (computed with compatible settings); archive information of
    merged windows is kept in sync with the windows' psd data.

    .. _`ObsPy Tutorial`: https://docs.obspy.org/tutorial/
    """
    NPZ_STORE_KEYS_LIST_TYPES = [
        # things related to processed data
        '_times_data',
        '_times_gaps',
        '_times_processed',
        '_binned_psds']
    NPZ_STORE_KEYS_VERSION_NUMBERS = [
        # version numbers
        'ppsd_version',
        'obspy_version',
        'numpy_version',
        'matplotlib_version']
    NPZ_STORE_KEYS_SIMPLE_TYPES = [
        # things related to Stats passed at __init__
        'id',
        'sampling_rate',
        # kwargs passed during __init__
        'skip_on_gaps',
        'ppsd_length',
        'overlap',
        'special_handling',
        # attributes derived during __init__
        '_len',
        '_nlap',
        '_nfft']
    NPZ_STORE_KEYS_ARRAY_TYPES = [
        # attributes derived during __init__
        '_db_bin_edges',
        '_psd_periods',
        '_period_binning']
    NPZ_STORE_KEYS = (
        NPZ_STORE_KEYS_ARRAY_TYPES +
        NPZ_STORE_KEYS_LIST_TYPES +
        NPZ_STORE_KEYS_SIMPLE_TYPES +
        NPZ_STORE_KEYS_VERSION_NUMBERS)
    # A mapping of values for storing info in the NPZ file. This is needed
    # because some types are not loadable without allowing pickle (#2409).
    NPZ_SIMPLE_TYPE_MAP = {None: ''}
    NPZ_SIMPLE_TYPE_MAP_R = {v: i for i, v in NPZ_SIMPLE_TYPE_MAP.items()}
    # Add current version as a class attribute to avoid hard coding it.
    _CURRENT_VERSION = 3
    # Keys for the optional spectra archive (see __init__ keywords
    # 'archive_spectra' and 'metadata_version' and method 'recalibrate').
    # These are stored in the npz file in addition to the regular keys (and
    # only when spectra archiving is enabled), so that npz files stay
    # readable as plain PPSD results both for older ObsPy versions and when
    # spectra archiving is not used.
    NPZ_ARCHIVE_FORMAT_KEY = '_psd_archive_format'
    NPZ_ARCHIVE_FORMAT_VERSION = 1
    NPZ_STORE_KEYS_ARCHIVE = [
        # raw (not instrument corrected) one-sided spectra, one row per
        # processed time window
        '_psd_archive_spectra',
        # per-window calibration records
        '_psd_archive_versions',
        '_psd_archive_channel_ids',
        '_psd_archive_lookup_times',
        '_psd_archive_epoch_starts',
        '_psd_archive_epoch_ends',
        '_psd_archive_epoch_starts_open',
        '_psd_archive_epoch_ends_open']

    def __init__(self, stats, metadata, skip_on_gaps=False,
                 db_bins=(-200, -50, 1.), ppsd_length=3600.0, overlap=0.5,
                 special_handling=None, period_smoothing_width_octaves=1.0,
                 period_step_octaves=0.125, period_limits=None,
                 archive_spectra=False, metadata_version=None,
                 **kwargs):  # @UnusedVariable
        """
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        The necessary instrument response information can be provided in
        several ways using the `metadata` keyword argument:

        * Providing an :class:`~obspy.core.inventory.inventory.Inventory`
          object (e.g. read from a StationXML file using
          :func:`~obspy.core.inventory.inventory.read_inventory` or fetched
          from a :mod:`FDSN <obspy.clients.fdsn>` webservice).
        * Providing an
          :class:`obspy.io.xseed.parser.Parser`,
          (e.g. containing metadata from a Dataless SEED file).
        * Providing the filename/path to a local RESP file.
        * Providing a dictionary containing poles and zeros information. Be
          aware that this leads to wrong results if the instrument's response
          is changing over the timespans that are added to the PPSD.
          Use with caution!

        :note: When using `special_handling="ringlaser"` the applied processing
               steps are changed. Differentiation of data (converting velocity
               to acceleration data) will be omitted and a flat instrument
               response is assumed, leaving away response removal and only
               dividing by `metadata['sensitivity']` specified in the provided
               `metadata` dictionary (other keys do not have to be present
               then). For scaling factors that are usually multiplied to the
               data remember to use the inverse as `metadata['sensitivity']`.

        :type stats: :class:`~obspy.core.trace.Stats`
        :param stats: Stats of the station/instrument to process
        :type metadata: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed.parser.Parser` or :class:`~pathlib.Path` or
            str or dict
        :param metadata: Response information of instrument. See above notes
            for details.
        :type skip_on_gaps: bool, optional
        :param skip_on_gaps: Determines whether time segments with gaps should
                be skipped entirely. [McNamara2004]_ merge gappy
                traces by filling with zeros. This results in a clearly
                identifiable outlier psd line in the PPSD visualization. Select
                `skip_on_gaps=True` for not filling gaps with zeros which might
                result in some data segments shorter than `ppsd_length` not
                used in the PPSD.
        :type db_bins: tuple(int, int, int) or tuple(float, float, float)
        :param db_bins: Specify the lower and upper boundary and the width of
                the db bins. The bin width might get adjusted to fit  a number
                of equally spaced bins in between the given boundaries.
        :type ppsd_length: float, optional
        :param ppsd_length: Length of data segments passed to psd in seconds.
                In the paper by [McNamara2004]_ a value of 3600 (1 hour) was
                chosen. Longer segments increase the upper limit of analyzed
                periods but decrease the number of analyzed segments.
        :type overlap: float, optional
        :param overlap: Overlap of segments passed to psd. Overlap may take
                values between 0 and 1 and is given as fraction of the length
                of one segment, e.g. `ppsd_length=3600` and `overlap=0.5`
                result in an overlap of 1800s of the segments.
        :type special_handling: str, optional
        :param special_handling: Switches on customized handling for
            data other than seismometer recordings. Can be one of: 'ringlaser'
            (no instrument correction, just division by
            `metadata["sensitivity"]` of provided metadata dictionary),
            'hydrophone' (no differentiation after instrument correction).
        :type period_smoothing_width_octaves: float
        :param period_smoothing_width_octaves: Determines over what
            period/frequency range the psd is smoothed around every central
            period/frequency. Given in fractions of octaves (default of ``1``
            means the psd is averaged over a full octave at each central
            frequency).
        :type period_step_octaves: float
        :param period_step_octaves: Step length on frequency axis in fraction
            of octaves (default of ``0.125`` means one smoothed psd value on
            the frequency axis is measured every 1/8 of an octave).
        :type period_limits: tuple or list[float, float]
        :param period_limits: Set custom lower and upper end of period range
            (e.g. ``(0.01, 100)``). The specified lower end of period range
            will be set as the central period of the first bin (geometric mean
            of left/right edges of smoothing interval). At the upper end of the
            specified period range, no more additional bins will be added after
            the bin whose center frequency exceeds the given upper end for the
            first time.
        :type archive_spectra: bool
        :param archive_spectra: If ``True``, archive for every processed time
            window the spectral information needed to later recompute the
            window's psd values with updated response information (see
            :meth:`recalibrate`). Currently only supported for regular
            seismic channels (i.e. no ``special_handling``) with metadata
            given as an :class:`~obspy.core.inventory.inventory.Inventory`.
            See the class documentation ("Archiving Spectra for Later
            Recalibration") for details on what is stored and on the
            trade-offs regarding npz read-back compatibility, archive size
            and recalibration accuracy.
        :type metadata_version: str
        :param metadata_version: Non-empty version label identifying the
            version of the provided metadata (e.g. of the StationXML file
            the Inventory was read from). Mandatory when
            ``archive_spectra=True``. The label is stored in the calibration
            record of every archived time window, so that it can later be
            told apart which metadata version was used for which window.
        """
        # save things related to args
        self.id = "%(network)s.%(station)s.%(location)s.%(channel)s" % stats
        self.sampling_rate = stats.sampling_rate
        if isinstance(metadata, Path):
            metadata = str(metadata)
        self.metadata = metadata

        # save things related to kwargs
        self.skip_on_gaps = skip_on_gaps
        self.db_bins = db_bins

        self.ppsd_length = ppsd_length
        self.overlap = overlap
        self.special_handling = special_handling and special_handling.lower()
        if self.special_handling == 'ringlaser':
            if not isinstance(self.metadata, dict):
                msg = ("When using `special_handling='ringlaser'`, `metadata` "
                       "must be a plain dictionary with key 'sensitivity' "
                       "stating the overall sensitivity`.")
                raise TypeError(msg)
        elif self.special_handling == 'hydrophone':
            pass
        # Add special handling option for infrasound
        elif self.special_handling == 'infrasound':
            pass
        elif self.special_handling is not None:
            msg = "Unsupported value for 'special_handling' parameter: %s"
            msg = msg % self.special_handling
            raise ValueError(msg)

        # set up optional spectra archiving for later recalibration with
        # updated response information
        self.archive_spectra = bool(archive_spectra)
        self.metadata_version = None
        if self.archive_spectra:
            if not isinstance(self.metadata, Inventory):
                msg = ("Spectra archiving ('archive_spectra=True') is "
                       "currently only supported with metadata provided as "
                       "an Inventory object, got: %s" % type(self.metadata))
                raise TypeError(msg)
            if self.special_handling is not None:
                msg = ("Spectra archiving ('archive_spectra=True') is "
                       "currently only supported for regular seismic "
                       "channels, i.e. without 'special_handling' (got: "
                       "'%s')." % self.special_handling)
                raise ValueError(msg)
            self._validate_metadata_version(metadata_version)
            self.metadata_version = metadata_version

        # save version numbers
        self.ppsd_version = self._CURRENT_VERSION
        self.obspy_version = __version__
        self.matplotlib_version = ".".join(map(str, MATPLOTLIB_VERSION))
        self.numpy_version = np.__version__

        # calculate derived attributes
        # nfft is determined mimicking the fft setup in McNamara&Buland
        # paper:
        # (they take 13 segments overlapping 75% and truncate to next lower
        #  power of 2)
        #  - take number of points of whole ppsd segment (default 1 hour)
        nfft = self.ppsd_length * self.sampling_rate
        #  - make 13 single segments overlapping by 75%
        #    (1 full segment length + 25% * 12 full segment lengths)
        nfft = nfft / 4.0
        #  - go to next smaller power of 2 for nfft
        self._nfft = prev_pow_2(nfft)
        #  - use 75% overlap
        #    (we end up with a little more than 13 segments..)
        self._nlap = int(0.75 * self.nfft)
        # Trace length for one psd segment.
        self._len = int(self.sampling_rate * self.ppsd_length)
        # make an initial dummy psd and to get the array of periods
        _, freq = mlab.psd(np.ones(self.len), self.nfft,
                           self.sampling_rate, noverlap=self.nlap)
        # leave out first entry (offset)
        freq = freq[1:]
        self._psd_periods = 1.0 / freq[::-1]

        if period_limits is None:
            period_limits = (self.psd_periods[0], self.psd_periods[-1])
        self._setup_period_binning(
            period_smoothing_width_octaves, period_step_octaves, period_limits)
        # setup db binning
        # Set up the binning for the db scale.
        num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
        self._db_bin_edges = np.linspace(db_bins[0], db_bins[1],
                                         num_bins + 1, endpoint=True)

        # lists related to persistent processed data
        self._times_processed = []
        self._times_data = []
        self._times_gaps = []
        self._binned_psds = []
        # per-window spectra archive for later recalibration (only filled
        # when 'archive_spectra' is enabled, then always aligned with
        # '_times_processed' and '_binned_psds')
        self._psd_archive = []

        # internal attributes for stacks on processed data
        self._current_hist_stack = None
        self._current_hist_stack_cumulative = None
        self._current_times_used = []
        self._current_times_all_details = []

    @staticmethod
    def _validate_metadata_version(metadata_version):
        """
        Check that the given metadata version label is a non-empty string,
        raising a ValueError otherwise.
        """
        if not isinstance(metadata_version, str) or \
                not metadata_version.strip():
            msg = ("'metadata_version' must be a non-empty string "
                   "identifying the version of the metadata in use (e.g. of "
                   "the StationXML file the Inventory was read from), got: "
                   "%r" % (metadata_version,))
            raise ValueError(msg)

    @property
    def network(self):
        return self.id.split(".")[0]

    @property
    def station(self):
        return self.id.split(".")[1]

    @property
    def location(self):
        return self.id.split(".")[2]

    @property
    def channel(self):
        return self.id.split(".")[3]

    @property
    def delta(self):
        return 1.0 / self.sampling_rate

    @property
    def len(self):
        """
        Trace length for one psd segment.
        """
        return self._len

    @property
    def step(self):
        """
        Time step between start times of adjacent psd segments in seconds
        (assuming gap-less data).
        """
        return self.ppsd_length * (1.0 - self.overlap)

    @property
    def nfft(self):
        return self._nfft

    @property
    def nlap(self):
        return self._nlap

    @property
    def merge_method(self):
        if self.skip_on_gaps:
            return -1
        else:
            return 0

    @property
    def db_bin_edges(self):
        return self._db_bin_edges

    @property
    def db_bin_centers(self):
        return (self.db_bin_edges[:-1] + self.db_bin_edges[1:]) / 2.0

    @property
    def psd_frequencies(self):
        return 1.0 / self.psd_periods[::-1]

    @property
    def psd_periods(self):
        return self._psd_periods

    @property
    def period_bin_centers(self):
        """
        Return centers of period bins (geometric mean of left and right edge of
        period smoothing ranges).
        """
        return self._period_binning[2, :]

    @property
    def period_xedges(self):
        """
        Returns edges of period histogram bins (one element longer than number
        of bins). These are the edges of the plotted histogram/pcolormesh, but
        not the edges used for smoothing along the period axis of the psd
        (before binning).
        """
        return np.concatenate([self._period_binning[1, 0:1],
                               self._period_binning[3, :]])

    @property
    def period_bin_left_edges(self):
        """
        Returns left edges of period bins (same length as number of bins).
        These are the edges used for smoothing along the period axis of the psd
        (before binning), not the edges of the histogram/pcolormesh in the
        plot.
        """
        return self._period_binning[0, :]

    @property
    def period_bin_right_edges(self):
        """
        Returns right edges of period bins (same length as number of bins).
        These are the edges used for smoothing along the period axis of the psd
        (before binning), not the edges of the histogram/pcolormesh in the
        plot.
        """
        return self._period_binning[4, :]

    @property
    def psd_values(self):
        """
        Returns all individual smoothed psd arrays as a list. The corresponding
        times can be accessed as :attr:`PPSD.times_processed`, the
        corresponding central periods in seconds (central frequencies in Hertz)
        can be accessed as :attr:`PPSD.psd_periods`
        (:attr:`PPSD.psd_frequencies`).
        """
        return self._binned_psds

    @property
    def times_processed(self):
        return [UTCDateTime(ns=ns) for ns in self._times_processed]

    @property
    def times_data(self):
        return [(UTCDateTime(ns=t1), UTCDateTime(ns=t2))
                for t1, t2 in self._times_data]

    @property
    def times_gaps(self):
        return [(UTCDateTime(ns=t1), UTCDateTime(ns=t2))
                for t1, t2 in self._times_gaps]

    @property
    def current_histogram(self):
        self.__check_histogram()
        return self._current_hist_stack

    @property
    def current_histogram_cumulative(self):
        self.__check_histogram()
        return self._current_hist_stack_cumulative

    @property
    def current_histogram_count(self):
        self.__check_histogram()
        return len(self._current_times_used)

    @property
    def current_times_used(self):
        self.__check_histogram()
        return [UTCDateTime(ns=ns) for ns in self._current_times_used]

    def _setup_period_binning(self, period_smoothing_width_octaves,
                              period_step_octaves, period_limits):
        """
        Set up period binning.
        """
        # we step through the period range at step width controlled by
        # period_step_octaves (default 1/8 octave)
        period_step_factor = 2 ** period_step_octaves
        # the width of frequencies we average over for every bin is controlled
        # by period_smoothing_width_octaves (default one full octave)
        period_smoothing_width_factor = \
            2 ** period_smoothing_width_octaves
        # calculate left/right edge and center of first period bin
        # set first smoothing bin's left edge such that the center frequency is
        # the lower limit specified by the user (or the lowest period in the
        # psd)
        per_left = (period_limits[0] /
                    (period_smoothing_width_factor ** 0.5))
        per_right = per_left * period_smoothing_width_factor
        per_center = math.sqrt(per_left * per_right)
        # build up lists
        per_octaves_left = [per_left]
        per_octaves_right = [per_right]
        per_octaves_center = [per_center]
        # do this for the whole period range and append the values to our lists
        while per_center < period_limits[1]:
            # move left edge of smoothing bin further
            per_left *= period_step_factor
            # determine right edge of smoothing bin
            per_right = per_left * period_smoothing_width_factor
            # determine center period of smoothing/binning
            per_center = math.sqrt(per_left * per_right)
            # append to lists
            per_octaves_left.append(per_left)
            per_octaves_right.append(per_right)
            per_octaves_center.append(per_center)
        per_octaves_left = np.array(per_octaves_left)
        per_octaves_right = np.array(per_octaves_right)
        per_octaves_center = np.array(per_octaves_center)
        valid = per_octaves_right > self.psd_periods[0]
        valid &= per_octaves_left < self.psd_periods[-1]
        per_octaves_left = per_octaves_left[valid]
        per_octaves_right = per_octaves_right[valid]
        per_octaves_center = per_octaves_center[valid]
        self._period_binning = np.vstack([
            # left edge of smoothing (for calculating the bin value from psd
            per_octaves_left,
            # left xedge of bin (for plotting)
            per_octaves_center / (period_step_factor ** 0.5),
            # bin center (for plotting)
            per_octaves_center,
            # right xedge of bin (for plotting)
            per_octaves_center * (period_step_factor ** 0.5),
            # right edge of smoothing (for calculating the bin value from psd
            per_octaves_right])

    def __sanity_check(self, trace):
        """
        Checks if trace is compatible for use in the current PPSD instance.
        Returns True if trace can be used or False if not.

        :type trace: :class:`~obspy.core.trace.Trace`
        """
        if trace.id != self.id:
            return False
        if trace.stats.sampling_rate != self.sampling_rate:
            return False
        return True

    def __insert_processed_data(self, utcdatetime, spectrum,
                                archive_entry=None):
        """
        Inserts the given UTCDateTime and processed/octave-binned spectrum at
        the right position in the lists, keeping the order intact.

        Replaces old
        `obspy.signal.spectral_estimation.PPSD.__insert_used_time()`
        private method and the addition ot the histogram stack that was
        performed directly in
        :meth:`obspy.signal.spectral_estimation.PPSD.__process()`.

        :type utcdatetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :type spectrum: :class:`numpy.ndarray`
        :type archive_entry: dict or None
        :param archive_entry: Spectra archive entry belonging to the inserted
            time window (see :meth:`__process`). Mandatory when spectra
            archiving is enabled, so that every window's archive entry
            travels together with its psd data.
        """
        if self.archive_spectra and archive_entry is None:
            msg = ("PPSD has spectra archiving enabled, but no archive "
                   "entry was provided together with the processed data.")
            raise ValueError(msg)
        t = utcdatetime._ns
        ind = bisect.bisect(self._times_processed, t)
        self._times_processed.insert(ind, t)
        self._binned_psds.insert(ind, spectrum)
        if self.archive_spectra:
            self._psd_archive.insert(ind, archive_entry)

    def __insert_gap_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_gaps += [[gap[4]._ns, gap[5]._ns]
                             for gap in stream.get_gaps()]

    def __insert_data_times(self, stream):
        """
        Gets gap information of stream and adds the encountered gaps to the gap
        list of the PPSD instance.

        :type stream: :class:`~obspy.core.stream.Stream`
        """
        self._times_data += \
            [[tr.stats.starttime._ns, tr.stats.endtime._ns]
             for tr in stream]

    def __check_time_present(self, utcdatetime):
        """
        Checks if the given UTCDateTime is already part of the current PPSD
        instance. That is, checks if from utcdatetime to utcdatetime plus
        ppsd_length there is already data in the PPSD.
        Returns True if adding ppsd_length starting at the given time
        would result in an overlap of the ppsd data base, False if it is OK to
        insert this piece of data.
        """
        if not self._times_processed:
            return False
        # new data comes before existing data.
        if utcdatetime._ns < self._times_processed[0]:
            overlap_seconds = (
                (utcdatetime._ns + self.ppsd_length * 1e9) -
                self._times_processed[0]) / 1e9
            # the new data is welcome if any overlap that would be introduced
            # is less or equal than the overlap used by default on continuous
            # data.
            if overlap_seconds / self.ppsd_length > self.overlap:
                return True
            else:
                return False
        # new data exactly at start of first data segment
        elif utcdatetime._ns == self._times_processed[0]:
            return True
        # new data comes after existing data.
        elif utcdatetime._ns > self._times_processed[-1]:
            overlap_seconds = (
                (self._times_processed[-1] + self.ppsd_length * 1e9) -
                utcdatetime._ns) / 1e9
            # the new data is welcome if any overlap that would be introduced
            # is less or equal than the overlap used by default on continuous
            # data.
            if overlap_seconds / self.ppsd_length > self.overlap:
                return True
            else:
                return False
        # new data exactly at start of last data segment
        elif utcdatetime._ns == self._times_processed[-1]:
            return True
        # otherwise we are somewhere within the currently already present time
        # range..
        else:
            index1 = bisect.bisect_left(self._times_processed,
                                        utcdatetime._ns)
            index2 = bisect.bisect_right(self._times_processed,
                                         utcdatetime._ns)
            # if bisect left/right gives same result, we are not exactly at one
            # sampling point but in between to timestamps
            if index1 == index2:
                t1 = self._times_processed[index1 - 1]
                t2 = self._times_processed[index1]
                # check if we are overlapping on left side more than the normal
                # overlap specified during init
                overlap_seconds_left = (
                    (t1 + self.ppsd_length * 1e9) - utcdatetime._ns) / 1e9
                # check if we are overlapping on right side more than the
                # normal overlap specified during init
                overlap_seconds_right = (
                    (utcdatetime._ns + self.ppsd_length * 1e9) - t2) / 1e9
                max_overlap = max(overlap_seconds_left,
                                  overlap_seconds_right) / self.ppsd_length
                if max_overlap > self.overlap:
                    return True
                else:
                    return False
            # if bisect left/right gives different results, we are at exactly
            # one timestamp that is already present
            else:
                return True
        raise NotImplementedError('This should not happen, please report on '
                                  'github.')

    def __check_histogram(self):
        # check if any data has been added yet
        if self._current_hist_stack is None:
            if self._times_processed:
                self.calculate_histogram()
            else:
                msg = 'No data accumulated'
                raise Exception(msg)

    def __invalidate_histogram(self):
        self._current_hist_stack = None
        self._current_hist_stack_cumulative = None
        self._current_times_used = []
        self._current_times_all_details = []

    def add(self, stream, verbose=False):
        """
        Process all traces with compatible information and add their spectral
        estimates to the histogram containing the probabilistic psd.
        Also ensures that no piece of data is inserted twice.

        :type stream: :class:`~obspy.core.stream.Stream` or
                :class:`~obspy.core.trace.Trace`
        :param stream: Stream or trace with data that should be added to the
                probabilistic psd histogram.
        :returns: True if appropriate data were found and the ppsd statistics
                were changed, False otherwise.
        """
        if self.metadata is None:
            msg = ("PPSD instance has no metadata attached, which are needed "
                   "for processing the data. When using 'PPSD.load_npz()' use "
                   "'metadata' kwarg to provide metadata.")
            raise Exception(msg)
        if self.archive_spectra:
            if not isinstance(self.metadata, Inventory):
                msg = ("PPSD instance has spectra archiving enabled, which "
                       "requires metadata provided as an Inventory object, "
                       "got: %s" % type(self.metadata))
                raise TypeError(msg)
            self._validate_metadata_version(self.metadata_version)
        # return later if any changes were applied to the ppsd statistics
        changed = False
        # prepare the list of traces to go through
        if isinstance(stream, Trace):
            stream = Stream([stream])
        if not stream:
            msg = 'Empty stream object provided to PPSD.add()'
            warnings.warn(msg)
            return False
        # select appropriate traces
        stream = stream.select(id=self.id)
        if not stream:
            msg = 'No traces with matching SEED ID in provided stream object.'
            warnings.warn(msg)
            return False
        stream = stream.select(sampling_rate=self.sampling_rate)
        if not stream:
            msg = ('No traces with matching sampling rate in provided stream '
                   'object.')
            warnings.warn(msg)
            return False
        # save information on available data and gaps
        self.__insert_data_times(stream)
        self.__insert_gap_times(stream)
        # merge depending on skip_on_gaps set during __init__
        stream.merge(self.merge_method, fill_value=0)

        for tr in stream:
            # the following check should not be necessary due to the select()..
            if not self.__sanity_check(tr):
                msg = "Skipping incompatible trace."
                warnings.warn(msg)
                continue
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            if t1 + self.ppsd_length - tr.stats.delta > t2:
                msg = (f"Trace is shorter than this PPSD's 'ppsd_length' "
                       f"({str(self.ppsd_length)} seconds). Skipping trace: "
                       f"{str(tr)}")
                warnings.warn(msg)
                continue
            while t1 + self.ppsd_length - tr.stats.delta <= t2:
                if self.__check_time_present(t1):
                    msg = "Already covered time spans detected (e.g. %s), " + \
                          "skipping these slices."
                    msg = msg % t1
                    warnings.warn(msg)
                else:
                    # throw warnings if trace length is different
                    # than ppsd_length..!?!
                    slice = tr.slice(t1, t1 + self.ppsd_length -
                                     tr.stats.delta)
                    # XXX not good, should be working in place somehow
                    # XXX how to do it with the padding, though?
                    success = self.__process(slice, t1)
                    if success:
                        if verbose:
                            print(t1)
                        changed = True
                t1 += self.step  # advance

            # enforce time limits, pad zeros if gaps
            # tr.trim(t, t+PPSD_LENGTH, pad=True)
        if changed:
            self.__invalidate_histogram()
        return changed

    def __process(self, tr, t):
        """
        Processes a segment of data and save the psd information.
        Whether `Trace` is compatible (station, channel, ...) has to
        checked beforehand.

        :type tr: :class:`~obspy.core.trace.Trace`
        :param tr: Compatible Trace with data of one PPSD segment
        :type t: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param t: Start time of the time window the data was cut for. This can
            be different usually on a subsample scale and then in some cases
            can lead to a directly following time segment being left out as
            they seem to overlap.
        :returns: `True` if segment was successfully processed,
            `False` otherwise.
        """

        # XXX DIRTY HACK!!
        if len(tr) == self.len + 1:
            tr.data = tr.data[:-1]
        # one last check..
        if len(tr) != self.len:
            msg = ("Got a piece of data with wrong length. Skipping:\n" +
                   str(tr))
            warnings.warn(msg)
            return False
        # being paranoid, only necessary if in-place operations would follow
        tr.data = tr.data.astype(np.float64)
        # if trace has a masked array we fill in zeros
        try:
            tr.data[tr.data.mask] = 0.0
        # if it is no masked array, we get an AttributeError
        # and have nothing to do
        except AttributeError:
            pass

        # restitution:
        # mcnamara apply the correction at the end in freq-domain,
        # does it make a difference?
        # probably should be done earlier on bigger chunk of data?!
        # Yes, you should avoid removing the response until after you
        # have estimated the spectra to avoid elevated lp noise

        spec, _freq = mlab.psd(tr.data, self.nfft, self.sampling_rate,
                               detrend=mlab.detrend_linear, window=fft_taper,
                               noverlap=self.nlap, sides='onesided',
                               scale_by_freq=True)

        # leave out first entry (offset)
        spec = spec[1:]

        # working with the periods not frequencies later so reverse spectrum
        spec = spec[::-1]

        # when archiving spectra for later recalibration, keep a copy of the
        # raw (i.e. not instrument corrected) one-sided spectrum. No
        # time-domain samples are kept, this spectral representation is
        # sufficient to redo the instrument correction with updated response
        # information (see recalibrate()).
        if self.archive_spectra:
            raw_spectrum = np.array(spec, dtype=np.float64, order="C")

        # Here we remove the response using the same conventions
        # since the power is squared we want to square the sensitivity
        # we can also convert to acceleration if we have non-rotational data
        if self.special_handling == "ringlaser":
            # in case of rotational data just remove sensitivity
            spec /= self.metadata['sensitivity'] ** 2
        # special_handling "hydrophone" does instrument correction same as
        # "normal" data
        else:
            # determine instrument response from metadata
            try:
                if self.archive_spectra:
                    # use the strict channel epoch lookup, so that the
                    # calibration record of the archived window unambiguously
                    # documents which channel epoch was used
                    resp, channel_epoch = \
                        self._get_response_from_inventory_strict(tr)
                else:
                    resp = self._get_response(tr)
            except Exception as e:
                msg = ("Error getting response from provided metadata:\n"
                       "%s: %s\n"
                       "Skipping time segment(s).")
                msg = msg % (e.__class__.__name__, str(e))
                warnings.warn(msg)
                return False

            resp = resp[1:]
            resp = resp[::-1]
            # Now get the amplitude response (squared)
            respamp = np.absolute(resp * np.conjugate(resp))
            # Make omega with the same conventions as spec
            w = 2.0 * math.pi * _freq[1:]
            w = w[::-1]
            # Here we do the response removal
            if self.special_handling in ("hydrophone", "infrasound"):
                spec = spec / respamp
            else:
                spec = (w ** 2) * spec / respamp

        # clip, convert to dB and smooth/bin into the period binning
        smoothed_psd = self._bin_spectrum_db(spec)

        if self.archive_spectra:
            archive_entry = {
                'spectrum': raw_spectrum,
                'version': self.metadata_version,
                'channel_id': self.id,
                # the actual time at which the response was looked up (start
                # time of the sliced trace, which can differ slightly from
                # the window start time 't' on a subsample scale)
                'lookup_time': tr.stats.starttime._ns,
                'epoch_start': (channel_epoch.start_date._ns
                                if channel_epoch.start_date is not None
                                else None),
                'epoch_end': (channel_epoch.end_date._ns
                              if channel_epoch.end_date is not None
                              else None),
            }
        else:
            archive_entry = None
        self.__insert_processed_data(t, smoothed_psd,
                                     archive_entry=archive_entry)
        return True

    def _bin_spectrum_db(self, spec):
        """
        Clip a linear, period-ordered spectrum at a tiny positive value,
        convert it to dB and smooth/bin it into the period binning of the
        PPSD. This is the common final processing step of :meth:`__process`
        and :meth:`recalibrate`, so that spectra recomputed from the spectra
        archive are computed exactly like during regular processing.

        :type spec: :class:`numpy.ndarray`
        :param spec: Linear spectrum, ordered like :attr:`psd_periods`. The
            array is modified in place.
        :rtype: :class:`numpy.ndarray` (float32)
        :returns: Smoothed and binned psd values in dB.
        """
        # avoid calculating log of zero
        idx = spec < dtiny
        spec[idx] = dtiny

        # go to dB
        spec = np.log10(spec)
        spec *= 10

        smoothed_psd = []
        # do this for the whole period range and append the values to our lists
        for per_left, per_right in zip(self.period_bin_left_edges,
                                       self.period_bin_right_edges):
            specs = spec[(per_left <= self.psd_periods) &
                         (self.psd_periods <= per_right)]
            smoothed_psd.append(specs.mean())
        return np.array(smoothed_psd, dtype=np.float32)

    def _get_times_all_details(self):
        # check if we can reuse a previously cached array of all times as
        # day of week as int and time of day in float hours
        if len(self._current_times_all_details) == len(self._times_processed):
            return self._current_times_all_details
        # otherwise compute it and store it for subsequent stacks on the
        # same data (has to be recomputed when additional data gets added)
        else:
            dtype = np.dtype([('time_of_day', np.float32),
                              ('iso_weekday', np.int8),
                              ('iso_week', np.int8),
                              ('year', np.int16),
                              ('month', np.int8)])
            times_all_details = np.empty(shape=len(self._times_processed),
                                         dtype=dtype)
            utc_times_all = [UTCDateTime(ns=t) for t in self._times_processed]
            times_all_details['time_of_day'][:] = \
                [t._get_hours_after_midnight() for t in utc_times_all]
            times_all_details['iso_weekday'][:] = \
                [t.isoweekday() for t in utc_times_all]
            times_all_details['iso_week'][:] = \
                [t.isocalendar()[1] for t in utc_times_all]
            times_all_details['year'][:] = \
                [t.year for t in utc_times_all]
            times_all_details['month'][:] = [t.month for t in utc_times_all]
            self._current_times_all_details = times_all_details
            return times_all_details

    def _stack_selection(self, starttime=None, endtime=None,
                         time_of_weekday=None, year=None, month=None,
                         isoweek=None, callback=None):
        """
        For details on restrictions see :meth:`calculate_histogram`.

        :rtype: :class:`numpy.ndarray` of bool
        :returns: Boolean array of which psd pieces should be included in the
            stack.
        """
        times_all = np.array(self._times_processed)
        selected = np.ones(len(times_all), dtype=bool)
        if starttime is not None:
            selected &= times_all >= starttime._ns
        if endtime is not None:
            selected &= times_all <= endtime._ns
        if time_of_weekday is not None:
            times_all_details = self._get_times_all_details()
            # we need to do a logical OR over all different user specified time
            # windows, so we start with an array of False and set all matching
            # pieces True for the final logical AND against the previous
            # restrictions
            selected_time_of_weekday = np.zeros(len(times_all), dtype=bool)
            for weekday, start, end in time_of_weekday:
                if weekday == -1:
                    selected_ = np.ones(len(times_all), dtype=bool)
                else:
                    selected_ = (
                        times_all_details['iso_weekday'] == weekday)
                selected_ &= times_all_details['time_of_day'] >= start
                selected_ &= times_all_details['time_of_day'] <= end
                selected_time_of_weekday |= selected_
            selected &= selected_time_of_weekday
        if year is not None:
            try:
                year[0]
            except TypeError:
                year = [year]
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['year'] == year[0]
            for year_ in year[1:]:
                selected_ |= times_all_details['year'] == year_
            selected &= selected_
        if month is not None:
            try:
                month[0]
            except TypeError:
                month = [month]
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['month'] == month[0]
            for month_ in month[1:]:
                selected_ |= times_all_details['month'] == month_
            selected &= selected_
        if isoweek is not None:
            times_all_details = self._get_times_all_details()
            selected_ = times_all_details['iso_week'] == isoweek[0]
            for isoweek_ in isoweek[1:]:
                selected_ |= times_all_details['iso_week'] == isoweek_
            selected &= selected_
        if callback is not None:
            selected &= callback(times_all)
        return selected

    def calculate_histogram(self, starttime=None, endtime=None,
                            time_of_weekday=None, year=None, month=None,
                            isoweek=None, callback=None):
        """
        Calculate and set current 2D histogram stack, optionally with start-
        and endtime and time of day restrictions.

        .. note::
            All restrictions to the stack are evaluated as a logical AND, i.e.
            only individual psd pieces are included in the stack that match
            *all* of the specified restrictions (e.g. `isoweek=40, month=2` can
            never match any data).

        .. note::
            All time restrictions are specified in UTC, so actual time in local
            time zone might not be the same across start/end date of daylight
            saving time periods.

        .. note::
            Time restrictions only check the starttime of the individual psd
            pieces.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: If set, data before the specified time is excluded
            from the returned stack.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: If set, data after the specified time is excluded
            from the returned stack.
        :type time_of_weekday: list(int, float, float)
        :param time_of_weekday: If set, restricts the data that is included
            in the stack by time of day and weekday. Monday is `1`, Sunday is
            `7`, `-1` for any day of week. For example, using
            `time_of_weekday=[(-1, 0, 2), (-1, 22, 24)]` only individual
            spectra that have a starttime in between 10pm and 2am are used in
            the stack for all days of week, using
            `time_of_weekday=[(5, 22, 24), (6, 0, 2), (6, 22, 24), (7, 0, 2)]`
            only spectra with a starttime in between Friday 10pm to Saturdays
            2am and Saturday 10pm to Sunday 2am are used.
            Note that time of day is specified in UTC (time of day might have
            to be adapted to daylight saving time). Also note that this setting
            filters only by starttime of the used psd time slice, so the length
            of individual slices (set at initialization:
            :meth:`PPSD(..., ppsd_length=XXX, ...) <PPSD.__init__>` in seconds)
            has to be taken into consideration (e.g. with a `ppsd_length` of
            one hour and a `time_of_weekday` restriction to 10pm-2am
            actually includes data from 10pm-3am).
        :type year: list[int]
        :param year: If set, restricts the data that is included in the stack
            by year. For example, using `year=[2015]` only individual spectra
            from year 2015 are used in the stack, using `year=[2013, 2015]`
            only spectra from exactly year 2013 or exactly year 2015 are used.
        :type month: list[int]
        :param month: If set, restricts the data that is included in the stack
            by month of year. For example, using `month=[2]` only individual
            spectra from February are used in the stack, using `month=[4, 7]`
            only spectra from exactly April or exactly July are used.
        :type isoweek: list[int]
        :param isoweek: If set, restricts the data that is included in the
            stack by ISO week number of year. For example, using `isoweek=[2]`
            only individual spectra from 2nd ISO week of any year are used in
            the stack, using `isoweek=[4, 7]` only spectra from exactly 4th ISO
            week or exactly 7th ISO week are used.
        :type callback: callable
        :param callback: Custom user defined callback function that can be used
            for more complex scenarios to specify whether an individual psd
            piece should be included in the stack or not. The function will be
            passed an array with the starttimes of all psd pieces (as a POSIX
            timestamp that can be used as a single argument to initialize a
            :class:`~obspy.core.utcdatetime.UTCDateTime` object) and
            should return a boolean array specifying which psd pieces should be
            included in the stack (`True`) and which should be excluded
            (`False`). Note that even when callback returns `True` the psd
            piece will be excluded if it does not match all other criteria
            (e.g. `starttime`).
        :rtype: None
        """
        # no data at all
        if not self._times_processed:
            self._current_hist_stack = None
            self._current_hist_stack_cumulative = None
            self._current_times_used = []
            return

        # determine which psd pieces should be used in the stack,
        # based on all selection criteria specified by user
        selected = self._stack_selection(
            starttime=starttime, endtime=endtime,
            time_of_weekday=time_of_weekday, year=year, month=month,
            isoweek=isoweek, callback=callback)
        used_indices = selected.nonzero()[0]
        used_count = len(used_indices)
        used_times = np.array(self._times_processed)[used_indices]

        num_period_bins = len(self.period_bin_centers)
        num_db_bins = len(self.db_bin_centers)

        # initial setup of 2D histogram
        hist_stack = np.zeros((num_period_bins, num_db_bins), dtype=np.uint64)

        # empty selection, set all histogram stacks to zeros
        if not used_count:
            self._current_hist_stack = hist_stack
            self._current_hist_stack_cumulative = np.zeros_like(
                hist_stack, dtype=np.float32)
            self._current_times_used = used_times
            return

        # concatenate all used spectra, evaluate index of amplitude bin each
        # value belongs to
        inds = np.hstack([self._binned_psds[i] for i in used_indices])
        # for "inds" now a number of ..
        #   - 0 means below lowest bin (bin index 0)
        #   - 1 means, hit lowest bin (bin index 0)
        #   - ..
        #   - len(self.db_bin_edges) means above top bin
        # we need minus one because searchsorted returns the insertion index in
        # the array of bin edges which is the index of the corresponding bin
        # plus one
        inds = self.db_bin_edges.searchsorted(inds, side="left") - 1
        # for "inds" now a number of ..
        #   - -1 means below lowest bin (bin index 0)
        #   - 0 means, hit lowest bin (bin index 0)
        #   - ..
        #   - (len(self.db_bin_edges)-1) means above top bin
        # values that are left of first bin edge have to be moved back into the
        # binning
        inds[inds == -1] = 0
        # same goes for values right of last bin edge
        inds[inds == num_db_bins] -= 1
        # reshape such that we can iterate over the array, extracting for
        # each period bin an array of all amplitude bins we have hit
        inds = inds.reshape((used_count, num_period_bins)).T
        for i, inds_ in enumerate(inds):
            # count how often each bin has been hit for this period bin,
            # set the current 2D histogram column accordingly
            hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

        # calculate and set the cumulative version (i.e. going from 0 to 1 from
        # low to high psd values for every period column) of the current
        # histogram stack.
        # sum up the columns to cumulative entries
        hist_stack_cumul = hist_stack.cumsum(axis=1)
        # normalize every column with its overall number of entries
        # (can vary from the number of self.times_processed because of values
        #  outside the histogram db ranges)
        norm = hist_stack_cumul[:, -1].copy().astype(np.float64)
        # avoid zero division
        norm[norm == 0] = 1
        hist_stack_cumul = (hist_stack_cumul.T / norm).T
        # set everything that was calculated
        self._current_hist_stack = hist_stack
        self._current_hist_stack_cumulative = hist_stack_cumul
        self._current_times_used = used_times

    def _get_response(self, tr):
        # check type of metadata and use the correct subroutine
        # first, to save some time, tried to do this in __init__ like:
        #   self._get_response = self._get_response_from_inventory
        # but that makes the object non-picklable
        if isinstance(self.metadata, Inventory):
            return self._get_response_from_inventory(tr)
        elif isinstance(self.metadata, Parser):
            return self._get_response_from_parser(tr)
        elif isinstance(self.metadata, dict):
            return self._get_response_from_paz_dict(tr)
        elif isinstance(self.metadata, str):
            return self._get_response_from_resp(tr)
        else:
            msg = "Unexpected type for `metadata`: %s" % type(self.metadata)
            raise TypeError(msg)

    def _get_response_from_inventory(self, tr):
        inventory = self.metadata
        response = inventory.get_response(self.id, tr.stats.starttime)
        resp, _ = response.get_evalresp_response(
            t_samp=self.delta, nfft=self.nfft, output="VEL")
        return resp

    def _get_response_from_inventory_strict(self, tr):
        """
        Get instrument response from Inventory metadata using the strict
        channel epoch lookup (:meth:`Inventory.get_channel_epoch`), which
        raises an error if no or more than one matching channel epoch with
        response information is found.

        :returns: 2-tuple with the response (like
            :meth:`_get_response_from_inventory`) and the
            :class:`~obspy.core.inventory.channel.Channel` epoch that was
            used.
        """
        inventory = self.metadata
        channel_epoch = inventory.get_channel_epoch(self.id,
                                                    tr.stats.starttime)
        resp, _ = channel_epoch.response.get_evalresp_response(
            t_samp=self.delta, nfft=self.nfft, output="VEL")
        return resp, channel_epoch

    def _get_response_from_parser(self, tr):
        parser = self.metadata
        resp_key = "RESP." + self.id
        for key, resp_file in parser.get_resp():
            if key == resp_key:
                break
        else:
            msg = "Response for %s not found in Parser" % self.id
            raise ValueError(msg)
        resp_file.seek(0, 0)
        resp = evalresp(t_samp=self.delta, nfft=self.nfft,
                        filename=resp_file, date=tr.stats.starttime,
                        station=self.station, channel=self.channel,
                        network=self.network, locid=self.location,
                        units="VEL", freq=False, debug=False)
        return resp

    def _get_response_from_paz_dict(self, tr):  # @UnusedVariable
        paz = self.metadata
        resp = paz_to_freq_resp(paz['poles'], paz['zeros'],
                                paz['gain'] * paz['sensitivity'],
                                self.delta, nfft=self.nfft)
        return resp

    def _get_response_from_resp(self, tr):
        resp = evalresp(t_samp=self.delta, nfft=self.nfft,
                        filename=self.metadata, date=tr.stats.starttime,
                        station=self.station, channel=self.channel,
                        network=self.network, locid=self.location,
                        units="VEL", freq=False, debug=False)
        return resp

    def get_percentile(self, percentile=50):
        """
        Returns periods and approximate psd values for given percentile value.

        :type percentile: int
        :param percentile: percentile for which to return approximate psd
                value. (e.g. a value of 50 is equal to the median.)
        :returns: (periods, percentile_values)
        """
        hist_cum = self.current_histogram_cumulative
        if hist_cum is None:
            return None
        # go to percent
        percentile = percentile / 100.0
        if percentile == 0:
            # only for this special case we have to search from the other side
            # (otherwise we always get index 0 in .searchsorted())
            side = "right"
        else:
            side = "left"
        percentile_values = [col.searchsorted(percentile, side=side)
                             for col in hist_cum]
        # map to power db values
        percentile_values = self.db_bin_edges[percentile_values]
        return (self.period_bin_centers, percentile_values)

    def get_mode(self):
        """
        Returns periods and mode psd values (i.e. for each frequency the psd
        value with the highest probability is selected).

        :returns: (periods, psd mode values)
        """
        hist = self.current_histogram
        if hist is None:
            return None
        mode = self.db_bin_centers[hist.argmax(axis=1)]
        return (self.period_bin_centers, mode)

    def get_mean(self):
        """
        Returns periods and mean psd values (i.e. for each frequency the mean
        psd value is selected).

        :returns: (periods, psd mean values)
        """
        hist = self.current_histogram
        if hist is None:
            return None
        hist_count = self.current_histogram_count
        mean = (hist * self.db_bin_centers / hist_count).sum(axis=1)
        return self.period_bin_centers, mean

    def recalibrate(self, metadata, metadata_version, starttime, endtime):
        """
        Recompute archived spectra in a given time span with updated
        instrument response information.

        This recomputes the psd values of all archived time windows whose
        start time falls into the half-open interval ``[starttime, endtime)``
        from their archived raw (not instrument corrected) spectra, using
        the response information of the given new metadata. It is intended
        for the case that corrected metadata (e.g. a corrected StationXML
        file) becomes available after the original waveform data is no
        longer at hand. The PPSD must have been set up with
        ``archive_spectra=True`` (see :meth:`PPSD.__init__`).

        For every selected window, the response is looked up in the new
        metadata with the strict channel epoch lookup
        (:meth:`obspy.core.inventory.inventory.Inventory.get_channel_epoch`)
        at the very same time that was used for the response lookup when the
        window was originally processed (the start time of the sliced data
        trace, which can differ slightly from the window start time on a
        subsample scale). The recomputed dB values are identical (up to
        floating point accuracy, well below 0.0001 dB) to a fresh processing
        of the same waveform data with the new metadata, since the archived
        float64 spectra go through the exact same computation steps.

        The operation is atomic: if any of the selected windows misses its
        archived spectrum, if the response lookup in the new metadata does
        not yield exactly one valid channel epoch, or if no valid spectrum
        can be computed, an exception is raised and no spectra, calibration
        records or histogram stacks are modified at all.

        Windows outside the given time span are left untouched, both their
        psd values and their calibration records. Window times, their order,
        data coverage and gap statistics are not changed. The current
        histogram stack is invalidated, so that it is recounted from the
        updated psd values on next access. The metadata attached to the PPSD
        object itself is not modified, i.e. a subsequent :meth:`add` still
        uses the original metadata.

        :type metadata: :class:`~obspy.core.inventory.inventory.Inventory`
        :param metadata: New (e.g. corrected) metadata to recompute the
            selected spectra with.
        :type metadata_version: str
        :param metadata_version: Non-empty version label identifying the
            version of the new metadata. It is stored in the calibration
            record of every recalibrated window.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of the time span of windows to recalibrate
            (inclusive), matched against the start times of the archived
            windows.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of the time span of windows to recalibrate
            (exclusive), matched against the start times of the archived
            windows.
        :rtype: list of :class:`~obspy.core.utcdatetime.UTCDateTime`
        :returns: Start times of the windows that were recalibrated.
        :raises PPSDRecalibrationError: if spectra archiving is not enabled,
            if a selected window lacks its archived spectrum, or if no valid
            spectrum can be computed for a selected window.
        :raises NoMatchingChannelEpochError: if the new metadata has no
            channel epoch with response information for a selected window's
            channel and lookup time.
        :raises MultipleMatchingChannelEpochsError: if the new metadata has
            more than one matching channel epoch for a selected window's
            channel and lookup time.
        """
        if not self.archive_spectra:
            msg = ("PPSD instance was not set up to archive spectra "
                   "('archive_spectra=True'), so no information is available "
                   "to recalibrate spectra with updated metadata.")
            raise PPSDRecalibrationError(msg)
        if not isinstance(metadata, Inventory):
            msg = ("Recalibration of archived spectra is currently only "
                   "supported with metadata provided as an Inventory "
                   "object, got: %s" % type(metadata))
            raise TypeError(msg)
        self._validate_metadata_version(metadata_version)
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)
        if not starttime < endtime:
            msg = ("'starttime' (%s) must be smaller than 'endtime' (%s), "
                   "the time span is interpreted as a half-open interval "
                   "[starttime, endtime) matched against the start times of "
                   "the archived spectra." % (starttime, endtime))
            raise ValueError(msg)
        # select windows to recalibrate by their start time, half-open
        # interval [starttime, endtime)
        selected = [i for i, t in enumerate(self._times_processed)
                    if starttime._ns <= t < endtime._ns]
        if not selected:
            msg = ("No archived spectra with start time in half-open "
                   "interval [%s, %s), nothing to recalibrate."
                   % (starttime, endtime))
            warnings.warn(msg)
            return []
        # set up the angular frequency array exactly like during regular
        # processing (see __process), so that recomputed spectra are
        # identical to a fresh processing of the same data
        _, _freq = mlab.psd(np.ones(self.len), self.nfft,
                            self.sampling_rate, noverlap=self.nlap)
        w = 2.0 * math.pi * _freq[1:]
        w = w[::-1]
        # first compute all new spectra and look up all new channel epochs,
        # so that any problem aborts the whole operation before anything is
        # modified
        new_spectra = {}
        new_channel_epochs = {}
        for i in selected:
            t = UTCDateTime(ns=self._times_processed[i])
            entry = None
            if i < len(self._psd_archive):
                entry = self._psd_archive[i]
            required_keys = ('spectrum', 'version', 'channel_id',
                             'lookup_time', 'epoch_start', 'epoch_end')
            if entry is None or \
                    any(key not in entry for key in required_keys):
                msg = ("No (complete) archived spectrum information "
                       "available for time window starting at %s, unable to "
                       "recalibrate. The whole recalibration is aborted, no "
                       "data was modified." % str(t))
                raise PPSDRecalibrationError(msg)
            spec = np.asarray(entry['spectrum'], dtype=np.float64)
            if spec.shape != (len(self._psd_periods),):
                msg = ("Archived spectrum for time window starting at %s "
                       "has unexpected shape %s (expected %s), unable to "
                       "recalibrate. The whole recalibration is aborted, "
                       "no data was modified."
                       % (str(t), spec.shape, (len(self._psd_periods),)))
                raise PPSDRecalibrationError(msg)
            # look up the response in the new metadata at the very same
            # time that was used for the original response lookup
            lookup_time = UTCDateTime(ns=entry['lookup_time'])
            channel_epoch = metadata.get_channel_epoch(
                entry['channel_id'], lookup_time)
            try:
                resp, _ = channel_epoch.response.get_evalresp_response(
                    t_samp=self.delta, nfft=self.nfft, output="VEL")
            except Exception as e:
                msg = ("Error evaluating response of new metadata for time "
                       "window starting at %s:\n%s: %s\n"
                       "The whole recalibration is aborted, no data was "
                       "modified."
                       % (str(t), e.__class__.__name__, str(e)))
                raise PPSDRecalibrationError(msg) from e
            resp = resp[1:]
            resp = resp[::-1]
            # Now get the amplitude response (squared)
            respamp = np.absolute(resp * np.conjugate(resp))
            # redo the instrument correction on the archived raw spectrum
            # (regular seismic channels only, see __init__)
            spec_new = (w ** 2) * spec / respamp
            binned_psd = self._bin_spectrum_db(spec_new)
            if np.isnan(binned_psd).any() or \
                    not np.isfinite(binned_psd).any():
                msg = ("Unable to compute a valid spectrum for time window "
                       "starting at %s with the new metadata (result "
                       "contains NaN or no finite values at all). The whole "
                       "recalibration is aborted, no data was modified."
                       % str(t))
                raise PPSDRecalibrationError(msg)
            new_spectra[i] = binned_psd
            new_channel_epochs[i] = channel_epoch
        # all computations succeeded, now apply the changes
        for i in selected:
            self._binned_psds[i] = new_spectra[i]
            entry = self._psd_archive[i]
            channel_epoch = new_channel_epochs[i]
            entry['version'] = metadata_version
            entry['epoch_start'] = (channel_epoch.start_date._ns
                                    if channel_epoch.start_date is not None
                                    else None)
            entry['epoch_end'] = (channel_epoch.end_date._ns
                                  if channel_epoch.end_date is not None
                                  else None)
        # invalidate the current histogram stack, so it gets recounted from
        # the updated psd values on next access
        self.__invalidate_histogram()
        return [UTCDateTime(ns=self._times_processed[i]) for i in selected]

    def get_archive_info(self, starttime=None):
        """
        Get calibration records of archived spectra.

        Only usable when the PPSD was set up with ``archive_spectra=True``
        (or loaded from an npz file with spectra archive information).

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param starttime: If given, return only the calibration record of
            the archived spectrum with exactly this start time.
        :rtype: dict or list of dict
        :returns: Calibration record(s) with keys ``'version'`` (metadata
            version label), ``'channel_id'`` (SEED ID the response was
            looked up for), ``'lookup_time'`` (the actual time at which the
            response was looked up, as
            :class:`~obspy.core.utcdatetime.UTCDateTime`) and
            ``'epoch_start'``/``'epoch_end'`` (validity time span of the
            channel epoch that was used, as
            :class:`~obspy.core.utcdatetime.UTCDateTime` or ``None`` for an
            open end).
        """
        if not self.archive_spectra:
            msg = ("PPSD instance was not set up to archive spectra "
                   "('archive_spectra=True'), no calibration records "
                   "available.")
            raise PPSDRecalibrationError(msg)

        def _info(entry):
            epoch_start = entry['epoch_start']
            if epoch_start is not None:
                epoch_start = UTCDateTime(ns=epoch_start)
            epoch_end = entry['epoch_end']
            if epoch_end is not None:
                epoch_end = UTCDateTime(ns=epoch_end)
            return {'version': entry['version'],
                    'channel_id': entry['channel_id'],
                    'lookup_time': UTCDateTime(ns=entry['lookup_time']),
                    'epoch_start': epoch_start,
                    'epoch_end': epoch_end}

        if starttime is None:
            return [_info(entry) for entry in self._psd_archive]
        t = UTCDateTime(starttime)._ns
        index = bisect.bisect_left(self._times_processed, t)
        if index >= len(self._times_processed) or \
                self._times_processed[index] != t:
            msg = ("No archived spectrum with start time %s."
                   % str(UTCDateTime(starttime)))
            raise KeyError(msg)
        return _info(self._psd_archive[index])

    def save_npz(self, filename):
        """
        Saves the PPSD as a compressed numpy binary (npz format).

        The resulting file can be restored using `my_ppsd.load_npz(filename)`.

        If spectra archiving is enabled (see :meth:`PPSD.__init__`), the
        spectra archive is stored in additional arrays in the same npz file.
        The file can still be loaded as a plain PPSD by ObsPy versions
        without spectra archiving support (the additional arrays are ignored
        then).

        :type filename: str
        :param filename: Name of numpy .npz output file
        """
        out = {}
        for key in self.NPZ_STORE_KEYS:
            value = getattr(self, key)
            # Some values need to be replaced to allow non-pickle
            # serialization (#2409).
            if key in self.NPZ_STORE_KEYS_SIMPLE_TYPES:
                value = self.NPZ_SIMPLE_TYPE_MAP.get(value, value)
            out[key] = value
        if self.archive_spectra:
            out.update(self._archive_npz_arrays())
        np.savez_compressed(filename, **out)

    def _archive_npz_arrays(self):
        """
        Convert the per-window spectra archive into a dictionary of numpy
        arrays for storage in the npz file (see :meth:`save_npz`).
        """
        count = len(self._psd_archive)
        spectra = np.empty((count, len(self._psd_periods)),
                           dtype=np.float64)
        lookup_times = np.zeros(count, dtype=np.int64)
        epoch_starts = np.zeros(count, dtype=np.int64)
        epoch_ends = np.zeros(count, dtype=np.int64)
        epoch_starts_open = np.zeros(count, dtype=np.bool_)
        epoch_ends_open = np.zeros(count, dtype=np.bool_)
        versions = []
        channel_ids = []
        for i, entry in enumerate(self._psd_archive):
            spectra[i] = entry['spectrum']
            versions.append(entry['version'])
            channel_ids.append(entry['channel_id'])
            lookup_times[i] = entry['lookup_time']
            if entry['epoch_start'] is None:
                epoch_starts_open[i] = True
            else:
                epoch_starts[i] = entry['epoch_start']
            if entry['epoch_end'] is None:
                epoch_ends_open[i] = True
            else:
                epoch_ends[i] = entry['epoch_end']
        out = {
            self.NPZ_ARCHIVE_FORMAT_KEY:
                np.int64(self.NPZ_ARCHIVE_FORMAT_VERSION),
            '_psd_archive_spectra': spectra,
            '_psd_archive_versions': np.array(versions),
            '_psd_archive_channel_ids': np.array(channel_ids),
            '_psd_archive_lookup_times': lookup_times,
            '_psd_archive_epoch_starts': epoch_starts,
            '_psd_archive_epoch_ends': epoch_ends,
            '_psd_archive_epoch_starts_open': epoch_starts_open,
            '_psd_archive_epoch_ends_open': epoch_ends_open,
        }
        return out

    @staticmethod
    def _read_archive_npz(data):
        """
        Read spectra archive entries from an open npz file (see
        :meth:`load_npz` and :meth:`_add_npz`).

        :returns: List of per-window spectra archive entry dictionaries, or
            ``None`` if the npz file does not contain spectra archive
            information.
        :raises ValueError: if the spectra archive information in the npz
            file is inconsistent.
        """
        if PPSD.NPZ_ARCHIVE_FORMAT_KEY not in data:
            return None
        format_version = int(data[PPSD.NPZ_ARCHIVE_FORMAT_KEY].item())
        if format_version != PPSD.NPZ_ARCHIVE_FORMAT_VERSION:
            msg = ("Unsupported PPSD spectra archive format version %d in "
                   "npz file (this ObsPy version supports version %d)."
                   % (format_version, PPSD.NPZ_ARCHIVE_FORMAT_VERSION))
            raise ValueError(msg)
        spectra = data['_psd_archive_spectra']
        count = spectra.shape[0]
        if spectra.shape != (count, len(data['_psd_periods'])):
            msg = ("Inconsistent spectra archive in npz file: archived "
                   "spectra have shape %s, but %d spectral points were "
                   "expected from the stored psd periods."
                   % (spectra.shape, len(data['_psd_periods'])))
            raise ValueError(msg)
        versions = [str(v) for v in data['_psd_archive_versions'].tolist()]
        channel_ids = [str(v)
                       for v in data['_psd_archive_channel_ids'].tolist()]
        lookup_times = data['_psd_archive_lookup_times'].tolist()
        epoch_starts = data['_psd_archive_epoch_starts'].tolist()
        epoch_ends = data['_psd_archive_epoch_ends'].tolist()
        epoch_starts_open = data['_psd_archive_epoch_starts_open'].tolist()
        epoch_ends_open = data['_psd_archive_epoch_ends_open'].tolist()
        for key, values in (('versions', versions),
                            ('channel_ids', channel_ids),
                            ('lookup_times', lookup_times),
                            ('epoch_starts', epoch_starts),
                            ('epoch_ends', epoch_ends),
                            ('epoch_starts_open', epoch_starts_open),
                            ('epoch_ends_open', epoch_ends_open)):
            if len(values) != count:
                msg = ("Inconsistent spectra archive in npz file: %d "
                       "archived spectra but %d entries in '%s'."
                       % (count, len(values), key))
                raise ValueError(msg)
        entries = []
        for i in range(count):
            entries.append({
                'spectrum': np.asarray(spectra[i], dtype=np.float64),
                'version': versions[i],
                'channel_id': channel_ids[i],
                'lookup_time': int(lookup_times[i]),
                'epoch_start': (None if epoch_starts_open[i]
                                else int(epoch_starts[i])),
                'epoch_end': (None if epoch_ends_open[i]
                              else int(epoch_ends[i])),
            })
        return entries

    @staticmethod
    def load_npz(filename, metadata=None, allow_pickle=False,
                 metadata_version=None):
        """
        Load previously computed PPSD results.

        Load previously computed PPSD results from a
        compressed numpy binary in npz format, written with
        :meth:`~obspy.signal.spectral_estimation.PPSD.save_npz`.
        If more data are to be added and processed, metadata have to be
        specified again during loading because they are not
        stored in the npz format.

        :type filename: str
        :param filename: Name of numpy .npz file with stored PPSD data
        :type metadata: :class:`~obspy.core.inventory.inventory.Inventory` or
            :class:`~obspy.io.xseed.parser.Parser` or str or dict
        :param metadata: Response information of instrument. See notes in
            :meth:`PPSD.__init__` for details.
        :type allow_pickle: bool
        :param allow_pickle:
            Allow the pickle protocol to be used when de-serializing saved
            PPSDs. This is only required for npz files written by ObsPy
            versions less than 1.2.0.
        :type metadata_version: str
        :param metadata_version: Non-empty version label identifying the
            version of the metadata provided with the `metadata` kwarg. Only
            relevant when the loaded npz file contains spectra archive
            information (see :meth:`PPSD.__init__`): it is used for the
            calibration record of windows that get archived when more data
            is processed with :meth:`add` afterwards. If not given,
            :meth:`add` will raise an error until a valid
            :attr:`metadata_version` is set on the object.
        """
        def _load(data):
            # the information regarding stats is set from the npz
            ppsd = PPSD(Stats(), metadata=metadata)
            # check ppsd_version version and raise if higher than current
            _check_npz_ppsd_version(ppsd, data)
            for key in ppsd.NPZ_STORE_KEYS:
                # data is stored as arrays in the npz.
                # we have to convert those back to lists (or simple types), so
                # that additionally processed data can be appended/inserted
                # later.
                try:
                    data_ = data[key]
                except ValueError:
                    msg = ("Loading PPSD results saved with ObsPy versions < "
                           "1.2 requires setting the allow_pickle parameter "
                           "of PPSD.load_npz to True")
                    raise ValueError(msg)
                if key in ppsd.NPZ_STORE_KEYS_LIST_TYPES:
                    if key in ['_times_data', '_times_gaps']:
                        data_ = data_.tolist()
                    else:
                        data_ = [d for d in data_]
                elif key in (ppsd.NPZ_STORE_KEYS_SIMPLE_TYPES +
                             ppsd.NPZ_STORE_KEYS_VERSION_NUMBERS):
                    data_ = data_.item()
                    data_ = ppsd.NPZ_SIMPLE_TYPE_MAP_R.get(data_, data_)
                # convert floating point POSIX second timestamps from older npz
                # files
                if (key in ppsd.NPZ_STORE_KEYS_LIST_TYPES and
                        data['ppsd_version'].item() == 1):
                    if key in ['_times_data', '_times_gaps']:
                        data_ = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                                 for start, end in data_]
                    elif key == '_times_processed':
                        data_ = [UTCDateTime(t)._ns for t in data_]
                setattr(ppsd, key, data_)
            # we converted all data, so update ppsd version
            ppsd.ppsd_version = PPSD._CURRENT_VERSION
            # load spectra archive information, if present in the npz file
            archive_entries = PPSD._read_archive_npz(data)
            if archive_entries is not None:
                if len(archive_entries) != len(ppsd._times_processed):
                    msg = ("Inconsistent spectra archive in npz file: %d "
                           "archive entries for %d processed time windows."
                           % (len(archive_entries),
                              len(ppsd._times_processed)))
                    raise ValueError(msg)
                ppsd._psd_archive = archive_entries
                ppsd.archive_spectra = True
            if metadata_version is not None:
                PPSD._validate_metadata_version(metadata_version)
                ppsd.metadata_version = metadata_version
            return ppsd

        try:
            with np.load(filename, allow_pickle=allow_pickle) as data:
                return _load(data)
        except ValueError as e:
            # re-raise errors that are unrelated to the pickle protocol
            # issue of very old npz files (e.g. inconsistent spectra archive
            # information)
            if "allow_pickle" not in str(e) and \
                    "Object arrays cannot be loaded" not in str(e):
                raise
            msg = ("Loading PPSD results saved with ObsPy versions < "
                   "1.2 requires setting the allow_pickle parameter "
                   "of PPSD.load_npz to True")
            raise ValueError(msg)

    def add_npz(self, filename, allow_pickle=False):
        """
        Add previously computed PPSD results to current PPSD instance.

        Load previously computed PPSD results from a
        compressed numpy binary in npz format, written with
        :meth:`obspy.signal.spectral_estimation.PPSD.save_npz` and add the
        information to the current PPSD instance.
        Before adding the data it is checked if the data was computed with the
        same settings, then any time periods that are not yet covered are added
        to the current PPSD (a warning is emitted if any segments are omitted).

        If the current PPSD instance has spectra archiving enabled (see
        :meth:`PPSD.__init__`), only npz files that also contain spectra
        archive information can be added; the archive entries of merged
        windows are kept in sync with the windows' psd data. Spectra archive
        information in files added to a PPSD without spectra archiving is
        discarded (with a warning).

        :type filename: str
        :param filename: Name of numpy .npz file(s) with stored PPSD data.
            Wildcards are possible and will be expanded using
            :py:func:`glob.glob`.
        :type allow_pickle: bool
        :param allow_pickle:
            Allow the pickle protocol to be used when de-serializing saved
            PPSDs. This is only required for npz files written by ObsPy
            versions less than 1.2.0.
        """
        for filename in glob.glob(filename):
            self._add_npz(filename, allow_pickle=allow_pickle)

    def _add_npz(self, filename, allow_pickle=False):
        """
        See :meth:`PPSD.add_npz()`.
        """
        def _add(data):
            # check ppsd_version version and raise if higher than current
            _check_npz_ppsd_version(self, data)
            # check if all metadata agree
            for key in self.NPZ_STORE_KEYS_SIMPLE_TYPES:
                value_ = data[key].item()
                value = self.NPZ_SIMPLE_TYPE_MAP_R.get(value_, value_)
                if getattr(self, key) != value:
                    msg = ("Mismatch in '%s' attribute.\n\tCurrent:\n\t%s\n\t"
                           "Loaded:\n\t%s")
                    msg = msg % (key, getattr(self, key), data[key].item())
                    raise AssertionError(msg)
            for key in self.NPZ_STORE_KEYS_ARRAY_TYPES:
                try:
                    np.testing.assert_array_equal(getattr(self, key),
                                                  data[key])
                except AssertionError as e:
                    msg = ("Mismatch in '%s' attribute.\n") % key
                    raise AssertionError(msg + str(e))
            # load new psd data
            for key in self.NPZ_STORE_KEYS_VERSION_NUMBERS:
                if getattr(self, key) != data[key].item():
                    msg = ("Mismatch in version numbers (%s) between current "
                           "data (%s) and loaded data (%s).") % (
                               key, getattr(self, key), data[key].item())
                    warnings.warn(msg)
            _times_data = data["_times_data"].tolist()
            _times_gaps = data["_times_gaps"].tolist()
            _times_processed = [d_ for d_ in data["_times_processed"]]
            _binned_psds = [d_ for d_ in data["_binned_psds"]]
            # convert floating point POSIX second timestamps from older npz
            # files
            if data['ppsd_version'].item() == 1:
                _times_data = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                               for start, end in _times_data]
                _times_gaps = [[UTCDateTime(start)._ns, UTCDateTime(end)._ns]
                               for start, end in _times_gaps]
                _times_processed = [
                    UTCDateTime(t)._ns for t in _times_processed]
            # load spectra archive information, if present in the npz file
            archive_entries = PPSD._read_archive_npz(data)
            if archive_entries is not None and \
                    len(archive_entries) != len(_times_processed):
                msg = ("Inconsistent spectra archive in npz file '%s': %d "
                       "archive entries for %d processed time windows."
                       % (filename, len(archive_entries),
                          len(_times_processed)))
                raise ValueError(msg)
            if self.archive_spectra:
                # a PPSD with spectra archiving enabled only accepts npz
                # files that also carry spectra archive information
                if archive_entries is None:
                    msg = ("Cannot add PPSD results from file '%s': the "
                           "current PPSD instance has spectra archiving "
                           "enabled, but the file does not contain spectra "
                           "archive information." % filename)
                    raise ValueError(msg)
            elif archive_entries is not None:
                msg = ("Discarding spectra archive information of file '%s' "
                       "because the current PPSD instance does not have "
                       "spectra archiving enabled." % filename)
                warnings.warn(msg)
                archive_entries = None
            # add new data
            self._times_data.extend(_times_data)
            self._times_gaps.extend(_times_gaps)
            duplicates = 0
            for i, (t, psd) in enumerate(zip(_times_processed, _binned_psds)):
                t = UTCDateTime(ns=t)
                if self.__check_time_present(t):
                    duplicates += 1
                    continue
                archive_entry = None
                if archive_entries is not None:
                    archive_entry = archive_entries[i]
                # insert the window's archive entry (if any) together with
                # its psd data, so both stay in sync
                self.__insert_processed_data(t, psd,
                                             archive_entry=archive_entry)
            # warn if some segments were omitted
            if duplicates:
                msg = ("%d/%d segments omitted in file '%s' "
                       "(time ranges already covered).")
                msg = msg % (duplicates, len(_times_processed), filename)
                warnings.warn(msg)

        try:
            with np.load(filename, allow_pickle=allow_pickle) as data:
                _add(data)
        except ValueError as e:
            # re-raise errors that are unrelated to the pickle protocol
            # issue of very old npz files (e.g. missing or inconsistent
            # spectra archive information)
            if "allow_pickle" not in str(e) and \
                    "Object arrays cannot be loaded" not in str(e):
                raise
            msg = ("Loading PPSD results saved with ObsPy versions < "
                   "1.2 requires setting the allow_pickle parameter "
                   "of PPSD.load_npz to True")
            raise ValueError(msg)

    def _split_lists(self, times, psds):
        """
        """
        t_diff_gapless = self.step * 1e9
        gap_indices = np.argwhere(np.diff(times) - t_diff_gapless)
        gap_indices = (gap_indices.flatten() + 1).tolist()

        if not len(gap_indices):
            return [(times, psds)]

        gapless = []
        indices_start = [0] + gap_indices
        indices_end = gap_indices + [len(times)]
        for start, end in zip(indices_start, indices_end):
            gapless.append((times[start:end], psds[start:end]))
        return gapless

    def _get_gapless_psd(self):
        """
        Helper routine to get a list of 2-tuples with gapless portions of
        processed PPSD time ranges.
        This means that PSD time history is split whenever to adjacent PSD
        timestamps are not separated by exactly
        ``self.ppsd_length * (1 - self.overlap)``.
        """
        return self._split_lists(self.times_processed, self.psd_values)

    def plot_spectrogram(self, cmap=obspy_sequential, clim=None, grid=True,
                         filename=None, show=True):
        """
        Plot the temporal evolution of the PSD in a spectrogram-like plot.

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Specify a custom colormap instance. If not specified, then
            the default ObsPy sequential colormap is used.
        :type clim: list
        :param clim: Minimum/maximum dB values for lower/upper end of colormap.
            Specified as type ``float`` or ``None`` for no clipping on one end
            of the scale (e.g. ``clim=[-150, None]`` for a lower limit of
            ``-150`` dB and no clipping on upper end).
        :type grid: bool
        :param grid: Enable/disable grid in histogram plot.
        :type filename: str
        :param filename: Name of output file
        :type show: bool
        :param show: Enable/disable immediately showing the plot.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        quadmeshes = []
        yedges = self.period_xedges

        for times, psds in self._get_gapless_psd():
            xedges = [t.matplotlib_date for t in times] + \
                [(times[-1] + self.step).matplotlib_date]
            meshgrid_x, meshgrid_y = np.meshgrid(xedges, yedges)
            data = np.array(psds).T

            quadmesh = ax.pcolormesh(meshgrid_x, meshgrid_y, data, cmap=cmap,
                                     zorder=-1)
            quadmeshes.append(quadmesh)

        if clim is None:
            cmin = min(qm.get_clim()[0] for qm in quadmeshes)
            cmax = max(qm.get_clim()[1] for qm in quadmeshes)
            clim = (cmin, cmax)

        for quadmesh in quadmeshes:
            quadmesh.set_clim(*clim)

        cb = plt.colorbar(quadmesh, ax=ax)

        if grid:
            ax.grid()

        if self.special_handling is None:
            cb.ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            cb.ax.set_ylabel('Amplitude [dB]')
        ax.set_ylabel('Period [s]')

        fig.autofmt_xdate()
        _set_xaxis_obspy_dates(ax)

        ax.set_yscale("log")
        ax.set_xlim(self.times_processed[0].matplotlib_date,
                    (self.times_processed[-1] + self.step).matplotlib_date)
        ax.set_ylim(yedges[0], yedges[-1])
        try:
            ax.set_facecolor('0.8')
        # mpl <2 has different API for setting Axes background color
        except AttributeError:
            ax.set_axis_bgcolor('0.8')

        fig.tight_layout()

        if filename is not None:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.draw()
            plt.show()
        else:
            plt.draw()
            return fig

    def extract_psd_values(self, period):
        """
        Extract PSD values for given period in seconds.

        Selects the period bin whose center period is closest to the specified
        period. Also returns the minimum, center and maximum period of the
        selected bin. The respective times of the PSD values can be accessed as
        :attr:`PPSD.times_processed`.

        :type period: float
        :param period: Period to extract PSD values for in seconds.
        :rtype: tuple(list, float, float, float)
        :returns: PSD values for requested period (at times)
        """
        # evaluate which period bin to extract
        period_diff = np.abs(self.period_bin_centers - period)
        index = np.argmin(period_diff)
        period_min = self.period_bin_left_edges[index]
        period_max = self.period_bin_right_edges[index]
        period_center = self.period_bin_centers[index]
        psd_values = [psd[index] for psd in self.psd_values]
        return psd_values, period_min, period_center, period_max

    def plot_temporal(self, period, color=None, legend=True, grid=True,
                      linestyle="-", marker=None, filename=None, show=True,
                      **temporal_restrictions):
        """
        Plot the evolution of PSD value of one (or more) period bins over time.

        If a filename is specified the plot is saved to this file, otherwise
        a matplotlib figure is returned or shown.

        Additional keyword arguments are passed on to :meth:`_stack_selection`
        to restrict at which times PSD values are selected (e.g. to compare
        temporal evolution during a specific time span of each day).

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type period: float or list(float)
        :param period: Period of PSD values to plot. The period bin with the
            central period that is closest to the specified value is selected.
            Multiple values can be specified in a list (``color`` option should
            then also be a list of color specifications, or left ``None``).
        :type color: valid matplotlib color or list(valid matplotlib color)
        :param color: Color specification understood by :mod:`matplotlib` (or a
            list thereof in case of multiple periods to plot). ``None`` for
            default colors.
        :type grid: bool
        :param grid: Enable/disable grid in histogram plot.
        :type legend: bool
        :param legend: Enable/disable grid in histogram plot.
        :type linestyle: str
        :param linestyle: Linestyle for lines in the plot (see
            :func:`matplotlib.pyplot.plot`).
        :type marker: str
        :param marker: Marker for lines in the plot (see
            :func:`matplotlib.pyplot.plot`).
        :type filename: str
        :param filename: Name of output file
        :type show: bool
        :param show: Enable/disable immediately showing the plot.
        """
        import matplotlib.pyplot as plt

        try:
            len(period)
        except TypeError:
            periods = [period]
        else:
            periods = period

        if color is None:
            colors = [None] * len(periods)
        else:
            if len(periods) == 1:
                colors = [color]
            else:
                colors = color

        times = self._times_processed

        if temporal_restrictions:
            mask = ~self._stack_selection(**temporal_restrictions)
            times = [x for i, x in enumerate(times) if not mask[i]]
        else:
            mask = None

        fig, ax = plt.subplots()

        for period, color in zip(periods, colors):
            cur_color = color
            # extract psd values for given period
            psd_values, period_min, _, period_max = \
                self.extract_psd_values(period)
            if mask is not None:
                psd_values = [x for i, x in enumerate(psd_values)
                              if not mask[i]]
            # if right edge of period range is less than one second we label
            # the line in Hertz
            if period_max < 1:
                label = "{:.2g}-{:.2g} [Hz]".format(
                    1.0 / period_max, 1.0 / period_min)
            else:
                label = "{:.2g}-{:.2g} [s]".format(period_min, period_max)

            for i, (times_, psd_values) in enumerate(
                    self._split_lists(times, psd_values)):
                # only label first line plotted for each period
                if i:
                    label = None
                # older matplotlib raises when passing in `color=None`
                if color is None:
                    if cur_color is None:
                        color_kwargs = {}
                    else:
                        color_kwargs = {'color': cur_color}
                else:
                    color_kwargs = {'color': color}
                times_ = [UTCDateTime(ns=t).matplotlib_date for t in times_]
                line = ax.plot(times_, psd_values, label=label, ls=linestyle,
                               marker=marker, **color_kwargs)[0]
                # plot the next lines with the same color (we can't easily
                # determine the color beforehand if we rely on the color cycle,
                # i.e. when user doesn't specify colors explictly)
                cur_color = line.get_color()

        if legend:
            ax.legend()

        if grid:
            ax.grid()

        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            ax.set_ylabel('Amplitude [dB]')

        fig.autofmt_xdate()
        _set_xaxis_obspy_dates(ax)

        if filename is not None:
            plt.savefig(filename)
            plt.close()
        elif show:
            plt.draw()
            plt.show()
        else:
            plt.draw()
            return fig

    def plot(self, filename=None, show_coverage=True, show_histogram=True,
             show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
             show_noise_models=True, grid=True, show=True,
             max_percentage=None, period_lim=(0.01, 179), show_mode=False,
             show_mean=False, cmap=obspy_sequential, cumulative=False,
             cumulative_number_of_colors=20, xaxis_frequency=False,
             show_earthquakes=None):
        """
        Plot the 2D histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        .. note::
            For example plots see the :ref:`Obspy Gallery <gallery>`.

        :type filename: str, optional
        :param filename: Name of output file
        :type show_coverage: bool, optional
        :param show_coverage: Enable/disable second axes with representation of
                data coverage time intervals.
        :type show_percentiles: bool, optional
        :param show_percentiles: Enable/disable plotting of approximated
                percentiles. These are calculated from the binned histogram and
                are not the exact percentiles.
        :type show_histogram: bool, optional
        :param show_histogram: Enable/disable plotting of histogram. This
                can be set ``False`` e.g. to make a plot with only percentiles
                plotted. Defaults to ``True``.
        :type percentiles: list[int]
        :param percentiles: percentiles to show if plotting of percentiles is
                selected.
        :type show_noise_models: bool, optional
        :param show_noise_models: Enable/disable plotting of noise models.
        :type show_earthquakes: bool, optional
        :param show_earthquakes: Enable/disable plotting of earthquake models
            like in [ClintonHeaton2002]_ and [CauzziClinton2013]_. Disabled by
            default (``None``). Specify ranges (minimum and maximum) for
            magnitude and distance of earthquake models given as four floats,
            e.g. ``(0, 5, 0, 99)`` for magnitude 1.5 - 4.5 at a epicentral
            distance of 10 km. Note only 10, 100 and 3000 km distances and
            magnitudes 1.5 to 7.5 are available. Alternatively, a distance can
            be specified in last float of a tuple of three, e.g. ``(0, 5, 10)``
            for 10 km distance, or magnitude and distance can be specified in
            a tuple of two floats, e.g. ``(5.5, 10)`` for magnitude 5.5 at 10
            km distance.
        :type grid: bool, optional
        :param grid: Enable/disable grid in histogram plot.
        :type show: bool, optional
        :param show: Enable/disable immediately showing the plot. If
            ``show=False``, then the matplotlib figure handle is returned.
        :type max_percentage: float, optional
        :param max_percentage: Maximum percentage to adjust the colormap. The
            default is 30% unless ``cumulative=True``, in which case this value
            is ignored.
        :type period_lim: tuple(float, float), optional
        :param period_lim: Period limits to show in histogram. When setting
            ``xaxis_frequency=True``, this is expected to be frequency range in
            Hz.
        :type show_mode: bool, optional
        :param show_mode: Enable/disable plotting of mode psd values.
        :type show_mean: bool, optional
        :param show_mean: Enable/disable plotting of mean psd values.
        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Colormap to use for the plot. To use the color map like in
            PQLX, [McNamara2004]_ use :class:`obspy.imaging.cm.pqlx`.
        :type cumulative: bool
        :param cumulative: Can be set to `True` to show a cumulative
            representation of the histogram, i.e. showing color coded for each
            frequency/amplitude bin at what percentage in time the value is
            not exceeded by the data (similar to the `percentile` option but
            continuously and color coded over the whole area). `max_percentage`
            is ignored when this option is specified.
        :type cumulative_number_of_colors: int
        :param cumulative_number_of_colors: Number of discrete color shades to
            use, `None` for a continuous colormap.
        :type xaxis_frequency: bool
        :param xaxis_frequency: If set to `True`, the x axis will be frequency
            in Hertz as opposed to the default of period in seconds.
        """
        import matplotlib.pyplot as plt
        self.__check_histogram()
        fig = plt.figure()
        fig.ppsd = AttribDict()

        if show_coverage:
            ax = fig.add_axes([0.12, 0.3, 0.90, 0.6])
            ax2 = fig.add_axes([0.15, 0.17, 0.7, 0.04])
        else:
            ax = fig.add_subplot(111)

        if show_percentiles:
            # for every period look up the approximate place of the percentiles
            for percentile in percentiles:
                periods, percentile_values = \
                    self.get_percentile(percentile=percentile)
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, percentile_values, color="black", zorder=8)

        if show_mode:
            periods, mode_ = self.get_mode()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mode_, color=color, zorder=9)

        if show_mean:
            periods, mean_ = self.get_mean()
            if xaxis_frequency:
                xdata = 1.0 / periods
            else:
                xdata = periods
            if cmap.name == "viridis":
                color = "0.8"
            else:
                color = "black"
            ax.plot(xdata, mean_, color=color, zorder=9)

        # Choose the correct noise model
        if self.special_handling == "infrasound":
            # Use IDC global infrasound models
            models = (get_idc_infra_hi_noise(), get_idc_infra_low_noise())
        else:
            # Use Peterson NHNM and NLNM
            models = (get_nhnm(), get_nlnm())

        if show_noise_models:
            for periods, noise_model in models:
                if xaxis_frequency:
                    xdata = 1.0 / periods
                else:
                    xdata = periods
                ax.plot(xdata, noise_model, '0.4', linewidth=2, zorder=10)

        if show_earthquakes is not None:
            if len(show_earthquakes) == 2:
                show_earthquakes = (show_earthquakes[0],
                                    show_earthquakes[0] + 0.1,
                                    show_earthquakes[1],
                                    show_earthquakes[1] + 1)
            if len(show_earthquakes) == 3:
                show_earthquakes += (show_earthquakes[-1] + 1, )
            min_mag, max_mag, min_dist, max_dist = show_earthquakes
            for key, data in earthquake_models.items():
                magnitude, distance = key
                frequencies, accelerations = data
                accelerations = np.array(accelerations)
                frequencies = np.array(frequencies)
                periods = 1.0 / frequencies
                # Eq.1 from Clinton and Cauzzi (2013) converts
                # power to density
                ydata = accelerations / (periods ** (-.5))
                ydata = 20 * np.log10(ydata / 2)
                if not (min_mag <= magnitude <= max_mag and
                        min_dist <= distance <= max_dist and
                        min(ydata) < self.db_bin_edges[-1]):
                    continue
                xdata = periods
                if xaxis_frequency:
                    xdata = frequencies
                ax.plot(xdata, ydata, '0.4', linewidth=2)
                leftpoint = np.argsort(xdata)[0]
                if not ydata[leftpoint] < self.db_bin_edges[-1]:
                    continue
                ax.text(xdata[leftpoint],
                        ydata[leftpoint],
                        'M%.1f\n%dkm' % (magnitude, distance),
                        ha='right', va='top',
                        color='w', weight='bold', fontsize='x-small',
                        path_effects=[withStroke(linewidth=3,
                                                 foreground='0.4')])

        if show_histogram:
            label = "[%]"
            if cumulative:
                label = "non-exceedance (cumulative) [%]"
                if max_percentage is not None:
                    msg = ("Parameter 'max_percentage' is ignored when "
                           "'cumulative=True'.")
                    warnings.warn(msg)
                max_percentage = 100
                if cumulative_number_of_colors is not None:
                    cmap = LinearSegmentedColormap(
                        name=cmap.name, segmentdata=cmap._segmentdata,
                        N=cumulative_number_of_colors)
            elif max_percentage is None:
                # Set default only if cumulative is not True.
                max_percentage = 30

            fig.ppsd.cumulative = cumulative
            fig.ppsd.cmap = cmap
            fig.ppsd.label = label
            fig.ppsd.max_percentage = max_percentage
            fig.ppsd.grid = grid
            fig.ppsd.xaxis_frequency = xaxis_frequency
            if max_percentage is not None:
                color_limits = (0, max_percentage)
                fig.ppsd.color_limits = color_limits

            self._plot_histogram(fig=fig)

        if xaxis_frequency:
            ax.set_xlabel('Frequency [Hz]')
            ax.invert_xaxis()
        else:
            ax.set_xlabel('Period [s]')
        ax.set_xscale('log')
        ax.set_xlim(period_lim)
        ax.set_ylim(self.db_bin_edges[0], self.db_bin_edges[-1])
        if self.special_handling is None:
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
        elif self.special_handling == "infrasound":
            ax.set_ylabel('Amplitude [$Pa^2/Hz$] [dB]')
        else:
            ax.set_ylabel('Amplitude [dB]')
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_title(self._get_plot_title())

        if show_coverage:
            self.__plot_coverage(ax2)
            # emulating fig.autofmt_xdate():
            for label in ax2.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(30)

        # Catch underflow warnings due to plotting on log-scale.
        with np.errstate(all="ignore"):
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            elif show:
                plt.draw()
                plt.show()
            else:
                plt.draw()
                return fig

    def _plot_histogram(self, fig, draw=False, filename=None):
        """
        Reuse a previously created figure returned by `plot(show=False)`
        and plot the current histogram stack (pre-computed using
        :meth:`calculate_histogram()`) into the figure. If a filename is
        provided, the figure will be saved to a local file.
        Note that many aspects of the plot are statically set during the first
        :meth:`plot()` call, so this routine can only be used to update with
        data from a new stack.
        """
        import matplotlib.pyplot as plt
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        if "quadmesh" in fig.ppsd:
            fig.ppsd.pop("quadmesh").remove()

        if fig.ppsd.cumulative:
            data = self.current_histogram_cumulative * 100.0
        else:
            # avoid divison with zero in case of empty stack
            data = (
                self.current_histogram * 100.0 /
                (self.current_histogram_count or 1))

        xedges = self.period_xedges
        if fig.ppsd.xaxis_frequency:
            xedges = 1.0 / xedges

        if "meshgrid" not in fig.ppsd:
            fig.ppsd.meshgrid = np.meshgrid(xedges, self.db_bin_edges)
        ppsd = ax.pcolormesh(
            fig.ppsd.meshgrid[0], fig.ppsd.meshgrid[1], data.T,
            cmap=fig.ppsd.cmap, zorder=-1)
        fig.ppsd.quadmesh = ppsd

        if "colorbar" not in fig.ppsd:
            cb = plt.colorbar(ppsd, ax=ax)
            cb.mappable.set_clim(*fig.ppsd.color_limits)
            cb.set_label(fig.ppsd.label)
            fig.ppsd.colorbar = cb

        if fig.ppsd.max_percentage is not None:
            ppsd.set_clim(*fig.ppsd.color_limits)

        if fig.ppsd.grid:
            if fig.ppsd.cmap.name == "viridis":
                color = {"color": "0.7"}
            else:
                color = {}
            ax.grid(True, which="major", **color)
            ax.grid(True, which="minor", **color)

        ax.set_xlim(*xlim)

        if filename is not None:
            plt.savefig(filename)
        elif draw:
            with np.errstate(under="ignore"):
                plt.draw()
        return fig

    def _get_plot_title(self):
        title = "%s   %s -- %s  (%i/%i segments)"
        title = title % (self.id,
                         UTCDateTime(ns=self._times_processed[0]).date,
                         UTCDateTime(ns=self._times_processed[-1]).date,
                         self.current_histogram_count,
                         len(self._times_processed))
        return title

    def plot_coverage(self, filename=None):
        """
        Plot the data coverage of the histogram of the current PPSD.
        If a filename is specified the plot is saved to this file, otherwise
        a plot window is shown.

        :type filename: str, optional
        :param filename: Name of output file
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        self.__plot_coverage(ax)
        fig.autofmt_xdate()
        ax.set_title(self._get_plot_title())

        plt.draw()
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def __plot_coverage(self, ax):
        """
        Helper function to plot coverage into given axes.
        """
        ax.figure
        ax.clear()
        ax.xaxis_date()
        ax.set_yticks([])

        # plot data used in histogram stack
        used_times = [UTCDateTime(ns=t) for t in self._times_processed
                      if t in self._current_times_used]
        unused_times = [UTCDateTime(ns=t) for t in self._times_processed
                        if t not in self._current_times_used]
        for times, color in zip((used_times, unused_times), ("b", "0.6")):
            # skip on empty lists (i.e. all data used, or none used in stack)
            if not times:
                continue
            starts = [t.matplotlib_date for t in times]
            ends = [(t + self.ppsd_length).matplotlib_date for t in times]
            startends = np.array([starts, ends])
            startends = compress_start_end(startends.T, 20,
                                           merge_overlaps=True)
            starts, ends = startends[:, 0], startends[:, 1]
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, 0, 0.6, fc=color, lw=0)
        # plot data that was fed to PPSD
        for start, end in self.times_data:
            start = start.matplotlib_date
            end = end.matplotlib_date
            ax.axvspan(start, end, 0.6, 1, facecolor="g", lw=0)
        # plot gaps in data fed to PPSD
        for start, end in self.times_gaps:
            start = start.matplotlib_date
            end = end.matplotlib_date
            ax.axvspan(start, end, 0.6, 1, facecolor="r", lw=0)

        ax.autoscale_view()


def get_nlnm():
    """
    Returns periods and psd values for the New Low Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


def get_nhnm():
    """
    Returns periods and psd values for the New High Noise Model.
    For information on New High/Low Noise Model see [Peterson1993]_.
    """
    data = np.load(NOISE_MODEL_FILE)
    periods = data['model_periods']
    nlnm = data['high_noise']
    return (periods, nlnm)


def get_idc_infra_low_noise():
    """
    Returns periods and psd values for the IDC infrasound global low noise
    model. For information on the IDC noise models, see [Brown2012]_.
    """
    data = np.load(NOISE_MODEL_FILE_INF)
    periods = data['model_periods']
    nlnm = data['low_noise']
    return (periods, nlnm)


def get_idc_infra_hi_noise():
    """
    Returns periods and psd values for the IDC infrasound global high noise
    model. For information on the IDC noise models, see [Brown2012]_.
    """
    data = np.load(NOISE_MODEL_FILE_INF)
    periods = data['model_periods']
    nhnm = data['high_noise']
    return (periods, nhnm)


def _check_npz_ppsd_version(ppsd, npzfile):
    # add some future-proofing and show a warning if older ObsPy
    # versions should read a more recent ppsd npz file, since this is very
    # like problematic
    if npzfile['ppsd_version'].item() > ppsd.ppsd_version:
        msg = ("Trying to read/add a PPSD npz with 'ppsd_version={}'. This "
               "file was written on a more recent ObsPy version that very "
               "likely has incompatible changes in PPSD internal "
               "structure and npz serialization. It can not safely be "
               "read with this ObsPy version (current 'ppsd_version' is "
               "{}). Please consider updating your ObsPy "
               "installation.").format(npzfile['ppsd_version'].item(),
                                       ppsd.ppsd_version)
        raise ObsPyException(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
