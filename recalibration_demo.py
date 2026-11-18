#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo/verification script for the PPSD spectra archiving and recalibration
feature (obspy 1.4.2 codebase).

Scenario: waveforms were processed into a PPSD with spectra archiving
enabled and then cleaned up. Later, a corrected StationXML file arrives and
only the affected spectral windows are recomputed from the archive, keeping
track of which metadata version was used for every window.

Rerun (from the repository root, with obspy importable):

    python3 recalibration_demo.py

The script writes two StationXML files (original "v1" and corrected "v2",
with genuinely different frequency responses) into a temporary directory and
reads them back with read_inventory(), simulating the arrival of corrected
metadata. All checks print PASS/FAIL and the script exits non-zero if any
check fails.
"""
import os
import sys
import tempfile

import numpy as np

from obspy import Trace, UTCDateTime, read_inventory
from obspy.core.inventory import (Inventory, Network, Station, Channel,
                                  Response, InstrumentSensitivity,
                                  NoMatchingChannelEpochError,
                                  MultipleMatchingChannelEpochsError)
from obspy.core.inventory.response import (PolesZerosResponseStage,
                                           ResponseStage)
from obspy.signal.spectral_estimation import PPSD, PPSDRecalibrationError

# two genuinely different instrument responses (different poles, different
# gain and sensitivity)
POLES_V1 = [(-0.037004 + 0.037016j), (-0.037004 - 0.037016j), (-251.33 + 0j),
            (-131.04 - 467.29j), (-131.04 + 467.29j)]
GAIN_V1 = 60077000.0
SENS_V1 = 2516778400.0
POLES_V2 = [(-0.08 + 0.06j), (-0.08 - 0.06j), (-180.0 + 0j),
            (-90.0 - 350.0j), (-90.0 + 350.0j)]
GAIN_V2 = 900000.0
SENS_V2 = 1500000000.0

RESULTS = []


def check(name, ok, details=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append(ok)
    line = "  [%s] %s" % (status, name)
    if details:
        line += " -- %s" % details
    print(line)


def make_inventory(poles, gain, sensitivity, location="",
                   start_date=None, end_date=None):
    """Build a one-channel inventory with a two-stage (PAZ + gain) response."""
    paz_stage = PolesZerosResponseStage(
        stage_sequence_number=1, stage_gain=gain, stage_gain_frequency=1.0,
        input_units="M/S", output_units="V",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0, zeros=[0j, 0j], poles=poles,
        normalization_factor=1.0)
    gain_stage = ResponseStage(
        stage_sequence_number=2, stage_gain=sensitivity / gain,
        stage_gain_frequency=1.0, input_units="V", output_units="COUNTS")
    response = Response(
        resource_id="DEMO_RESPONSE",
        instrument_sensitivity=InstrumentSensitivity(
            sensitivity, 1.0, "M/S", "COUNTS"),
        response_stages=[paz_stage, gain_stage])
    channel = Channel(code="BHZ", location_code=location, latitude=0.0,
                      longitude=0.0, elevation=0.0, depth=0.0,
                      response=response, start_date=start_date,
                      end_date=end_date, sample_rate=20.0)
    station = Station(code="DEMO", latitude=0.0, longitude=0.0, elevation=0.0,
                      channels=[channel])
    return Inventory(networks=[Network(code="XX", stations=[station])],
                     source="recalibration_demo")


def make_trace(starttime=UTCDateTime("2011-03-31T00:00:00"),
               sampling_rate=20.0, duration=3600.0, seed=42, location=""):
    """Synthetic noise trace with psd values inside the default dB binning."""
    rng = np.random.RandomState(seed)
    data = rng.randn(int(duration * sampling_rate)) * 1e-5
    tr = Trace(data=data)
    tr.stats.network = "XX"
    tr.stats.station = "DEMO"
    tr.stats.location = location
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = starttime
    return tr


def max_finite_diff(values_a, values_b):
    """Maximum absolute difference of finite entries of two psd arrays."""
    diff = np.abs(np.asarray(values_a, dtype=np.float64) -
                  np.asarray(values_b, dtype=np.float64))
    finite = np.isfinite(diff)
    if not finite.any():
        return np.nan
    return diff[finite].max()


def scenario_main(workdir):
    print("=" * 78)
    print("Scenario 1: cache histogram, recalibrate only the middle windows")
    print("=" * 78)
    # StationXML v1 (original) and v2 (corrected), written to disk and read
    # back, like in the real workflow
    xml_v1 = os.path.join(workdir, "XX.DEMO.v1.xml")
    xml_v2 = os.path.join(workdir, "XX.DEMO.v2.xml")
    make_inventory(POLES_V1, GAIN_V1, SENS_V1,
                   start_date=UTCDateTime("2010-01-01")).write(
                       xml_v1, format="STATIONXML")
    make_inventory(POLES_V2, GAIN_V2, SENS_V2,
                   start_date=UTCDateTime("2010-01-01")).write(
                       xml_v2, format="STATIONXML")
    inv_v1 = read_inventory(xml_v1)
    inv_v2 = read_inventory(xml_v2)

    # show that the two StationXML files genuinely differ in frequency
    # response (amplitude of the VEL response at a few frequencies)
    print("  amplitude response |H(f)| (VEL), v1 vs v2:")
    t_resp = UTCDateTime("2011-03-31T00:00:00")
    freqs = [0.02, 0.05, 0.1, 0.2, 1.0, 5.0]
    r1 = inv_v1.get_response("XX.DEMO..BHZ", t_resp)
    r2 = inv_v2.get_response("XX.DEMO..BHZ", t_resp)
    a1 = np.abs(r1.get_evalresp_response_for_frequencies(
        freqs, output="VEL"))
    a2 = np.abs(r2.get_evalresp_response_for_frequencies(
        freqs, output="VEL"))
    for f, v1, v2 in zip(freqs, a1, a2):
        print("    f=%5.2f Hz: v1=%.4e  v2=%.4e  (%+.2f dB)"
              % (f, v1, v2, 20 * np.log10(v2 / v1)))
    check("v1/v2 StationXML genuinely differ in frequency response",
          np.abs(20 * np.log10(a2 / a1)).max() > 1.0,
          "max difference %.1f dB" % np.abs(20 * np.log10(a2 / a1)).max())

    tr = make_trace()
    ppsd = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5,
                archive_spectra=True, metadata_version="v1")
    ppsd.add(tr)
    n_windows = len(ppsd.times_processed)
    print("  processed %d windows (ppsd_length=600 s, overlap=0.5)"
          % n_windows)

    # cache the histogram before recalibration
    ppsd.calculate_histogram()
    hist_before = ppsd.current_histogram.copy()
    values_before = [np.array(v) for v in ppsd.psd_values]

    # recalibrate only the middle windows [t2, t5) with the corrected
    # StationXML
    times = ppsd.times_processed
    t_start, t_end = times[2], times[5]
    updated = ppsd.recalibrate(inv_v2, "v2", t_start, t_end)
    check("recalibrate returns the selected window start times",
          updated == times[2:5], "%d windows" % len(updated))

    # reference: same waveforms, same settings, computed only with the
    # corrected response
    ppsd_ref = PPSD(tr.stats, inv_v2, ppsd_length=600.0, overlap=0.5)
    ppsd_ref.add(tr)
    ref = {t._ns: v for t, v in zip(ppsd_ref.times_processed,
                                    ppsd_ref.psd_values)}
    maxdiff = max(max_finite_diff(v, ref[t._ns])
                  for t, v in zip(ppsd.times_processed, ppsd.psd_values)
                  if t_start <= t < t_end)
    check("recalibrated windows match standalone computation (<= 0.0001 dB)",
          maxdiff <= 1e-4, "max abs diff = %.3g dB" % maxdiff)

    # windows outside the interval are untouched (values and provenance)
    outside_ok = all(
        np.array_equal(np.asarray(v), values_before[i])
        for i, (t, v) in enumerate(zip(ppsd.times_processed,
                                       ppsd.psd_values))
        if not (t_start <= t < t_end))
    infos = ppsd.get_archive_info()
    provenance_ok = all(info['version'] == ("v2" if t_start <= t < t_end
                                            else "v1")
                        for t, info in zip(ppsd.times_processed, infos))
    check("windows outside the interval unchanged (values)", outside_ok)
    check("provenance updated inside / kept outside the interval",
          provenance_ok,
          "versions: %s" % [i['version'] for i in infos])
    check("per-window calibration record complete",
          all(info['channel_id'] == "XX.DEMO..BHZ" and
              info['lookup_time'] == t and
              info['epoch_start'] == UTCDateTime("2010-01-01") and
              info['epoch_end'] is None
              for t, info in zip(ppsd.times_processed, infos)))

    # histogram is recounted from the updated values
    hist_after = ppsd.current_histogram
    check("histogram recounted after recalibration",
          not np.array_equal(hist_before, hist_after),
          "changed bin counts: %d" % np.abs(
              hist_before.astype(np.int64)
              - hist_after.astype(np.int64)).sum())

    # window identity, order, coverage and gap statistics unchanged
    check("window times/order and coverage/gap stats unchanged",
          ppsd.times_processed == times and len(ppsd.times_gaps) == 0)

    # next add() still uses the object's original metadata
    tr_later = make_trace(starttime=tr.stats.starttime + 3600, seed=43)
    ppsd.add(tr_later)
    new_versions = {ppsd.get_archive_info(t)['version']
                    for t in ppsd.times_processed
                    if t >= tr_later.stats.starttime}
    check("next add() still uses original metadata (version 'v1')",
          new_versions == {"v1"})
    return ppsd, inv_v1, inv_v2


def scenario_lookup_time_offset():
    print("=" * 78)
    print("Scenario 2: archived window start differs from actual lookup time")
    print("=" * 78)
    # step of 50.5 s with 1 Hz sampling: every other window start falls
    # between two samples, the sliced trace start (= response lookup time)
    # differs from the archived window start
    tr = make_trace(sampling_rate=1.0, duration=400.0)
    inv_v1 = make_inventory(POLES_V1, GAIN_V1, SENS_V1,
                            start_date=UTCDateTime("2010-01-01"))
    ppsd = PPSD(tr.stats, inv_v1, ppsd_length=101.0, overlap=0.5,
                archive_spectra=True, metadata_version="v1")
    ppsd.add(tr)
    times = ppsd.times_processed
    infos = ppsd.get_archive_info()
    i_off = next(i for i, (t, info) in enumerate(zip(times, infos))
                 if info['lookup_time'] != t)
    t_off, lookup = times[i_off], infos[i_off]['lookup_time']
    print("  window start %s, actual lookup time %s" % (t_off, lookup))
    check("found window with lookup time offset from window start",
          t_off != lookup)

    # corrected metadata with an epoch boundary between window start and
    # lookup time
    boundary = t_off + (lookup - t_off) / 2.0
    inv_old = make_inventory(POLES_V1, GAIN_V1, SENS_V1,
                             start_date=UTCDateTime("2010-01-01"),
                             end_date=boundary)
    inv_new = make_inventory(POLES_V2, GAIN_V2, SENS_V2, start_date=boundary)
    inv_v2 = inv_old + inv_new
    epoch_at_start = inv_v2.get_channel_epoch(ppsd.id, t_off)
    epoch_at_lookup = inv_v2.get_channel_epoch(ppsd.id, lookup)
    check("epoch at window start differs from epoch at lookup time",
          epoch_at_start is not epoch_at_lookup)

    ppsd.recalibrate(inv_v2, "v2", t_off, t_off + 1)
    info = ppsd.get_archive_info(t_off)
    check("recalibration used epoch valid at the original lookup time",
          info['epoch_start'] == epoch_at_lookup.start_date and
          info['epoch_end'] == epoch_at_lookup.end_date)
    ppsd_ref = PPSD(tr.stats, inv_v2, ppsd_length=101.0, overlap=0.5)
    ppsd_ref.add(tr)
    ref = {t._ns: v for t, v in zip(ppsd_ref.times_processed,
                                    ppsd_ref.psd_values)}
    diff = max_finite_diff(ppsd.psd_values[i_off], ref[t_off._ns])
    check("values match standalone computation (<= 0.0001 dB)",
          diff <= 1e-4, "max abs diff = %.3g dB" % diff)


def scenario_failures():
    print("=" * 78)
    print("Scenario 3: epoch gap / epoch overlap / invalid response -> atomic "
          "failure")
    print("=" * 78)
    tr = make_trace()
    inv_v1 = make_inventory(POLES_V1, GAIN_V1, SENS_V1,
                            start_date=UTCDateTime("2010-01-01"))
    ppsd = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5,
                archive_spectra=True, metadata_version="v1")
    ppsd.add(tr)
    ppsd.calculate_histogram()
    values_before = [np.array(v) for v in ppsd.psd_values]
    infos_before = ppsd.get_archive_info()
    hist_before = ppsd.current_histogram.copy()
    times = ppsd.times_processed

    def state_unchanged():
        return (all(np.array_equal(np.asarray(v), e)
                    for v, e in zip(ppsd.psd_values, values_before)) and
                ppsd.get_archive_info() == infos_before and
                np.array_equal(ppsd.current_histogram, hist_before))

    # (a) epoch gap in the corrected metadata
    inv_gap = make_inventory(POLES_V2, GAIN_V2, SENS_V2,
                             start_date=times[6], end_date=times[8])
    try:
        ppsd.recalibrate(inv_gap, "v2", times[2], times[5])
        gap_raised = False
    except NoMatchingChannelEpochError:
        gap_raised = True
    check("epoch gap -> NoMatchingChannelEpochError", gap_raised)
    check("state preserved after failed recalibration (gap)",
          state_unchanged())

    # (b) overlapping epochs in the corrected metadata
    inv_overlap = make_inventory(POLES_V2, GAIN_V2, SENS_V2,
                                 start_date=UTCDateTime("2010-01-01"))
    import copy
    inv_overlap[0][0].channels.append(
        copy.deepcopy(inv_overlap[0][0].channels[0]))
    try:
        ppsd.recalibrate(inv_overlap, "v2", times[2], times[5])
        overlap_raised = False
    except MultipleMatchingChannelEpochsError:
        overlap_raised = True
    check("overlapping epochs -> MultipleMatchingChannelEpochsError",
          overlap_raised)
    check("state preserved after failed recalibration (overlap)",
          state_unchanged())

    # (c) broken response (zero sensitivity) -> no valid spectrum
    # (note: evaluating the broken response makes the evalresp C library
    # print an "EVRESP ERROR ... zero stage gain" message to stderr, which
    # is expected here)
    inv_broken = make_inventory(POLES_V2, 1.0, 0.0,
                                start_date=UTCDateTime("2010-01-01"))
    try:
        ppsd.recalibrate(inv_broken, "v2", times[2], times[5])
        broken_raised = False
    except PPSDRecalibrationError:
        broken_raised = True
    check("invalid spectrum -> PPSDRecalibrationError", broken_raised)
    check("state preserved after failed recalibration (invalid spectrum)",
          state_unchanged())

    # (d) strict lookup entry on Inventory, directly callable
    try:
        inv_gap.get_channel_epoch("XX.DEMO..BHZ", times[0])
        direct_raised = False
    except NoMatchingChannelEpochError:
        direct_raised = True
    check("Inventory.get_channel_epoch() usable standalone", direct_raised)


def scenario_location():
    print("=" * 78)
    print("Scenario 4: same channel code at different locations")
    print("=" * 78)
    tr = make_trace(location="00")
    inv_v1 = make_inventory(POLES_V1, GAIN_V1, SENS_V1, location="00",
                            start_date=UTCDateTime("2010-01-01"))
    ppsd = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5,
                archive_spectra=True, metadata_version="v1")
    ppsd.add(tr)
    inv_00 = make_inventory(POLES_V2, GAIN_V2, SENS_V2, location="00",
                            start_date=UTCDateTime("2010-01-01"))
    inv_10 = make_inventory(POLES_V1, 1.0, 1.0, location="10",
                            start_date=UTCDateTime("2010-01-01"))
    inv_v2 = inv_00 + inv_10
    times = ppsd.times_processed
    ppsd.recalibrate(inv_v2, "v2", times[0], times[1])
    info = ppsd.get_archive_info(times[0])
    ppsd_ref = PPSD(tr.stats, inv_00, ppsd_length=600.0, overlap=0.5)
    ppsd_ref.add(tr)
    diff = max_finite_diff(ppsd.psd_values[0], ppsd_ref.psd_values[0])
    check("recalibration picked the '00' epoch (channel id kept)",
          info['channel_id'] == "XX.DEMO.00.BHZ")
    check("values match standalone computation with '00' response",
          diff <= 1e-4, "max abs diff = %.3g dB" % diff)


def scenario_repeated_and_npz(workdir, inv_v1, inv_v2):
    print("=" * 78)
    print("Scenario 5: repeated recalibration, save/reload, merge, old npz")
    print("=" * 78)
    tr = make_trace()
    ppsd = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5,
                archive_spectra=True, metadata_version="v1")
    ppsd.add(tr)
    times = ppsd.times_processed
    t_start, t_end = times[2], times[5]

    # repeated recalibration
    ppsd.recalibrate(inv_v2, "v2", t_start, t_end)
    values_first = [np.array(v) for v in ppsd.psd_values]
    ppsd.recalibrate(inv_v2, "v2", t_start, t_end)
    check("repeated recalibration is idempotent",
          all(np.array_equal(np.asarray(v), e)
              for v, e in zip(ppsd.psd_values, values_first)))

    # save / reload
    npz_archive = os.path.join(workdir, "ppsd_archive.npz")
    ppsd.save_npz(npz_archive)
    ppsd_loaded = PPSD.load_npz(npz_archive, metadata=inv_v2,
                                metadata_version="v2")
    check("archive survives save/load (values + provenance)",
          ppsd_loaded.archive_spectra and
          ppsd_loaded.get_archive_info() == ppsd.get_archive_info() and
          all(np.array_equal(np.asarray(a), np.asarray(b))
              for a, b in zip(ppsd_loaded.psd_values, ppsd.psd_values)))
    # recalibrate after reload
    ppsd_loaded.recalibrate(inv_v1, "v1.1", times[0], times[1])
    check("recalibration works after npz reload",
          ppsd_loaded.get_archive_info(times[0])['version'] == "v1.1")

    # merge two archive npz files
    tr2 = make_trace(starttime=tr.stats.starttime + 3600, seed=43)
    ppsd2 = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5,
                 archive_spectra=True, metadata_version="v1")
    ppsd2.add(tr2)
    npz_archive2 = os.path.join(workdir, "ppsd_archive2.npz")
    ppsd2.save_npz(npz_archive2)
    n1 = len(ppsd.times_processed)
    ppsd.add_npz(npz_archive2)
    infos = ppsd.get_archive_info()
    check("archive npz merge keeps windows and calibration records in sync",
          len(ppsd.times_processed) == n1 + len(ppsd2.times_processed) and
          len(infos) == len(ppsd.times_processed) and
          infos[-1]['version'] == "v1")

    # archive PPSD refuses a plain npz
    ppsd_plain = PPSD(tr.stats, inv_v1, ppsd_length=600.0, overlap=0.5)
    ppsd_plain.add(tr2)
    npz_plain = os.path.join(workdir, "ppsd_plain.npz")
    ppsd_plain.save_npz(npz_plain)
    try:
        ppsd.add_npz(npz_plain)
        refused = False
    except ValueError:
        refused = True
    check("archive PPSD refuses npz without archive information", refused)

    # old npz (plain format, no archive keys) loads as before
    ppsd_old = PPSD.load_npz(npz_plain, metadata=inv_v1)
    ppsd_old.calculate_histogram()
    try:
        ppsd_old.recalibrate(inv_v2, "v2", times[0], times[1])
        old_raised = False
    except PPSDRecalibrationError:
        old_raised = True
    check("old npz loads and plots as before (archiving off)",
          not ppsd_old.archive_spectra)
    check("recalibration of old npz raises a clear error", old_raised)

    # hand-crafted minimal npz with only the ppsd_version=3 keys (as written
    # by older ObsPy versions) loads fine as well
    npz_v3 = os.path.join(workdir, "ppsd_v3_only.npz")
    out = {}
    for key in PPSD.NPZ_STORE_KEYS:
        out[key] = np.load(npz_plain)[key]
    np.savez_compressed(npz_v3, **out)
    ppsd_v3 = PPSD.load_npz(npz_v3, metadata=inv_v1)
    check("npz with only ppsd_version=3 keys loads fine",
          len(ppsd_v3.times_processed) == len(ppsd_plain.times_processed))


def scenario_size(workdir, inv_v1):
    print("=" * 78)
    print("Scenario 6: archive size comparison (100 Hz, 1 h windows, 12 h)")
    print("=" * 78)
    tr = make_trace(sampling_rate=100.0, duration=12 * 3600.0, seed=44)
    kwargs = dict(ppsd_length=3600.0, overlap=0.5)
    ppsd_plain = PPSD(tr.stats, inv_v1, **kwargs)
    ppsd_plain.add(tr)
    ppsd_arch = PPSD(tr.stats, inv_v1, archive_spectra=True,
                     metadata_version="v1", **kwargs)
    ppsd_arch.add(tr)
    npz_plain = os.path.join(workdir, "size_plain.npz")
    npz_arch = os.path.join(workdir, "size_archive.npz")
    ppsd_plain.save_npz(npz_plain)
    ppsd_arch.save_npz(npz_arch)
    size_plain = os.path.getsize(npz_plain)
    size_arch = os.path.getsize(npz_arch)
    n_windows = len(ppsd_arch.times_processed)
    n_freqs = len(ppsd_arch.psd_periods)
    raw = n_windows * n_freqs * 8
    print("  windows: %d, spectral points per window (nfft/2): %d"
          % (n_windows, n_freqs))
    print("  raw archive payload: %d windows x %d freq x 8 byte = %.2f MB"
          % (n_windows, n_freqs, raw / 1e6))
    print("  plain npz:   %10d bytes (%.3f MB)" % (size_plain,
                                                   size_plain / 1e6))
    print("  archive npz: %10d bytes (%.3f MB)" % (size_arch,
                                                   size_arch / 1e6))
    print("  ratio: %.1fx" % (size_arch / size_plain))
    check("archive npz holds raw spectra (no time-domain samples)",
          len(ppsd_arch._psd_archive) == n_windows and
          ppsd_arch._psd_archive[0]['spectrum'].shape == (n_freqs,))
    check("plain npz is much smaller than archive npz", size_plain < size_arch)


def main():
    print("Python %s" % sys.version.split()[0])
    import obspy
    print("ObsPy %s" % obspy.__version__)
    print("rerun: python3 recalibration_demo.py   (from the repository "
          "root, with obspy importable)")
    with tempfile.TemporaryDirectory() as workdir:
        ppsd, inv_v1, inv_v2 = scenario_main(workdir)
        scenario_lookup_time_offset()
        scenario_failures()
        scenario_location()
        scenario_repeated_and_npz(workdir, inv_v1, inv_v2)
        scenario_size(workdir, inv_v1)
    print("=" * 78)
    n_ok = sum(RESULTS)
    print("SUMMARY: %d/%d checks passed" % (n_ok, len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
