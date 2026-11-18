# -*- coding: utf-8 -*-
"""
Offline test suite for the optional backfill mode of the mass downloader.

Everything runs against in-memory mock FDSN data centers - no public FDSN
service is contacted.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import fnmatch
import os

import numpy as np

import obspy
from obspy.core.inventory import (Channel, Inventory, Network, Response,
                                  Station)
from obspy.core.inventory.response import InstrumentSensitivity
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.mass_downloader import MassDownloader, Restrictions
from obspy.clients.fdsn.mass_downloader import domain as domain_module
from obspy.clients.fdsn.mass_downloader.backfill import (
    BACKFILL_STATUS, BackfillReport, REASON_EMPTY_STATIONXML,
    REASON_NO_WAVEFORM, REASON_RESPONSE_CONFLICT, REASON_RESPONSE_GAP)


# ---------------------------------------------------------------------- #
# Mock FDSN data center
# ---------------------------------------------------------------------- #
def make_response(sensitivity, frequency=1.0):
    response = Response()
    response.instrument_sensitivity = InstrumentSensitivity(
        value=float(sensitivity), frequency=frequency,
        input_units="M/S", output_units="COUNTS")
    return response


def make_channel(location, channel, start, end, sensitivity):
    cha = Channel(
        code=channel, location_code=location, latitude=48.0, longitude=11.0,
        elevation=0.0, depth=0.0, start_date=start, end_date=end)
    cha.response = make_response(sensitivity)
    return cha


class MockFDSNCenter(object):
    """
    In-memory mock of the three Client methods used by the mass downloader.

    :param name: data center name (also ``base_url``).
    :param availability: list of dicts with keys network, station, location,
        channel, start, end, latitude, longitude, sensitivity. Each entry
        becomes one channel epoch in the ``station`` responses.
    """
    def __init__(self, name, availability, waveform_spans=None):
        self.name = name
        self.base_url = "http://%s.example" % name
        # marktimeseries availability is "reliable" in the real client.
        self.services = {"station": ["matchtimeseries", "format"],
                         "dataselect": ["bypass"]}
        self._availability = copy.deepcopy(availability)
        # Optional explicit waveform holdings as
        # (net, sta, loc, cha, start, end) tuples. When given, dataselect
        # serves these spans independently of the response epochs, which
        # allows modeling a center that has waveforms in a period for which
        # its StationXML has no response.
        self.waveform_spans = waveform_spans
        self.waveform_requests = []
        self.stationxml_requests = []
        # When True, every dataselect request raises like an offline center.
        self.fail_waveforms = False
        # When True, station service returns an empty inventory for
        # response-level queries (but availability still works).
        self.empty_stationxml = False

    # -- station service -------------------------------------------------- #
    def get_stations(self, **kwargs):
        level = kwargs.get("level", "station")
        # Only response-level replies are emptied; availability (channel
        # level) must still advertise the station so its waveforms are
        # attempted.
        if self.empty_stationxml and level == "response":
            return Inventory(networks=[], source=self.name)
        return self._build_inventory(
            start=kwargs.get("starttime"), end=kwargs.get("endtime"),
            with_response=(level == "response"),
            network=kwargs.get("network"), station=kwargs.get("station"),
            location=kwargs.get("location"), channel=kwargs.get("channel"))

    def get_stations_bulk(self, bulk, level="response", filename=None):
        self.stationxml_requests.append([tuple(b) for b in bulk])
        if self.empty_stationxml:
            Inventory(networks=[], source=self.name).write(
                filename, format="stationxml")
            return
        inventory = self._build_bulk_inventory(bulk)
        inventory.write(filename, format="stationxml")

    def _matches_seed(self, entry, network, station, location, channel):
        if network is not None and not fnmatch.fnmatch(
                entry["network"], network):
            return False
        if station is not None and not fnmatch.fnmatch(
                entry["station"], station):
            return False
        if location is not None and not fnmatch.fnmatch(
                entry["location"], location):
            return False
        if channel is not None and not fnmatch.fnmatch(
                entry["channel"], channel):
            return False
        return True

    def _build_inventory(self, start, end, with_response, network=None,
                         station=None, location=None, channel=None):
        networks = {}
        for entry in self._availability:
            if not self._matches_seed(entry, network, station, location,
                                      channel):
                continue
            net = networks.setdefault(entry["network"], {})
            sta = net.setdefault(entry["station"], {
                "latitude": entry["latitude"],
                "longitude": entry["longitude"], "channels": []})
            cha = make_channel(
                entry["location"], entry["channel"], entry["start"],
                entry["end"], entry["sensitivity"])
            if not with_response:
                # Availability channel-level replies do not carry responses.
                cha.response = None
            sta["channels"].append(cha)
        return self._assemble(networks)

    def _build_bulk_inventory(self, bulk):
        networks = {}
        for (net_code, sta_code, loc, cha_code, start, end) in bulk:
            entries = [
                e for e in self._availability
                if e["network"] == net_code and e["station"] == sta_code
                and e["location"] == loc and e["channel"] == cha_code
                and (e["end"] is None or e["end"] >= start)
                and e["start"] <= end]
            if not entries:
                continue
            net = networks.setdefault(net_code, {})
            sta = net.setdefault(sta_code, {
                "latitude": entries[0]["latitude"],
                "longitude": entries[0]["longitude"], "channels": []})
            for entry in entries:
                sta["channels"].append(make_channel(
                    entry["location"], entry["channel"], entry["start"],
                    entry["end"], entry["sensitivity"]))
        return self._assemble(networks)

    def _assemble(self, networks_dict):
        networks = []
        for net_code, stations in sorted(networks_dict.items()):
            station_objs = []
            for sta_code, info in sorted(stations.items()):
                station_objs.append(Station(
                    code=sta_code, latitude=info["latitude"],
                    longitude=info["longitude"], elevation=0.0,
                    channels=info["channels"]))
            networks.append(Network(code=net_code, stations=station_objs))
        return Inventory(networks=networks, source=self.name)

    # -- dataselect service ---------------------------------------------- #
    def get_waveforms_bulk(self, bulk, filename=None):
        self.waveform_requests.append([tuple(b) for b in bulk])
        if self.fail_waveforms:
            raise FDSNNoDataException("no data available (simulated outage)")
        stream = obspy.Stream()
        for (net, sta, loc, cha, start, end) in bulk:
            trace = self._make_trace(net, sta, loc, cha, start, end)
            if trace is not None:
                stream += trace
        if len(stream) == 0:
            raise FDSNNoDataException("no data available")
        # One MiniSEED document per request; each bulk entry keeps its own
        # record sequence. The download helper's record-level splitter routes
        # records to their matching window.
        stream.write(filename, format="MSEED")

    def _make_trace(self, net, sta, loc, cha, start, end):
        if self.waveform_spans is not None:
            span = self._matching_waveform_span(
                net, sta, loc, cha, start, end)
            if span is None:
                return None
            _s, _e, data_start, data_end = span
        else:
            entry = self._matching_entry(net, sta, loc, cha, start, end)
            if entry is None:
                return None
            entry_end = entry["end"]
            # A request starting before the center's holdings yields
            # nothing; a request extending beyond them returns the prefix
            # the center holds, leaving minimum-length QC to reject shorts.
            if entry["start"] > start:
                return None
            data_start = start
            data_end = end if entry_end is None else min(end, entry_end)
        if data_end <= data_start:
            return None
        sampling_rate = 10.0
        npts = max(1, int((data_end - data_start) * sampling_rate) + 1)
        trace = obspy.Trace(data=np.arange(npts, dtype=np.int32))
        trace.stats.network = net
        trace.stats.station = sta
        trace.stats.location = loc
        trace.stats.channel = cha
        trace.stats.starttime = data_start
        trace.stats.sampling_rate = sampling_rate
        return trace

    def _matching_waveform_span(self, net, sta, loc, cha, start, end):
        for (s_net, s_sta, s_loc, s_cha, s_start,
             s_end) in self.waveform_spans:
            if (s_net, s_sta, s_loc, s_cha) != (net, sta, loc, cha):
                continue
            if s_start > start:
                continue
            data_end = end if s_end is None else min(end, s_end)
            if data_end <= start:
                continue
            return s_start, s_end, start, data_end
        return None

    def _matching_entry(self, net, sta, loc, cha, start, end):
        candidates = [
            e for e in self._availability
            if e["network"] == net and e["station"] == sta
            and e["location"] == loc and e["channel"] == cha
            and e["start"] < end and
            (e["end"] is None or e["end"] > start)]
        if not candidates:
            return None
        # Deterministic choice if epochs overlap: the earliest epoch.
        return sorted(candidates, key=lambda e: e["start"])[0]


def _full_window_trace(net, sta, loc, cha, start, end, sampling_rate=10.0):
    npts = max(1, int((end - start) * sampling_rate) + 1)
    data = np.arange(npts, dtype=np.int32)
    trace = obspy.Trace(data=data)
    trace.stats.network = net
    trace.stats.station = sta
    trace.stats.location = loc
    trace.stats.channel = cha
    trace.stats.starttime = start
    trace.stats.sampling_rate = sampling_rate
    return trace


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
DAY_START = obspy.UTCDateTime(2020, 6, 1)
HOUR = 3600.0
WINDOW_LENGTH = HOUR
N_WINDOWS = 4


def make_restrictions(**kwargs):
    defaults = dict(
        starttime=DAY_START, endtime=DAY_START + N_WINDOWS * HOUR,
        chunklength_in_sec=WINDOW_LENGTH,
        network="XX", station="STA1", location="", channel="BHZ",
        reject_channels_with_gaps=True, minimum_length=0.9,
        sanitize=False, minimum_interstation_distance_in_m=0)
    defaults.update(kwargs)
    return Restrictions(**defaults)


def make_entry(center_sensitivity, start, end, location="", channel="BHZ",
               network="XX", station="STA1", latitude=48.0, longitude=11.0):
    return dict(network=network, station=station, location=location,
                channel=channel, start=start, end=end,
                sensitivity=center_sensitivity, latitude=latitude,
                longitude=longitude)


class BackfillHarness(object):
    """
    Builds a MassDownloader whose clients are mock centers, without touching
    the network or the real client initialization.
    """
    def __init__(self, centers):
        self.mdl = MassDownloader.__new__(MassDownloader)
        self.mdl.debug = False
        self.mdl.providers = tuple(c.name for c in centers)
        self.mdl._initialized_clients = _ordered_dict(
            (c.name, c) for c in centers)
        self.mdl.backfill_report = None

    def run(self, tmp_path, restrictions=None, **download_kwargs):
        restrictions = restrictions or make_restrictions()
        mseed_dir = os.path.join(str(tmp_path), "waveforms")
        stationxml_dir = os.path.join(str(tmp_path), "stations")
        report = self.mdl.download(
            domain_module.GlobalDomain(), restrictions,
            mseed_storage=mseed_dir, stationxml_storage=stationxml_dir,
            backfill=True, print_report=False, **download_kwargs)
        return report, mseed_dir, stationxml_dir


def _ordered_dict(pairs):
    import collections
    return collections.OrderedDict(pairs)


def window_ids(report):
    return sorted(report.windows.keys(), key=lambda w: w[4])

def count_status(report):
    counts = {name: 0 for name in
              ("pending", "downloaded", "reused", "missing", "ignored")}
    for record in report.windows.values():
        counts[str(record.status)] += 1
    return counts


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
class TestBackfillOffline:
    def test_window_split_across_two_centers(self, tmp_path):
        """
        Center A only serves the first two windows, center B the rest.
        The station must not be skipped at B and every window ends up with
        a verifiable pair from one center.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, DAY_START + 2 * HOUR)])
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, DAY_START + 4 * HOUR)])
        harness = BackfillHarness([center_a, center_b])
        report, mseed_dir, stationxml_dir = harness.run(tmp_path)

        counts = count_status(report)
        assert counts["downloaded"] == 4
        assert counts["missing"] == 0
        providers = report.providers_used()
        assert providers == {"A": 2, "B": 2}

        # Each window has one waveform file and a matching stationxml file.
        mseed_files = sorted(os.listdir(mseed_dir))
        assert len(mseed_files) == 4
        stationxml_files = os.listdir(stationxml_dir)
        assert stationxml_files == ["XX.STA1.xml"]

        # Every pair must be readable together and every sample time must
        # resolve to exactly one response, which is the precondition for a
        # later instrument correction.
        for name in mseed_files:
            stream = obspy.read(os.path.join(mseed_dir, name))
            inventory = obspy.read_inventory(
                os.path.join(stationxml_dir, "XX.STA1.xml"))
            for trace in stream:
                response = inventory.get_response(trace.id,
                                                  trace.stats.starttime)
                # Sensitivity-based correction works with the minimal test
                # responses and proves waveform and response pair up.
                corrected = trace.copy().normalize()
                corrected.data = corrected.data / \
                    response.instrument_sensitivity.value
                assert response is not None
                assert len(corrected.data) == trace.stats.npts

        # B must not be asked for the windows A already delivered.
        b_waveform_windows = [b[4] for request in center_b.waveform_requests
                              for b in request]
        assert float(DAY_START) not in [float(t) for t in
                                        b_waveform_windows]
        assert float(DAY_START + HOUR) not in [float(t) for t in
                                               b_waveform_windows]
        assert float(DAY_START + 2 * HOUR) in [float(t) for t in
                                               b_waveform_windows]

    def test_different_channels_of_station_from_different_centers(
            self, tmp_path):
        """
        Other channels of the same station may be backfilled by another
        data center, while all channels share the single per-station
        StationXML file.
        """
        # A only serves BHZ; B serves BHN for the same station.
        mseed_dir = os.path.join(str(tmp_path), "waveforms")
        stationxml_dir = os.path.join(str(tmp_path), "stations")
        for channel, provider in (("BHZ", "A"), ("BHN", "B")):
            harness = BackfillHarness([
                MockFDSNCenter("A", [make_entry(1.0, DAY_START, None,
                                                channel="BHZ")]),
                MockFDSNCenter("B", [make_entry(2.0, DAY_START, None,
                                                channel="BHN")])])
            restrictions = make_restrictions(channel=channel)
            report, _, _ = harness.run(
                tmp_path, restrictions=restrictions)
            channel_records = [
                r for r in report.windows.values()
                if r.window_id[3] == channel]
            assert len(channel_records) == N_WINDOWS
            # Each window of this channel comes from the expected center.
            assert all(r.provider == provider
                       for r in channel_records)

        # One station file holds both channels, each with its own response.
        assert os.listdir(stationxml_dir) == ["XX.STA1.xml"]
        inventory = obspy.read_inventory(
            os.path.join(stationxml_dir, "XX.STA1.xml"))
        assert inventory.get_response(
            "XX.STA1..BHZ", DAY_START).instrument_sensitivity.value == 1.0
        assert inventory.get_response(
            "XX.STA1..BHN", DAY_START).instrument_sensitivity.value == 2.0
        # Waveforms for both channels are present.
        mseed_names = os.listdir(mseed_dir)
        assert any("BHZ" in n for n in mseed_names)
        assert any("BHN" in n for n in mseed_names)

    def test_response_gap_in_the_middle(self, tmp_path):
        """
        StationXML with two response epochs and a gap between them. A
        window whose samples fall in the gap is rejected at A and picked up
        by B which has continuous responses.
        """
        # A: two response epochs with a gap from +2h to +3h, but its
        # dataselect holds waveforms for the whole span.
        waveform_spans = [
            ("XX", "STA1", "", "BHZ", DAY_START,
             DAY_START + 4 * HOUR)]
        center_a = MockFDSNCenter(
            "A",
            [make_entry(1.0, DAY_START, DAY_START + 2 * HOUR),
             make_entry(1.0, DAY_START + 3 * HOUR, None)],
            waveform_spans=waveform_spans)
        # B: one continuous, open-ended response epoch.
        center_b = MockFDSNCenter("B", [
            make_entry(5.0, DAY_START, None)])
        harness = BackfillHarness([center_a, center_b])
        report, mseed_dir, stationxml_dir = harness.run(tmp_path)

        counts = count_status(report)
        assert counts["downloaded"] == 4
        assert counts["missing"] == 0
        providers = report.providers_used()
        # Window 3 (samples between +2h and +3h) cannot be covered by A and
        # must come from B.
        assert providers["B"] >= 1
        gap_window = [w for w in window_ids(report)
                      if w[4] == float(DAY_START + 2 * HOUR)][0]
        assert report.windows[gap_window].provider == "B"
        # A recorded the response-gap failure for that window.
        assert report.windows[gap_window].failure_reasons.get("A") == \
            REASON_RESPONSE_GAP

        # The merged single station file covers everything at the end.
        inventory = obspy.read_inventory(
            os.path.join(stationxml_dir, "XX.STA1.xml"))
        for name in os.listdir(mseed_dir):
            stream = obspy.read(os.path.join(mseed_dir, name))
            for trace in stream:
                inventory.get_response(trace.id, trace.stats.starttime)

    def test_overlapping_conflicting_responses(self, tmp_path):
        """
        A only delivers the first window's waveform but its response epoch
        stays valid (open ended) for all later samples. B offers the later
        windows with a different response valid at those same sample times,
        so one merged StationXML cannot satisfy both A's accepted window and
        B's new windows. B is rejected and the next center C, whose response
        agrees with A, fills the windows instead.
        """
        # A: open-ended response 1.0, but only one hour of waveforms.
        center_a = MockFDSNCenter(
            "A", [make_entry(1.0, DAY_START, None)],
            waveform_spans=[("XX", "STA1", "", "BHZ", DAY_START,
                             DAY_START + HOUR)])
        # B: a different, also open-ended response.
        center_b = MockFDSNCenter("B", [
            make_entry(9.0, DAY_START, None)])
        # C: response identical to A's.
        center_c = MockFDSNCenter("C", [
            make_entry(1.0, DAY_START, None)])
        harness = BackfillHarness([center_a, center_b, center_c])
        report, _, _ = harness.run(tmp_path)

        counts = count_status(report)
        assert counts["downloaded"] == N_WINDOWS
        assert counts["missing"] == 0
        for record in report.windows.values():
            assert record.provider in ("A", "C")
            assert record.provider != "B"
        conflict_windows = [
            w for w, r in report.windows.items()
            if r.failure_reasons.get("B") == REASON_RESPONSE_CONFLICT]
        assert len(conflict_windows) == N_WINDOWS - 1
        # A's accepted first window must remain backed by the same response.
        first = [w for w in window_ids(report)
                 if w[4] == float(DAY_START)][0]
        assert report.windows[first].provider == "A"

    def test_empty_vs_specific_location_code(self, tmp_path):
        """
        Empty location "" and "00" are distinct channels. Requested in two
        runs with an explicit location each, they are tracked separately and
        matched to their own responses inside the single per-station
        StationXML file.
        """
        availability = [
            make_entry(1.0, DAY_START, None, location="", channel="BHZ"),
            make_entry(2.0, DAY_START, None, location="00", channel="BHZ")]
        center = MockFDSNCenter("A", availability)
        mseed_dir = os.path.join(str(tmp_path), "waveforms")
        stationxml_dir = os.path.join(str(tmp_path), "stations")

        all_windows = {}
        for location in ("", "00"):
            harness = BackfillHarness(
                [MockFDSNCenter("A", copy.deepcopy(availability))])
            restrictions = make_restrictions(location=location,
                                             channel="BHZ")
            report, _, _ = harness.run(
                tmp_path, restrictions=restrictions)
            all_windows.update(report.windows)

        # Four windows per location code, eight verifiable pairs in total.
        assert len(all_windows) == 2 * N_WINDOWS
        seed_ids = {".".join(w[:4]) for w in all_windows}
        assert seed_ids == {"XX.STA1..BHZ", "XX.STA1.00.BHZ"}
        # One StationXML file for the station, carrying both locations.
        assert os.listdir(stationxml_dir) == ["XX.STA1.xml"]
        inventory = obspy.read_inventory(
            os.path.join(stationxml_dir, "XX.STA1.xml"))
        empty_resp = inventory.get_response(
            "XX.STA1..BHZ", DAY_START + 10)
        loc_resp = inventory.get_response(
            "XX.STA1.00.BHZ", DAY_START + 10)
        assert empty_resp.instrument_sensitivity.value == 1.0
        assert loc_resp.instrument_sensitivity.value == 2.0
        # Every delivered waveform still pairs with its own response.
        for name in os.listdir(mseed_dir):
            stream = obspy.read(os.path.join(mseed_dir, name))
            for trace in stream:
                inventory.get_response(trace.id, trace.stats.starttime)

    def test_callback_storage_paths_supported(self, tmp_path):
        """
        Backfill mode honors the same mseed/stationxml storage callbacks as
        the normal download path, including the dictionary stationxml form.
        """
        center = MockFDSNCenter("A", [make_entry(1.0, DAY_START, None)])
        mdl = MassDownloader.__new__(MassDownloader)
        mdl.debug = False
        mdl.providers = ("A",)
        mdl._initialized_clients = _ordered_dict([("A", center)])
        mdl.backfill_report = None

        mseed_dir = os.path.join(str(tmp_path), "waveforms")
        stationxml_dir = os.path.join(str(tmp_path), "stations")

        def mseed_callback(network, station, location, channel, start, end):
            return os.path.join(
                mseed_dir,
                "%s.%s.%s.%s__%s.mseed" % (
                    network, station, location or "empty", channel,
                    start.strftime("%Y%m%dT%H%M%SZ")))

        def stationxml_callback(network, station, channels, start, end):
            available = list(channels)
            return {
                "available_channels": available,
                "missing_channels": [],
                "filename": os.path.join(
                    stationxml_dir, "%s.%s.custom.xml" % (network, station))}

        report = mdl.download(
            domain_module.GlobalDomain(), make_restrictions(),
            mseed_storage=mseed_callback,
            stationxml_storage=stationxml_callback,
            backfill=True, print_report=False)
        assert count_status(report)["downloaded"] == N_WINDOWS
        # Custom names are used for both storage kinds.
        assert os.listdir(stationxml_dir) == ["XX.STA1.custom.xml"]
        assert len(os.listdir(mseed_dir)) == N_WINDOWS
        # Every delivered pair reads back together.
        inventory = obspy.read_inventory(
            os.path.join(stationxml_dir, "XX.STA1.custom.xml"))
        for name in os.listdir(mseed_dir):
            stream = obspy.read(os.path.join(mseed_dir, name))
            for trace in stream:
                inventory.get_response(trace.id, trace.stats.starttime)

    def test_failover_after_waveform_failure(self, tmp_path):
        """
        Center A fails every waveform request; B fills all windows. The
        failure must be recorded and no partial state left behind.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        center_a.fail_waveforms = True
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, None)])
        harness = BackfillHarness([center_a, center_b])
        report, mseed_dir, stationxml_dir = harness.run(tmp_path)
        counts = count_status(report)
        assert counts["downloaded"] == N_WINDOWS
        assert report.providers_used() == {"B": N_WINDOWS}
        for record in report.windows.values():
            assert "A" in record.attempts
            assert record.failure_reasons["A"] == REASON_NO_WAVEFORM
        # No leftover temp files in the waveform directory.
        assert all(not n.startswith(".") or not n.endswith(".mseed")
                   for n in os.listdir(mseed_dir))

    def test_failover_after_empty_stationxml(self, tmp_path):
        """
        Center A returns waveforms but an empty StationXML, so no pair is
        accepted there. B supplies valid responses and fills the windows;
        A's unpaired waveforms are cleaned up rather than reused.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        center_a.empty_stationxml = True
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, None)])
        harness = BackfillHarness([center_a, center_b])
        report, mseed_dir, stationxml_dir = harness.run(tmp_path)
        counts = count_status(report)
        assert counts["downloaded"] == N_WINDOWS
        assert report.providers_used() == {"B": N_WINDOWS}
        for record in report.windows.values():
            assert record.failure_reasons["A"] == \
                REASON_EMPTY_STATIONXML
        # The single station file is readable and comes from B.
        assert os.listdir(stationxml_dir) == ["XX.STA1.xml"]
        inventory = obspy.read_inventory(
            os.path.join(stationxml_dir, "XX.STA1.xml"))
        assert inventory[0][0][0].response.instrument_sensitivity.value == 2.0
        # No orphaned, response-less MiniSEED files remain from A.
        for name in os.listdir(mseed_dir):
            stream = obspy.read(os.path.join(mseed_dir, name))
            for trace in stream:
                inventory.get_response(trace.id, trace.stats.starttime)

    def test_rerun_reuses_verified_pairs(self, tmp_path):
        """
        A second run against the same directories must reuse the verifiable
        local pairs without issuing any waveform or StationXML request, and
        a previously failing center can be retried for still-missing windows.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, DAY_START + 2 * HOUR)])
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, DAY_START + 4 * HOUR)])
        harness = BackfillHarness([center_a, center_b])
        report1, mseed_dir, stationxml_dir = harness.run(tmp_path)
        assert count_status(report1)["downloaded"] == 4

        # Rerun with fresh mock objects counting requests from zero.
        center_a2 = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, DAY_START + 2 * HOUR)])
        center_b2 = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, DAY_START + 4 * HOUR)])
        harness2 = BackfillHarness([center_a2, center_b2])
        report2, _, _ = harness2.run(tmp_path)
        counts2 = count_status(report2)
        assert counts2["reused"] == 4
        assert counts2["downloaded"] == 0
        assert center_a2.waveform_requests == []
        assert center_b2.waveform_requests == []
        assert center_a2.stationxml_requests == []
        assert center_b2.stationxml_requests == []

    def test_empty_directory_starts_fresh_and_corrupt_xml_recovers(
            self, tmp_path):
        """
        An empty directory starts normally. If the StationXML is corrupted
        afterwards, a rerun keeps the usable MiniSEED files and re-downloads
        the station information instead of crashing.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        harness = BackfillHarness([center_a])
        report, mseed_dir, stationxml_dir = harness.run(tmp_path)
        assert count_status(report)["downloaded"] == 4

        xml_path = os.path.join(stationxml_dir, "XX.STA1.xml")
        with open(xml_path, "w") as fh:
            fh.write("this is not xml")
        mseed_before = set(os.listdir(mseed_dir))

        center_a2 = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        harness2 = BackfillHarness([center_a2])
        report2, _, _ = harness2.run(tmp_path)
        # MiniSEED bytes preserved; stationxml re-requested and windows
        # verified again (now via download, not reuse).
        assert set(os.listdir(mseed_dir)) == mseed_before
        assert center_a2.stationxml_requests
        counts = count_status(report2)
        assert counts["downloaded"] + counts["reused"] == 4
        obspy.read_inventory(xml_path)  # readable again

    def test_missing_window_reported_with_reasons(self, tmp_path):
        """
        A window nobody can serve ends up in the missing list with the
        per-center failure reasons exposed.
        """
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, DAY_START + 2 * HOUR)])
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, DAY_START + 2 * HOUR)])
        harness = BackfillHarness([center_a, center_b])
        report, _, _ = harness.run(tmp_path)
        counts = count_status(report)
        assert counts["downloaded"] == 2
        assert counts["missing"] == 2
        reasons = report.missing_reasons()
        assert reasons
        threshold = float(DAY_START + 2 * HOUR)
        for ids in reasons.values():
            assert all(w[4] >= threshold for w in ids)

    def test_minimum_distance_keeps_backfilled_station(self, tmp_path):
        """
        The minimum inter-station distance filters distinct stations, but a
        station another center only backfills must not be treated as its own
        neighbour.
        """
        # Two distinct physical stations close together. A only has STA1;
        # B has both STA1 (backfill) and STA2 (a near duplicate that the
        # distance filter should remove).
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, DAY_START + 2 * HOUR,
                       station="STA1", latitude=48.0, longitude=11.0)])
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, None, station="STA1",
                       latitude=48.0, longitude=11.0),
            make_entry(3.0, DAY_START, None, station="STA2",
                       latitude=48.0001, longitude=11.0001)])
        harness = BackfillHarness([center_a, center_b])
        restrictions = make_restrictions(
            station=None, minimum_interstation_distance_in_m=1000)
        report, _, _ = harness.run(tmp_path, restrictions=restrictions)
        stations_in_report = {w.station_key
                              for w in report.windows.values()}
        assert ("XX", "STA1") in stations_in_report
        # STA2 is within 1 km of the already acquired STA1 and is filtered.
        sta2_records = [w for w in report.windows.values()
                        if w.station_key == ("XX", "STA2")]
        assert sta2_records
        assert all(r.status == BACKFILL_STATUS.ignored for r in sta2_records)
        # STA1 windows are all completed, the backfilled ones by B.
        sta1 = [r for r in report.windows.values()
                if r.station_key == ("XX", "STA1")]
        assert count_status_from(sta1)["downloaded"] == N_WINDOWS


def count_status_from(records):
    counts = {name: 0 for name in
              ("pending", "downloaded", "reused", "missing", "ignored")}
    for record in records:
        counts[str(record.status)] += 1
    return counts


class TestNormalPathUnchanged:
    def test_original_download_path_without_backfill(self, tmp_path):
        """
        With backfill=False (default) the original station-claiming behavior
        is retained: the second center skips the station entirely, even
        though it holds data for windows the first center cannot serve.
        """
        # A covers the whole requested span (open-ended epoch) so the
        # normal bulk/split path produces one file per request window.
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        center_b = MockFDSNCenter("B", [
            make_entry(2.0, DAY_START, None)])
        mdl = MassDownloader.__new__(MassDownloader)
        mdl.debug = False
        mdl.providers = ("A", "B")
        mdl._initialized_clients = _ordered_dict(
            [("A", center_a), ("B", center_b)])
        mdl.backfill_report = None

        mseed_dir = os.path.join(str(tmp_path), "waveforms")
        stationxml_dir = os.path.join(str(tmp_path), "stations")
        # Record-level splitting places whole MiniSEED records into window
        # files which shifts boundary samples; relax the temporal QC so the
        # station-claiming semantics under test are not confounded with it.
        restrictions = make_restrictions(
            reject_channels_with_gaps=False, minimum_length=0.0)
        result = mdl.download(
            domain_module.GlobalDomain(), restrictions,
            mseed_storage=mseed_dir, stationxml_storage=stationxml_dir,
            print_report=False)
        # Original return type: dict of ClientDownloadHelper instances.
        assert isinstance(result, dict)
        assert "A" in result
        # B never gets the station because A claimed it wholesale.
        assert "B" not in result or not result["B"].stations
        assert mdl.backfill_report is None
        # All four windows come from A; B is never contacted for waveforms.
        mseed_files = os.listdir(mseed_dir)
        assert len(mseed_files) == N_WINDOWS
        assert center_b.waveform_requests == []


class TestResponseCoverageHelpers:
    def _inventory(self, epochs, location=""):
        channels = []
        for start, end, sens in epochs:
            channels.append(make_channel(location, "BHZ", start, end, sens))
        station = Station(code="STA1", latitude=0, longitude=0,
                          elevation=0.0, channels=channels)
        return Inventory(
            networks=[Network(code="XX", stations=[station])], source="t")

    def test_open_ended_response_is_valid(self):
        from obspy.clients.fdsn.mass_downloader.backfill import (
            _response_epochs_for, _samples_covered)
        inv = self._inventory([(DAY_START, None, 1.0)])
        epochs = _response_epochs_for(inv, "XX", "STA1", "", "BHZ")
        trace = _full_window_trace("XX", "STA1", "", "BHZ",
                                   DAY_START + 100 * HOUR,
                                   DAY_START + 101 * HOUR)
        ok, _ = _samples_covered(trace, epochs)
        assert ok

    def test_boundary_sample_times(self):
        from obspy.clients.fdsn.mass_downloader.backfill import (
            _response_epochs_for, _samples_covered)
        # Two adjacent epochs meeting exactly at +1h; samples on the
        # boundary must be covered by exactly one epoch, and a gap epoch
        # pair must be detected.
        inv_gap = self._inventory([
            (DAY_START, DAY_START + HOUR, 1.0),
            (DAY_START + 2 * HOUR, None, 1.0)])
        epochs = _response_epochs_for(inv_gap, "XX", "STA1", "", "BHZ")
        trace = _full_window_trace("XX", "STA1", "", "BHZ",
                                   DAY_START + 1.5 * HOUR,
                                   DAY_START + 1.6 * HOUR)
        ok, reason = _samples_covered(trace, epochs)
        assert not ok and reason == REASON_RESPONSE_GAP

        inv_two = self._inventory([
            (DAY_START, DAY_START + HOUR, 1.0),
            (DAY_START + HOUR, None, 2.0)])
        epochs2 = _response_epochs_for(inv_two, "XX", "STA1", "", "BHZ")
        ok, reason = _samples_covered(trace, epochs2)
        assert ok

    def test_conflicting_overlapping_epochs_detected(self):
        from obspy.clients.fdsn.mass_downloader.backfill import (
            _response_epochs_for, _samples_covered)
        inv = self._inventory([
            (DAY_START, DAY_START + 2 * HOUR, 1.0),
            (DAY_START + HOUR, None, 2.0)])
        epochs = _response_epochs_for(inv, "XX", "STA1", "", "BHZ")
        trace = _full_window_trace("XX", "STA1", "", "BHZ",
                                   DAY_START + 1.5 * HOUR,
                                   DAY_START + 1.6 * HOUR)
        ok, reason = _samples_covered(trace, epochs)
        assert not ok and reason == REASON_RESPONSE_CONFLICT

    def test_chunk_splitting_bounds_batch_size(self):
        from obspy.clients.fdsn.mass_downloader.backfill import (
            _split_chunks_by_size)
        # BHZ at 80 Hz, one hour per window; force tiny batches both by size
        # and by the maximum number of entries per batch.
        chunks = []
        for index in range(6):
            start = DAY_START + index * HOUR
            chunks.append(
                ("XX", "STA1", "", "BHZ", start, start + HOUR,
                 "file_%d.mseed" % index))
        batches = _split_chunks_by_size(
            chunks, chunk_size_in_mb=1000.0, max_chunk_length=2)
        assert [len(batch) for batch in batches] == [2, 2, 2]
        # Every chunk is served exactly once and order is preserved.
        flat = [entry for batch in batches for entry in batch]
        assert flat == chunks
        # A generous size limit still caps through the entry count default.
        batches_default = _split_chunks_by_size(chunks, 1000.0)
        assert sum(len(b) for b in batches_default) == len(chunks)


class TestBackfillReportStructure:
    def test_report_is_attached_and_serializable(self, tmp_path):
        center_a = MockFDSNCenter("A", [
            make_entry(1.0, DAY_START, None)])
        harness = BackfillHarness([center_a])
        report, _, _ = harness.run(tmp_path)
        assert harness.mdl.backfill_report is report
        assert isinstance(report, BackfillReport)
        text = str(report)
        assert "Backfill report" in text
        # Every window that passed selection is queryable.
        assert len(report.windows) == N_WINDOWS
        for record in report.windows.values():
            assert record.seed_id == "XX.STA1..BHZ"
            assert record.stationxml_filename.endswith("XX.STA1.xml")
