# -*- coding: utf-8 -*-
"""
Optional backfill ("gap-filling") mode for the mass downloader.

In contrast to the normal download mode - where a data center that returns a
station claims the whole station for the rest of the run - the backfill mode
works on the level of individual request windows (the time intervals yielded
by :class:`~.restrictions.Restrictions`). Each window is assigned to the first
data center, in the downloader's provider order, that can deliver a
*verifiable pair* of

1. a MiniSEED waveform file passing the normal quality controls, and
2. a StationXML instrument response that is valid for every actual sample time
   of that waveform (read back from the file, including the full SEED id).

Verified windows are removed from the waveform requests sent to subsequent
data centers; the StationXML request of a data center only covers the windows
its waveforms actually delivered. Each station keeps a single StationXML file
that is merged across data centers and only ever replaced once it has been
demonstrated that all already accepted windows of that station still read a
valid response from the new file.

The module is fully offline capable: it only talks to the FDSN client objects
handed to it, which may be mocks in the test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import copy
import os
import shutil
import tempfile

import obspy

from . import utils


#: Status of an individual request window in backfill mode.
BACKFILL_STATUS = obspy.core.util.Enum([
    # The window still has to be attempted at a data center.
    "pending",
    # A verifiable waveform/response pair was downloaded from a data center.
    "downloaded",
    # A verifiable waveform/response pair already existed on disk and was
    # reused without any network request.
    "reused",
    # No data center could deliver a verifiable pair. See the window's
    # ``failure_reasons`` for the per-data-center explanations.
    "missing",
    # The window was not requested, e.g. the mseed storage callback returned
    # True or the station was removed by the inter-station distance filter.
    "ignored"])


# Reason strings used in the report. They are part of the public reporting
# API and thus kept stable and human readable.
REASON_NOT_OFFERED = "not offered by any remaining data center"
REASON_DISTANCE_FILTERED = \
    "station removed by minimum inter-station distance filtering"
REASON_STORAGE_IGNORED = "mseed storage callback returned True"
REASON_NO_WAVEFORM = "no usable MiniSEED waveform delivered"
REASON_WAVEFORM_QC = "MiniSEED failed minimum length/gap quality control"
REASON_EMPTY_STATIONXML = "StationXML response was empty or unreadable"
REASON_RESPONSE_GAP = \
    "samples fall in a gap of the instrument response coverage"
REASON_RESPONSE_CONFLICT = \
    "conflicting instrument responses at the same sample time"
REASON_RESPONSE_MISMATCH = \
    "response could not be matched to the read-back full channel id"
REASON_WRITE_FAILED = "merged StationXML could not be written"
REASON_NO_LOCAL_PAIR = "no verifiable local waveform/response pair"


def _response_fingerprint(response):
    """
    Hashable identity of a response that is stable across a StationXML
    write/read roundtrip.

    The full ``Response == Response`` comparison cannot be used here because
    the StationXML serialization turns some ``None`` values into the string
    ``"None"``. The overall sensitivity (value, frequency, units) together
    with the number and sequence numbers of the response stages identify an
    instrument response for the purpose of conflict detection.
    """
    if response is None:
        return None
    sens = response.instrument_sensitivity
    if sens is not None:
        sens_tuple = (sens.value, sens.frequency, sens.input_units,
                      sens.output_units)
    else:
        sens_tuple = None
    stages = tuple((stage.stage_sequence_number,
                    getattr(stage, "stage_gain", None))
                   for stage in response.response_stages)
    return (sens_tuple, stages,
            response.instrument_polynomial is not None,
            response.resource_id)


class WindowRecord(object):
    """
    Bookkeeping record for a single request window of a single
    ``NET.STA.LOC.CHA`` channel.

    ObsPy :class:`~obspy.core.utcdatetime.UTCDateTime` instances are
    explicitly unhashable, thus the window key stores the window bounds as
    floating point POSIX timestamps; the original UTCDateTime bounds are
    available from :attr:`starttime` and :attr:`endtime`.

    :type window_id: tuple
    :param window_id: ``(network, station, location, channel, starttime,
        endtime)`` tuple; ``starttime`` and ``endtime`` may be
        :class:`~obspy.core.utcdatetime.UTCDateTime` instances or floats.
    """
    __slots__ = ["window_id", "status", "provider", "mseed_filename",
                 "stationxml_filename", "attempts", "failure_reasons"]

    def __init__(self, window_id):
        net, sta, loc, cha, start, end = window_id
        self.window_id = (net, sta, loc, cha, float(start), float(end))
        self.status = BACKFILL_STATUS.pending
        #: The data center that ultimately delivered the verifiable pair.
        self.provider = None
        self.mseed_filename = None
        self.stationxml_filename = None
        #: Data centers at which the window was attempted, in order.
        self.attempts = []
        #: Mapping data center name -> reason string for failed attempts.
        self.failure_reasons = collections.OrderedDict()

    @property
    def key(self):
        return self.window_id

    @property
    def starttime(self):
        return obspy.UTCDateTime(self.window_id[4])

    @property
    def endtime(self):
        return obspy.UTCDateTime(self.window_id[5])

    @property
    def seed_id(self):
        return ".".join(self.window_id[:4])

    @property
    def station_key(self):
        return (self.window_id[0], self.window_id[1])

    def __repr__(self):
        return ("WindowRecord(%s, %s - %s, status='%s', provider=%r)" % (
            self.seed_id, self.starttime, self.endtime,
            self.status, self.provider))


class BackfillReport(object):
    """
    Result of a backfill-mode download run.

    Returned (and also attached to :attr:`MassDownloader.backfill_report`) by
    :meth:`MassDownloader.download` when backfill mode is enabled. Every
    window that passed the normal station and channel selection is recorded
    here.
    """
    def __init__(self):
        #: dict mapping window ids to :class:`WindowRecord` instances.
        self.windows = {}
        #: dict mapping station keys to the single StationXML filename kept
        #: for the station.
        self.stationxml_files = {}

    def _get(self, window_id):
        # WindowRecord normalizes the unhashable UTCDateTime bounds into
        # float timestamps, so both UTCDateTime and float window ids can be
        # used as dictionary keys.
        window_id = WindowRecord(window_id).window_id
        if window_id not in self.windows:
            self.windows[window_id] = WindowRecord(window_id)
        return self.windows[window_id]

    @property
    def downloaded(self):
        """Records satisfied by newly downloaded verifiable pairs."""
        return [w for w in self.windows.values()
                if w.status == BACKFILL_STATUS.downloaded]

    @property
    def reused(self):
        """Records satisfied from pre-existing, verifiable local pairs."""
        return [w for w in self.windows.values()
                if w.status == BACKFILL_STATUS.reused]

    @property
    def missing(self):
        """Records no data center could satisfy."""
        return [w for w in self.windows.values()
                if w.status == BACKFILL_STATUS.missing]

    @property
    def ignored(self):
        """Records that were never requested."""
        return [w for w in self.windows.values()
                if w.status == BACKFILL_STATUS.ignored]

    def providers_used(self):
        """
        :returns: dict mapping data center names to the number of windows
            newly downloaded from them.
        """
        counter = collections.Counter(
            w.provider for w in self.downloaded if w.provider is not None)
        return dict(counter)

    def missing_reasons(self):
        """
        :returns: dict mapping reason strings to lists of window ids.
        """
        reasons = collections.OrderedDict()
        for record in self.missing:
            # Use the reason of the last attempt if available, otherwise
            # the generic "not offered" reason.
            if record.failure_reasons:
                reason = list(record.failure_reasons.values())[-1]
            else:
                reason = REASON_NOT_OFFERED
            reasons.setdefault(reason, []).append(record.window_id)
        return reasons

    def __str__(self):
        lines = ["Backfill report:",
                 "  %i windows total" % len(self.windows),
                 "  %i downloaded from data centers" % len(self.downloaded),
                 "  %i reused from local files" % len(self.reused),
                 "  %i missing" % len(self.missing),
                 "  %i ignored" % len(self.ignored)]
        for provider, count in sorted(self.providers_used().items()):
            lines.append("    %s: %i windows" % (provider, count))
        for reason, ids in self.missing_reasons().items():
            lines.append("    missing (%s): %i windows" % (reason, len(ids)))
        return "\n".join(lines)


class _StationState(object):
    """
    Internal state for one station across all data centers.
    """
    __slots__ = ["network", "station", "latitude", "longitude",
                 "stationxml_filename", "inventory", "channels",
                 "distance_filtered"]

    def __init__(self, network, station, latitude, longitude):
        self.network = network
        self.station = station
        self.latitude = latitude
        self.longitude = longitude
        self.stationxml_filename = None
        #: The merged inventory currently backing the on-disk StationXML
        #: file (None if no file exists yet).
        self.inventory = None
        #: Set of ``(location, channel)`` tuples handled for the station.
        self.channels = set()
        self.distance_filtered = False

    @property
    def key(self):
        return (self.network, self.station)


class BackfillDownloader(object):
    """
    Runs a download in backfill mode.

    :type mass_downloader: :class:`~.mass_downloader.MassDownloader`
    :param mass_downloader: The owning mass downloader with the initialized
        FDSN clients.
    :type domain: :class:`~.domain.Domain`
    :param domain: The spatial domain.
    :type restrictions: :class:`~.restrictions.Restrictions`
    :param restrictions: The temporal and SEED-id restrictions; the request
        windows are taken from its iterator.
    :param mseed_storage: MiniSEED storage path, folder or callback - exactly
        the same argument as for the normal download.
    :param stationxml_storage: StationXML storage path, folder or callback.
    """
    def __init__(self, mass_downloader, domain, restrictions, mseed_storage,
                 stationxml_storage, logger, chunk_size_in_mb,
                 threads_per_client):
        self.mass_downloader = mass_downloader
        self.clients = mass_downloader._initialized_clients
        self.domain = domain
        self.restrictions = restrictions
        self.mseed_storage = mseed_storage
        self.stationxml_storage = stationxml_storage
        self.logger = logger
        self.chunk_size_in_mb = chunk_size_in_mb
        # The parameter is accepted for API symmetry with the normal
        # download path. Backfill processes the bounded bulk chunks of one
        # data center sequentially on purpose: a chunk's windows have to be
        # verified against the single, merged per-station StationXML before
        # the next center is allowed to touch the same station, so concurrent
        # writers to that file would only add locking complexity without
        # reducing the number of requests.
        self.threads_per_client = threads_per_client

        self.report = BackfillReport()
        self._stations = {}
        self._request_windows = [
            (obspy.UTCDateTime(s), obspy.UTCDateTime(e))
            for s, e in restrictions]

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self):
        # Pick up results of earlier runs sharing the same storage
        # directories so verifiable local pairs can be reused without any
        # network request.
        self._discover_existing_stationxml()
        for client_name, client in self.clients.items():
            self._process_provider(client_name, client)
        self._finalize()
        return self.report

    def _discover_existing_stationxml(self):
        """
        Pre-load any readable StationXML files already present in the
        configured storage. Only plain directory (or path-like) storage is
        scanned automatically; callback/template storage is handled lazily per
        station through the normal path-resolution logic.
        """
        if isinstance(self.stationxml_storage, (str, bytes,
                                                os.PathLike)):
            folder = os.fspath(self.stationxml_storage)
        else:
            return
        if not os.path.isdir(folder):
            return
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".xml"):
                continue
            path = os.path.join(folder, filename)
            try:
                inventory = obspy.read_inventory(path)
            except Exception:
                # Unreadable file: leave it alone; it will be replaced once a
                # new verifiable response is available.
                continue
            for net in inventory:
                for sta in net:
                    key = (net.code, sta.code)
                    state = self._stations.get(key)
                    if state is None:
                        state = _StationState(
                            net.code, sta.code, sta.latitude,
                            sta.longitude)
                        self._stations[key] = state
                    # Only adopt the file if it is not already superseded.
                    if state.inventory is None:
                        state.inventory = inventory
                        state.stationxml_filename = path
                        self.report.stationxml_files[key] = path
                    for channel in sta.channels:
                        state.channels.add(
                            (channel.location_code, channel.code))

    # ------------------------------------------------------------------ #
    # Per-provider processing
    # ------------------------------------------------------------------ #
    def _process_provider(self, client_name, client):
        helper = self._get_availability_helper(client_name, client)
        if helper is None:
            self.logger.info(
                "Backfill: client '%s' offers no data." % client_name)
            return

        # Register all offered windows in the report and remember the
        # station coordinates for the distance filter.
        offered = self._register_offered_windows(helper)
        if not offered:
            return

        # Minimum inter-station distance. Stations already acquired by an
        # earlier data center are fixed reference points; a station that
        # another data center only uses to backfill is never treated as a
        # neighbour of itself.
        self._apply_minimum_distance(helper)

        # Build the set of windows this data center should actually attempt:
        # pending, not ignored, and belonging to a station that survived the
        # filters.
        pending = self._collect_pending_windows(client_name, helper)
        if not pending:
            self.logger.info(
                "Backfill: client '%s' has no outstanding windows." %
                client_name)
            return

        # Attempt to reuse complete local pairs before touching the network.
        pending = self._reuse_local_pairs(client_name, pending)
        if not pending:
            return

        # Bulk MiniSEED download for the remaining windows.
        downloaded = self._download_waveforms(client_name, client, pending)
        if not downloaded:
            for record in pending:
                self._record_failure(
                    record, client_name,
                    REASON_NO_WAVEFORM)
            return

        # StationXML per station, covering only the windows whose waveforms
        # this data center actually delivered.
        self._download_and_verify_stationxml(
            client_name, client, downloaded)

    def _get_availability_helper(self, client_name, client):
        from .download_helpers import ClientDownloadHelper
        helper = ClientDownloadHelper(
            client=client, client_name=client_name,
            restrictions=self.restrictions, domain=self.domain,
            mseed_storage=self.mseed_storage,
            stationxml_storage=self.stationxml_storage, logger=self.logger)
        helper.get_availability()
        if not helper:
            return None
        return helper

    def _register_offered_windows(self, helper):
        """
        Create a report entry for every offered NET.STA.LOC.CHA window and
        remember the station. Returns the number of windows registered.
        """
        count = 0
        for station in helper.stations.values():
            state = self._stations.get((station.network, station.station))
            if state is None:
                state = _StationState(
                    station.network, station.station,
                    station.latitude, station.longitude)
                self._stations[state.key] = state
            for channel in station.channels:
                state.channels.add((channel.location, channel.channel))
                for interval in channel.intervals:
                    window_id = (station.network, station.station,
                                 channel.location, channel.channel,
                                 interval.start, interval.end)
                    self.report._get(window_id)
                    count += 1
        return count

    def _apply_minimum_distance(self, helper):
        """
        Apply the minimum inter-station distance to stations that have not
        yet been acquired by an earlier data center.

        Stations with at least one satisfied window are fixed; brand new
        stations are filtered against those fixed stations and against each
        other, reusing the proven filtering logic of
        :class:`~.ClientDownloadHelper`.
        """
        min_dist = self.restrictions.minimum_interstation_distance_in_m
        if not min_dist:
            return

        fixed_states = [s for s in self._stations.values()
                        if self._station_has_satisfied_windows(s)]
        new_station_objects = []
        new_state_by_obj = {}
        for station in helper.stations.values():
            state = self._stations[(station.network, station.station)]
            if state in fixed_states or self._station_has_satisfied_windows(
                    state):
                continue
            if state.distance_filtered:
                continue
            new_station_objects.append(station)
            new_state_by_obj[(station.network, station.station)] = state

        if not new_station_objects:
            return

        # Build a throw-away helper carrying only the new stations and run
        # the existing, well tested filtering implementation with the fixed
        # stations as "existing" helpers.
        reference_helper = _DistanceReferenceHelper(
            restrictions=self.restrictions, logger=self.logger,
            stations=_station_objects_for_states(fixed_states))
        candidate_helper = _DistanceReferenceHelper(
            restrictions=self.restrictions, logger=self.logger,
            stations={key: sta for key, sta in helper.stations.items()
                      if key in new_state_by_obj})
        rejected = candidate_helper.filter_stations_based_on_minimum_distance(
            [reference_helper]) if reference_helper.stations else \
            candidate_helper.filter_stations_based_on_minimum_distance([])

        for station_key in rejected:
            state = new_state_by_obj[station_key]
            state.distance_filtered = True
            for record in self._records_of_station(state.key):
                if record.status == BACKFILL_STATUS.pending:
                    record.status = BACKFILL_STATUS.ignored
                    record.failure_reasons["__filter__"] = \
                        REASON_DISTANCE_FILTERED

    def _station_has_satisfied_windows(self, state):
        return any(r.station_key == state.key and r.status in
                   (BACKFILL_STATUS.downloaded, BACKFILL_STATUS.reused)
                   for r in self.report.windows.values())

    def _records_of_station(self, station_key):
        return [r for r in self.report.windows.values()
                if r.station_key == station_key]

    def _collect_pending_windows(self, client_name, helper):
        """
        Return the :class:`WindowRecord` instances this provider must
        attempt. Also marks storage-ignored windows.
        """
        pending = []
        helper_keys = set(helper.stations.keys())
        for record in self.report.windows.values():
            if record.station_key not in helper_keys:
                continue
            if record.status != BACKFILL_STATUS.pending:
                continue
            state = self._stations[record.station_key]
            if state.distance_filtered:
                continue
            # Honor the mseed storage callback: returning True means the
            # caller considers the window already handled externally.
            target = utils.get_mseed_filename(
                self.mseed_storage, record.window_id[0],
                record.window_id[1], record.window_id[2],
                record.window_id[3], record.starttime,
                record.endtime)
            if target is True:
                record.status = BACKFILL_STATUS.ignored
                record.failure_reasons["__storage__"] = \
                    REASON_STORAGE_IGNORED
                continue
            record.mseed_filename = target
            pending.append(record)
        return pending

    # ------------------------------------------------------------------ #
    # Local reuse
    # ------------------------------------------------------------------ #
    def _reuse_local_pairs(self, client_name, pending):
        remaining = []
        for record in pending:
            result = self._verify_local_pair(record)
            if result is True:
                record.status = BACKFILL_STATUS.reused
                record.provider = None
                record.attempts.append(client_name)
                state = self._stations[record.station_key]
                record.stationxml_filename = state.stationxml_filename
                self.logger.info(
                    "Backfill: reusing verifiable local pair for %s (%s)."
                    % (record.seed_id, record.starttime))
            else:
                remaining.append(record)
        return remaining

    def _verify_local_pair(self, record):
        """
        Verify that a pre-existing MiniSEED/StationXML pair on disk can be
        read back and that every sample of the waveform has a valid,
        non-conflicting response in the station file.
        """
        mseed_path = record.mseed_filename
        if not mseed_path or not os.path.isfile(mseed_path):
            return False
        state = self._stations[record.station_key]
        # Prefer a StationXML path already recorded for the station (e.g. by
        # a previous run sharing the same directories); otherwise derive it
        # from the storage settings.
        stationxml_path = state.stationxml_filename or \
            self.report.stationxml_files.get(state.key) or \
            self._resolve_stationxml_path(state)
        if not stationxml_path or not os.path.isfile(stationxml_path):
            return False
        try:
            stream = obspy.read(mseed_path, headonly=True)
        except Exception:
            return False
        ok, _reason = self._check_waveform_qc(record, stream)
        if not ok:
            return False
        try:
            inventory = obspy.read_inventory(stationxml_path)
        except Exception:
            return False
        ok, _reason = self._verify_window_against_inventory(
            record, stream, inventory)
        return ok

    # ------------------------------------------------------------------ #
    # MiniSEED downloading
    # ------------------------------------------------------------------ #
    def _download_waveforms(self, client_name, client, pending):
        """
        Bulk-download the waveforms for all pending windows, run the normal
        QC checks and return the records that passed.

        Verified windows from earlier data centers are not part of
        ``pending`` and thus never requested again.
        """
        tmpdir = tempfile.mkdtemp(prefix="obspy_backfill_mseed_")
        chunks = []
        record_by_tmpfile = {}
        failed_records = []
        try:
            for record in pending:
                net, sta, loc, cha = record.window_id[:4]
                start, end = record.starttime, record.endtime
                tmp_filename = os.path.join(
                    tmpdir,
                    "%s.%s.%s.%s__%s__%s.mseed" % (
                        net, sta, loc, cha,
                        start.strftime("%Y%m%dT%H%M%SZ"),
                        end.strftime("%Y%m%dT%H%M%SZ")))
                chunks.append((net, sta, loc, cha, start, end, tmp_filename))
                record_by_tmpfile[tmp_filename] = record

            # Honor the configured chunk size so a large batch of windows is
            # sent as several bounded bulk requests instead of one huge one.
            for chunk in _split_chunks_by_size(
                    chunks, self.chunk_size_in_mb):
                try:
                    # Do not merge adjacent windows: a center may only serve
                    # part of a longer span and each window must stay
                    # independently attributable for QC and backfill.
                    utils.download_and_split_mseed_bulk(
                        client, client_name, chunk, logger=self.logger,
                        merge_adjacent=False)
                except utils.ERRORS as e:
                    # A failed chunk only fails its own windows; windows from
                    # other chunks can still be delivered.
                    self.logger.error(
                        "Backfill: waveform request to '%s' failed: %s" %
                        (client_name, str(e)))
                    for entry in chunk:
                        failed_records.append(record_by_tmpfile[entry[6]])
            for record in failed_records:
                self._record_failure(record, client_name,
                                     REASON_NO_WAVEFORM)

            downloaded = []
            for tmp_filename, record in record_by_tmpfile.items():
                if not os.path.isfile(tmp_filename) or \
                        os.path.getsize(tmp_filename) == 0:
                    self._record_failure(record, client_name,
                                         REASON_NO_WAVEFORM)
                    continue
                try:
                    stream = obspy.read(tmp_filename, headonly=True)
                except Exception:
                    self._record_failure(record, client_name,
                                         REASON_NO_WAVEFORM)
                    continue
                ok, reason = self._check_waveform_qc(record, stream)
                if not ok:
                    self._record_failure(record, client_name, reason)
                    continue
                # Move the QC-passed waveform to its final location.
                target = record.mseed_filename
                os.makedirs(os.path.dirname(target), exist_ok=True)
                try:
                    shutil.move(tmp_filename, target)
                except OSError as e:
                    self._record_failure(record, client_name,
                                         "could not write MiniSEED: %s" % e)
                    continue
                record.attempts.append(client_name)
                downloaded.append((record, target, stream))
            return downloaded
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _check_waveform_qc(self, record, stream):
        """
        Apply the same QC rules as the normal download path: readable,
        non-empty, gap/overlap rejection if requested, and the minimum
        length fraction of the requested window.
        """
        if len(stream) == 0:
            return False, REASON_NO_WAVEFORM
        if self.restrictions.reject_channels_with_gaps and len(stream) > 1:
            return False, REASON_WAVEFORM_QC
        if self.restrictions.minimum_length:
            duration = sum(tr.stats.endtime - tr.stats.starttime
                           for tr in stream)
            expected = self.restrictions.minimum_length * (
                record.endtime - record.starttime)
            if duration < expected:
                return False, REASON_WAVEFORM_QC
        return True, None

    # ------------------------------------------------------------------ #
    # StationXML downloading, merging and verification
    # ------------------------------------------------------------------ #
    def _download_and_verify_stationxml(self, client_name, client,
                                        downloaded):
        # Group delivered windows by station.
        by_station = {}
        for record, mseed_path, stream in downloaded:
            by_station.setdefault(record.station_key, []).append(
                (record, mseed_path, stream))

        for station_key, items in by_station.items():
            state = self._stations[station_key]
            channel_times = self._stationxml_bulk_ranges(items)
            bulk = []
            for (loc, cha), (start, end) in sorted(
                    channel_times.items()):
                bulk.append((state.network, state.station, loc, cha,
                             start, end))

            tmp = tempfile.NamedTemporaryFile(
                prefix="obspy_backfill_stationxml_", suffix=".xml",
                delete=False)
            tmp.close()
            try:
                result = utils.download_stationxml(
                    client, client_name, bulk, tmp.name,
                    logger=self.logger)
                if result is None or not os.path.isfile(tmp.name) or \
                        os.path.getsize(tmp.name) == 0:
                    self._fail_items(items, client_name,
                                     REASON_EMPTY_STATIONXML)
                    continue
                try:
                    candidate = obspy.read_inventory(tmp.name)
                except Exception:
                    self._fail_items(items, client_name,
                                     REASON_EMPTY_STATIONXML)
                    continue
                if len(candidate.networks) == 0 or \
                        sum(len(sta.channels) for net in candidate
                            for sta in net) == 0:
                    self._fail_items(items, client_name,
                                     REASON_EMPTY_STATIONXML)
                    continue

                target = self._resolve_stationxml_path(state)
                if target is None:
                    self._fail_items(items, client_name,
                                     REASON_WRITE_FAILED)
                    continue

                # Merge with whatever already exists, rejecting the new
                # windows on conflicts without touching the old file.
                merged, conflicts = self._merge_inventory(state, candidate,
                                                           items)
                if conflicts:
                    self._fail_items(items, client_name,
                                     REASON_RESPONSE_CONFLICT)
                    continue

                # Verify every window of the station against the merged
                # inventory: new samples must find their source responses,
                # old accepted windows must keep theirs.
                failed = self._verify_all_windows(state, merged, items,
                                                  client_name)
                if failed:
                    # Remove the unverifiable waveforms so they cannot be
                    # mistaken for reusable local pairs at the next center.
                    for item in items:
                        record, mseed_path, _stream = item
                        if record.window_id in failed and mseed_path and \
                                os.path.exists(mseed_path):
                            try:
                                utils.safe_delete(mseed_path)
                            except Exception:
                                pass
                    continue

                # Atomic replace; keep the old file if anything goes wrong.
                if not self._atomic_write_inventory(merged, target, state):
                    self._fail_items(items, client_name,
                                     REASON_WRITE_FAILED)
                    continue

                state.inventory = merged
                state.stationxml_filename = target
                self.report.stationxml_files[station_key] = target
                for record, mseed_path, _stream in items:
                    record.status = BACKFILL_STATUS.downloaded
                    record.provider = client_name
                    record.mseed_filename = mseed_path
                    record.stationxml_filename = target
            finally:
                if os.path.exists(tmp.name):
                    utils.safe_delete(tmp.name)

    def _stationxml_bulk_ranges(self, items):
        """
        Compute one ``(start, end)`` range per ``(location, channel)`` for
        the StationXML bulk request. The ranges follow the actual sample
        times read back from the delivered waveforms (not the nominal
        request windows) and adjacent/overlapping windows of the same
        channel are merged, so that a multi-hour continuous stretch results
        in one StationXML request rather than one per window.
        """
        ranges = {}
        for record, _mseed_path, stream in items:
            loc, cha = record.window_id[2], record.window_id[3]
            sample_start, sample_end = _stream_sample_bounds(stream)
            if sample_start is None:
                continue
            if (loc, cha) not in ranges:
                ranges[(loc, cha)] = [sample_start, sample_end]
            else:
                cur = ranges[(loc, cha)]
                cur[0] = min(cur[0], sample_start)
                cur[1] = max(cur[1], sample_end)
        return {key: tuple(value) for key, value in ranges.items()}

    def _resolve_stationxml_path(self, state):
        """
        Determine the single StationXML filename kept for a station,
        honoring folder paths, templates and callback return values.
        """
        channels = sorted(state.channels)
        starts = [r.starttime for r in
                  self._records_of_station(state.key)]
        ends = [r.endtime for r in
                self._records_of_station(state.key)]
        if not starts:
            return None
        storage = utils.get_stationxml_filename(
            self.stationxml_storage, state.network, state.station,
            channels, min(starts), max(ends))
        if isinstance(storage, (str, bytes)):
            return storage
        # Dictionary callback form.
        return storage["filename"]

    def _merge_inventory(self, state, candidate, items):
        """
        Merge ``candidate`` into the inventory already accepted for the
        station.

        Epoch validity periods are preserved, and conflicts are judged only
        at the actual sample times of the newly delivered windows: if such a
        sample is covered by a candidate epoch and by an already accepted
        epoch carrying a different response, the affected windows are
        rejected and the on-disk station file is left untouched. Identical
        responses are merged harmlessly, and adjacent non-conflicting epochs
        have their touching bounds aligned so a boundary sample is covered
        by exactly one epoch.

        Returns ``(merged_inventory, conflicts)`` where ``conflicts`` maps
        window ids to reason strings. Empty conflicts mean the merge is
        usable and must still pass the full window verification.
        """
        if state.inventory is not None:
            merged = copy.deepcopy(state.inventory)
        else:
            merged = obspy.Inventory(networks=[], source="ObsPy backfill")

        # Actual sample-time ranges of every new window per channel.
        sample_ranges = collections.defaultdict(list)
        for record, _mseed_path, stream in items:
            loc, cha = record.window_id[2], record.window_id[3]
            for trace in stream:
                if trace.stats.npts <= 0:
                    continue
                rates = [trace.stats.sampling_rate] if \
                    trace.stats.sampling_rate else []
                tolerance = (0.5 / min(rates)) if rates else 0.0
                sample_ranges[(loc, cha)].append(
                    (record.window_id,
                     trace.stats.starttime,
                     trace.stats.starttime +
                     (trace.stats.npts - 1) / trace.stats.sampling_rate,
                     tolerance, trace.stats.npts,
                     trace.stats.sampling_rate))

        conflicts = {}

        for net in candidate:
            for sta in net:
                if (net.code, sta.code) != state.key:
                    continue
                m_net = _get_or_create_network(merged, net)
                m_sta = _get_or_create_station(m_net, sta)
                for channel in sta.channels:
                    key = (channel.location_code, channel.code)
                    if key not in sample_ranges:
                        continue
                    affected, reason = self._merge_channel(
                        m_sta, channel, sample_ranges[key])
                    if reason is not None:
                        for window_id in affected:
                            conflicts.setdefault(window_id, reason)
        return merged, conflicts

    def _merge_channel(self, m_sta, channel, window_ranges):
        """
        Merge a candidate channel epoch into the station without shortening
        its validity period.

        The full epoch (including an open end) is preserved, because an
        accepted epoch must keep backing its already accepted windows and a
        later candidate has to be judged against that full validity. A
        conflict is detected only at the actual sample times of the new
        windows: if such a sample is covered by the candidate and by an
        accepted epoch carrying a different response, the affected windows
        are rejected.

        :returns: ``(affected_window_ids, reason)``; ``reason`` is None on
            success, otherwise the windows that could not be merged and the
            reason.
        """
        new_fp = _response_fingerprint(channel.response)
        far_future = obspy.UTCDateTime(2599, 1, 1)

        # The windows this candidate epoch actually backs.
        affected = []
        for (window_id, sample_start, sample_end, _tol, _n,
             _sr) in window_ranges:
            if _epoch_covers_time(channel.start_date, channel.end_date,
                                  sample_start) or \
                    _epoch_covers_time(channel.start_date,
                                       channel.end_date, sample_end):
                affected.append(window_id)
        if not affected:
            return [], None

        # Conflict check at the actual new sample times against the already
        # accepted, fully preserved epochs. Boundary samples (a sample on an
        # epoch's end date) belong to the epoch starting there and are not an
        # interior sample, so adjacent epochs with different responses do not
        # conflict.
        for existing in m_sta.channels:
            if existing.location_code != channel.location_code or \
                    existing.code != channel.code:
                continue
            if _response_fingerprint(existing.response) == new_fp:
                # Same instrument: overlapping documentation is harmless.
                continue
            ex_end = existing.end_date if existing.end_date is not None \
                else far_future
            for (_wid, sample_start, _sample_end, tol, npts,
                 sr) in window_ranges:
                if _any_sample_strictly_inside(
                        sample_start, npts, sr, existing.start_date,
                        ex_end, tol):
                    return affected, REASON_RESPONSE_CONFLICT

        # No conflicting sample: keep the candidate epoch. Before storing,
        # align its bounds with accepted epochs of the same channel so that
        # adjacent epochs do not both cover the shared boundary sample. An
        # accepted epoch ending at time T keeps T; the new one starts at T.
        candidate_channel = copy.deepcopy(channel)
        candidate_channel.start_date, candidate_channel.end_date = \
            self._align_epoch_bounds(m_sta, candidate_channel, window_ranges)
        self._append_epoch_if_new(m_sta, candidate_channel)
        return affected, None

    def _align_epoch_bounds(self, m_sta, channel, window_ranges):
        """
        Adjust a non-conflicting candidate epoch so it does not double-cover
        a boundary sample of an adjacent accepted epoch.

        If an accepted epoch ends at time T and the candidate starts before
        T but its new samples only begin at/after T, the candidate start is
        moved to T. The symmetric case applies at the candidate's end. Epoch
        validity beyond the actual samples is otherwise preserved.
        """
        new_start = channel.start_date
        new_end = channel.end_date
        sample_times = [(s, e) for _wid, s, e, _tol, _n, _sr
                        in window_ranges]
        if not sample_times:
            return new_start, new_end
        min_sample = min(s for s, _e in sample_times)
        max_sample = max(e for _s, e in sample_times)

        for existing in m_sta.channels:
            if existing.location_code != channel.location_code or \
                    existing.code != channel.code:
                continue
            # Accepted epoch ending exactly where the new samples begin.
            if existing.end_date is not None and \
                    existing.start_date is not None and \
                    existing.end_date <= min_sample and \
                    (new_start is None or new_start < existing.end_date):
                new_start = max(
                    new_start if new_start is not None else min_sample,
                    existing.end_date)
            # Accepted epoch starting exactly where the new samples end.
            if existing.start_date is not None and \
                    existing.start_date >= max_sample and \
                    (new_end is None or new_end > existing.start_date):
                new_end = min(
                    new_end if new_end is not None
                    else existing.start_date,
                    existing.start_date)
        return new_start, new_end

    @staticmethod
    def _append_epoch_if_new(m_sta, channel):
        new_fp = _response_fingerprint(channel.response)
        for existing in m_sta.channels:
            if existing.location_code == channel.location_code and \
                    existing.code == channel.code and \
                    existing.start_date == channel.start_date and \
                    existing.end_date == channel.end_date and \
                    _response_fingerprint(existing.response) == new_fp:
                return
        m_sta.channels.append(copy.deepcopy(channel))

    def _verify_all_windows(self, state, inventory, new_items, client_name):
        """
        Verify the new windows and every already accepted window of the
        station against ``inventory``. Returns a dict of failed window
        ids -> reason (empty means all good).
        """
        failed = {}
        # New windows: use the freshly read streams.
        for record, _mseed_path, stream in new_items:
            ok, reason = self._verify_window_against_inventory(
                record, stream, inventory)
            if not ok:
                self._record_failure(record, client_name, reason)
                failed[record.window_id] = reason

        # Previously accepted windows must remain verifiable.
        for record in self.report.windows.values():
            if record.station_key != state.key:
                continue
            if record.status not in (BACKFILL_STATUS.downloaded,
                                     BACKFILL_STATUS.reused):
                continue
            if record.window_id in failed:
                continue
            try:
                old_stream = obspy.read(record.mseed_filename,
                                        headonly=True)
            except Exception:
                reason = REASON_NO_LOCAL_PAIR
                failed[record.window_id] = reason
                continue
            ok, reason = self._verify_window_against_inventory(
                record, old_stream, inventory)
            if not ok:
                failed[record.window_id] = reason
        return failed

    def _verify_window_against_inventory(self, record, stream, inventory):
        """
        Check that every trace's full SEED id matches the window and that
        each actual sample time (as read back from the MiniSEED) is covered
        by exactly one non-conflicting response epoch in the inventory.
        """
        net, sta, loc, cha, win_start, win_end = record.window_id
        for trace in stream:
            parts = trace.id.split(".")
            if len(parts) != 4 or tuple(parts) != (net, sta, loc, cha):
                return False, REASON_RESPONSE_MISMATCH
            epochs = _response_epochs_for(inventory, net, sta, loc, cha)
            if epochs is None:
                return False, REASON_RESPONSE_MISMATCH
            ok, reason = _samples_covered(trace, epochs)
            if not ok:
                return False, reason
        return True, None

    def _atomic_write_inventory(self, inventory, filename, state):
        directory = os.path.dirname(filename)
        tmp_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".tmp_stationxml_", suffix=".xml", dir=directory)
            os.close(fd)
            inventory.write(tmp_path, format="stationxml", validate=True)
            # Make sure the replacement itself parses before swapping.
            obspy.read_inventory(tmp_path)
            os.replace(tmp_path, filename)
            return True
        except Exception as e:
            self.logger.error(
                "Backfill: failed to write merged StationXML '%s': %s" %
                (filename, str(e)))
            if tmp_path is not None:
                try:
                    utils.safe_delete(tmp_path)
                except Exception:
                    pass
            return False

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _record_failure(self, record, client_name, reason):
        if client_name not in record.attempts:
            record.attempts.append(client_name)
        record.failure_reasons[client_name] = reason
        record.status = BACKFILL_STATUS.pending
        self.logger.info(
            "Backfill: window %s (%s) rejected at '%s': %s" % (
                record.seed_id, record.starttime, client_name, reason))

    def _fail_items(self, items, client_name, reason):
        for record, mseed_path, _stream in items:
            self._record_failure(record, client_name, reason)
            # A waveform without a verifiable response is not a success and
            # must not be left behind for a later data center to mistake for
            # a reusable local pair.
            if mseed_path and os.path.exists(mseed_path):
                try:
                    utils.safe_delete(mseed_path)
                except Exception:
                    pass
            record.mseed_filename = mseed_path

    def _finalize(self):
        for record in self.report.windows.values():
            if record.status == BACKFILL_STATUS.pending:
                record.status = BACKFILL_STATUS.missing
                if not record.failure_reasons:
                    record.failure_reasons["__none__"] = \
                        REASON_NOT_OFFERED


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
class _DistanceReferenceHelper(object):
    """
    Minimal object exposing the interface used by
    :meth:`ClientDownloadHelper.filter_stations_based_on_minimum_distance`.
    """
    def __init__(self, restrictions, logger, stations):
        self.restrictions = restrictions
        self.logger = logger
        self.stations = dict(stations)

    def filter_stations_based_on_minimum_distance(self, existing_helpers):
        # Delegate to the real implementation by composition; the method
        # only accesses restrictions, logger and stations.
        from .download_helpers import ClientDownloadHelper
        # Build a real helper without invoking its __init__ (we do not have
        # a client here and do not need one).
        shim = ClientDownloadHelper.__new__(ClientDownloadHelper)
        shim.restrictions = self.restrictions
        shim.logger = self.logger
        shim.stations = dict(self.stations)
        rejected = shim.filter_stations_based_on_minimum_distance(
            existing_helpers)
        self.stations = shim.stations
        return rejected


def _station_objects_for_states(states):
    """
    Build lightweight Station-like objects for already fixed stations so
    they can serve as distance-filter reference points.
    """
    from .download_helpers import Station
    stations = {}
    for state in states:
        stations[state.key] = Station(
            network=state.network, station=state.station,
            latitude=state.latitude, longitude=state.longitude,
            channels=[])
    return stations


def _get_or_create_network(inventory, network):
    for net in inventory.networks:
        if net.code == network.code:
            return net
    new_net = obspy.core.inventory.Network(
        code=network.code, stations=[],
        description=network.description)
    inventory.networks.append(new_net)
    return new_net


def _get_or_create_station(network, station):
    for sta in network.stations:
        if sta.code == station.code:
            return sta
    new_sta = obspy.core.inventory.Station(
        code=station.code, latitude=station.latitude,
        longitude=station.longitude, elevation=station.elevation)
    network.stations.append(new_sta)
    return new_sta


def _epoch_covers_time(start, end, time):
    """True if a channel epoch with the given (possibly open) bounds covers
    ``time``."""
    if start is not None and time < start:
        return False
    if end is not None and time > end:
        return False
    return True


def _any_sample_strictly_inside(start, npts, sampling_rate, epoch_start,
                                epoch_end, tolerance=0.0):
    """
    Walk the actual sample times and return True if at least one of them
    falls strictly inside the epoch, i.e. ``epoch_start < t < epoch_end``.

    A sample exactly on an epoch boundary is assigned to the epoch starting
    there and thus never counts as strictly inside another epoch. Sample
    times reconstructed by ObsPy lie on a regular grid, so the comparison is
    exact for boundary samples; ``tolerance`` (half a sample interval) is only
    applied to skip samples still before the epoch start.
    """
    if npts <= 0 or not sampling_rate:
        return False
    first = start
    last = start + (npts - 1) / sampling_rate
    if epoch_start is not None:
        # Index of the first sample at or after the epoch start.
        skip = int(round((epoch_start - first) * sampling_rate))
        if first + skip / sampling_rate < epoch_start - tolerance:
            skip += 1
        skip = max(0, min(skip, npts - 1))
    else:
        skip = 0
    for index in range(skip, npts):
        t = first + index / sampling_rate
        if t > last:
            break
        if epoch_start is not None and t <= epoch_start:
            # On the start boundary: belongs to this epoch but is not an
            # interior point, hence cannot conflict with an epoch ending
            # there.
            continue
        if t >= epoch_end:
            # At or past the end: this and all later samples belong to a
            # following epoch.
            return False
        # Strictly inside: epoch_start < t < epoch_end.
        return True
    return False


#: Rough sampling-rate guess per band code, mirroring the estimate used by the
#: normal download path's bulk chunking.
_BAND_SAMPLING_RATE = {
    "F": 5000, "G": 5000, "D": 1000, "C": 1000, "E": 250, "S": 80,
    "H": 250, "B": 80, "M": 10, "L": 1, "V": 0.1, "U": 0.01,
    "R": 0.001, "P": 0.0001, "T": 0.00001, "Q": 0.000001, "A": 5000,
    "O": 5000}


def _split_chunks_by_size(chunks, chunk_size_in_mb, max_chunk_length=50):
    """
    Split a list of MiniSEED request chunks into bounded batches.

    The size estimate assumes four bytes per sample and a STEIM compression
    factor of about three, exactly like
    :meth:`ClientDownloadHelper.download_mseed`. Each returned batch stays
    below ``chunk_size_in_mb`` megabytes (estimated) and contains at most
    ``max_chunk_length`` entries.
    """
    if not chunks:
        return []
    batches = []
    current = []
    current_mb = 0.0
    for chunk in chunks:
        _net, _sta, _loc, channel, start, end, _fn = chunk
        try:
            rate = _BAND_SAMPLING_RATE[channel[0].upper()]
        except (IndexError, KeyError):
            rate = 1.0
        estimated_mb = rate * (end - start) * 4.0 / 3.0 / 1024.0 / 1024.0
        if current and (current_mb + estimated_mb >= chunk_size_in_mb or
                        len(current) >= max_chunk_length):
            batches.append(current)
            current = []
            current_mb = 0.0
        current.append(chunk)
        current_mb += estimated_mb
    if current:
        batches.append(current)
    return batches


def _stream_sample_bounds(stream):
    """
    Return ``(first_sample_time, last_sample_time)`` across all traces of a
    waveform read back from disk, or ``(None, None)`` if there is no data.
    """
    starts = []
    ends = []
    for trace in stream:
        if trace.stats.npts <= 0:
            continue
        starts.append(trace.stats.starttime)
        ends.append(trace.stats.starttime +
                    (trace.stats.npts - 1) / trace.stats.sampling_rate)
    if not starts:
        return None, None
    return min(starts), max(ends)


def _response_epochs_for(inventory, network, station, location, channel):
    """
    Return the sorted list of ``(start, end_or_None, response_fingerprint)``
    response epochs for one channel. Returns None if the channel id does not
    exist in the inventory.
    """
    epochs = []
    found = False
    for net in inventory:
        if net.code != network:
            continue
        for sta in net:
            if sta.code != station:
                continue
            for cha in sta.channels:
                if cha.code != channel or \
                        cha.location_code != location:
                    continue
                found = True
                if cha.response is None:
                    continue
                epochs.append((cha.start_date, cha.end_date,
                               _response_fingerprint(cha.response)))
    if not found:
        return None
    epochs.sort(key=lambda e: (e[0] is None, e[0]))
    return epochs


def _samples_covered(trace, epochs):
    """
    Check that every sample time of ``trace`` is covered by at least one
    response epoch and never by two conflicting epochs.

    Boundaries are judged on the actual sample times read back from the
    MiniSEED and the response validity dates read back from the StationXML.
    An epoch covers a sample if ``start_date <= t < end_date`` (an open
    ``end_date`` covers every later sample); a sample landing exactly on the
    boundary between two adjacent epochs thus belongs to the epoch starting
    there, never to both. Floating-point roundoff in the reconstructed
    sample times is absorbed by comparing on the epoch side only: a sample
    an epsilon before an epoch start still counts, while a sample an epsilon
    past an epoch end is handed to the next epoch rather than declared a
    gap.
    """
    npts = trace.stats.npts
    sr = trace.stats.sampling_rate
    start = trace.stats.starttime
    far_future = obspy.UTCDateTime(2599, 1, 1)
    # Tolerance: half a sample interval guards against floating point
    # roundoff in the reconstructed sample times.
    eps = (0.5 / sr) if sr else 0.0

    for index in range(npts):
        t = start + index / sr
        covering = []
        strict_covering = []
        for ep_start, ep_end, fp in epochs:
            lo = ep_start if ep_start is not None else None
            hi = ep_end if ep_end is not None else far_future
            # Gap/coverage check with a half-sample tolerance on both sides:
            # a sample landing within half a sample of an epoch boundary is
            # still considered covered by that epoch. This keeps the last
            # sample of a window ending exactly on an epoch end date covered.
            if (lo is None or t >= lo - eps) and t <= hi + eps:
                covering.append(fp)
            # Strict end-exclusive coverage used for the conflict decision: a
            # sample on an epoch end boundary belongs to the next epoch and
            # must never count as double-covered.
            if (lo is None or t >= lo) and t < hi:
                strict_covering.append(fp)
        if not covering:
            return False, REASON_RESPONSE_GAP
        # Two distinct responses both strictly valid at the same sample time
        # are a genuine conflict.
        if len(set(strict_covering)) > 1:
            return False, REASON_RESPONSE_CONFLICT
    return True, None
