"""
delsys_lsl.py — minimal LSL bridge for the Delsys demo.
Streams the EMG channels into an LSL outlet named 'DelsysEMG'.
Self-initializes on the first data poll using the TrignoBase object.

RATE NOTE (2026-08-05)
----------------------
The July BBT pilot recordings declared a nominal rate of 1111.111 Hz while
delivering ~1248-1249 Hz, consistently across all six files. A wrong declared
rate matters: downstream feature extraction (filter design, spectral
features, window lengths) uses it, so a 12% error propagates into every
EMG feature.

1111.111 Hz and 1259.259 Hz are both valid Trigno rates, so the likely cause
is _find_emg_rate reading the rate of a non-EMG channel. Set
EMG_RATE_OVERRIDE below once the true rate is confirmed in the Delsys
configuration. Until then, the runtime monitor prints the measured rate so a
mismatch is visible during the session rather than during analysis.
"""
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet

# Set this once the true EMG rate is confirmed from the Delsys configuration
# (e.g. 1259.259). Leave as None to use whatever the API reports.
EMG_RATE_OVERRIDE = None

# Warn if measured throughput differs from the declared rate by more than this.
RATE_TOLERANCE = 0.02          # 2%
RATE_REPORT_INTERVAL_S = 10.0  # how often the monitor prints

# Your sensor -> muscle mapping, in EMG-channel order.
# If this count doesn't match the detected EMG channels, generic names are used.
CHANNEL_NAMES = [
    "S1_FlexorGrasp",
    "S2_ExtensorOpen",
    "S3_Pronation",
    "S4_Supination",
    "S5_ThumbPinch2",
    "S26_ThumbPinch1_Mini",
]


def _channel_rates(trigno_base, emg_idx):
    """
    Read the sample rate of EVERY EMG channel, not just the first.

    The original version took chobjs[emg_idx[0]].sample_rate. If channel 0 is
    not actually an EMG channel, or if Avanti and Mini sensors report
    different rates, that single read is silently wrong for the rest.
    """
    rates = {}

    chobjs = getattr(trigno_base, "channelobjects", None)
    if chobjs:
        for i in emg_idx:
            try:
                rates[i] = float(chobjs[i].sample_rate)
            except Exception:
                pass
    if rates:
        return rates

    for attr in ("SampleRates", "sampleRates", "samplerates", "sample_rates"):
        arr = getattr(trigno_base, attr, None)
        if arr is None:
            continue
        for i in emg_idx:
            try:
                rates[i] = float(arr[i])
            except Exception:
                pass
        if rates:
            return rates

    return rates


def _resolve_emg_rate(trigno_base, emg_idx):
    """
    Decide the rate to declare, and say out loud how it was decided.
    Returns (rate, per_channel_rates).
    """
    rates = _channel_rates(trigno_base, emg_idx)

    if EMG_RATE_OVERRIDE is not None:
        if rates:
            reported = sorted(set(rates.values()))
            if not any(abs(r - EMG_RATE_OVERRIDE) < 0.01 for r in reported):
                print(f"[LSL] NOTE: override {EMG_RATE_OVERRIDE} Hz differs from "
                      f"API-reported {reported}")
        return float(EMG_RATE_OVERRIDE), rates

    if not rates:
        print("[LSL] WARNING: could not read any EMG sample rate; declaring 0.0 "
              "(irregular). Downstream features will have no rate to work from.")
        return 0.0, rates

    distinct = sorted(set(rates.values()))
    if len(distinct) > 1:
        print(f"[LSL] WARNING: EMG channels report different rates: {rates}")
        print(f"[LSL]          declaring the most common; set EMG_RATE_OVERRIDE "
              f"to resolve this properly.")
        most_common = max(distinct, key=lambda r: list(rates.values()).count(r))
        return float(most_common), rates

    return float(distinct[0]), rates


class EMGStreamer:
    def __init__(self):
        self.outlet = None
        self.emg_idx = None
        self.initialized = False
        self.declared_rate = 0.0
        self._samples_pushed = 0
        self._t0 = None
        self._last_report = None
        self._rate_warned = False

    def _init(self, trigno_base, outArr):
        emg_idx = list(getattr(trigno_base, "emgChannelsIdx", []) or [])
        if not emg_idx:                       # fallback: stream all channels
            emg_idx = list(range(len(outArr)))
            print("[LSL] WARNING: emgChannelsIdx was empty; streaming ALL "
                  "channels. Verify that non-EMG channels are not included.")
        self.emg_idx = emg_idx

        srate, per_channel = _resolve_emg_rate(trigno_base, emg_idx)
        self.declared_rate = srate

        n_ch = len(emg_idx)
        labels = CHANNEL_NAMES if len(CHANNEL_NAMES) == n_ch \
                 else [f"EMG_{i}" for i in range(n_ch)]
        if labels is not CHANNEL_NAMES:
            print(f"[LSL] WARNING: CHANNEL_NAMES has {len(CHANNEL_NAMES)} entries "
                  f"but {n_ch} EMG channels were detected; using generic labels. "
                  f"Montage attribution will not be recoverable from the file.")

        info = StreamInfo("DelsysEMG", "EMG", n_ch, srate, "float32", "delsys_real")
        chns = info.desc().append_child("channels")
        for name in labels:
            c = chns.append_child("channel")
            c.append_child_value("label", name)
            c.append_child_value("unit", "volts")
            c.append_child_value("type", "EMG")

        # Record how the rate was arrived at, so the file is self-documenting.
        acq = info.desc().append_child("acquisition")
        acq.append_child_value("rate_source",
                               "override" if EMG_RATE_OVERRIDE is not None else "api")
        acq.append_child_value("api_reported_rates",
                               ",".join(f"{i}:{r}" for i, r in sorted(per_channel.items()))
                               or "unavailable")

        self.outlet = StreamOutlet(info)
        self.initialized = True

        print("\n[LSL] DelsysEMG outlet created")
        print(f"[LSL]   total channels detected : {len(outArr)}")
        print(f"[LSL]   EMG channel indices      : {emg_idx}")
        print(f"[LSL]   EMG channel count        : {n_ch}")
        print(f"[LSL]   per-channel rates (API)  : {per_channel or 'unavailable'}")
        print(f"[LSL]   declared sample rate     : {srate}  (0.0 = irregular)")
        print(f"[LSL]   rate source              : "
              f"{'EMG_RATE_OVERRIDE' if EMG_RATE_OVERRIDE is not None else 'API'}")
        print(f"[LSL]   labels                   : {labels}\n")

    def _monitor_rate(self, n_new):
        """
        Compare actual throughput against the declared rate.

        A stream that declares one rate and delivers another looks perfectly
        healthy in LabRecorder. This makes the discrepancy visible while the
        session is still running.
        """
        now = time.time()
        if self._t0 is None:
            self._t0 = now
            self._last_report = now
            return

        self._samples_pushed += n_new
        elapsed = now - self._t0
        if elapsed <= 0 or now - self._last_report < RATE_REPORT_INTERVAL_S:
            return

        measured = self._samples_pushed / elapsed
        self._last_report = now

        if self.declared_rate > 0:
            drift = abs(measured - self.declared_rate) / self.declared_rate
            flag = "  <-- MISMATCH" if drift > RATE_TOLERANCE else ""
            print(f"[LSL] throughput {measured:8.2f} Hz  vs declared "
                  f"{self.declared_rate:8.2f} Hz  ({drift:+.1%}){flag}")
            if drift > RATE_TOLERANCE and not self._rate_warned:
                self._rate_warned = True
                print(f"[LSL] The declared rate is wrong by {drift:.1%}. Every "
                      f"EMG feature computed from this file will inherit that "
                      f"error. Confirm the true rate in the Delsys "
                      f"configuration and set EMG_RATE_OVERRIDE.")
        else:
            print(f"[LSL] throughput {measured:8.2f} Hz  (no rate declared)")

    def push(self, trigno_base, outArr):
        if outArr is None:
            return
        if not self.initialized:
            try:
                self._init(trigno_base, outArr)
            except Exception as e:
                print("[LSL] init failed:", e)
                return
        try:
            chans = [np.asarray(outArr[i][0], dtype=np.float32) for i in self.emg_idx]
            n = min((len(c) for c in chans), default=0)
            if n > 0:
                block = np.column_stack([c[:n] for c in chans])  # (n_samples, n_ch)
                self.outlet.push_chunk(block.tolist())
                self._monitor_rate(n)
        except Exception as e:
            print("[LSL] push failed:", e)