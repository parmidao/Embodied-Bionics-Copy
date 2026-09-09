# Cardiac Signal Processing — Methods Text

Draft paragraphs for the Methods section, covering (1) the decision to derive
cardiac features from RR intervals without raw ECG, and (2) the artifact
correction applied to the RR series. Written to be defensible to a reviewer.

---

## Cardiac data acquisition and the omission of raw ECG

Cardiac activity was recorded with a Polar H10 chest strap, which performs
on-device R-peak detection and streams inter-beat (RR) intervals in addition to
a raw electrocardiogram. Cardiac workload features in this study are drawn
entirely from the RR interval series. Raw ECG was not used, for two reasons.
First, all cardiac features employed here are time-domain heart-rate-variability
(HRV) measures — mean heart rate, SDNN, RMSSD, and pNN50 — each of which is
computed directly from the sequence of inter-beat intervals and does not require
the underlying waveform. Second, the Polar H10's on-device beat detection
provides RR intervals of sufficient quality for time-domain HRV, a configuration
widely used in ambulatory workload and stress research where raw ECG is
impractical or unavailable. Frequency-domain HRV (e.g. LF/HF), which benefits
more from waveform-level access and requires longer stationary segments than the
present windowing affords, was not computed and is left to future work.

The principal trade-off of an RR-only configuration is that beat-detection
artifacts cannot be verified against the source waveform and must instead be
identified from the interval series itself. The correction procedure below was
adopted specifically to address this, and its performance was validated against
the artifact patterns observed in pilot recordings.

---

## RR interval artifact correction

Inter-beat interval series are subject to characteristic beat-detection
artifacts: a missed beat produces an interval approximately two (or three) times
the surrounding intervals, while a spurious detection produces a pair of short
intervals whose sum approximates one normal interval. Because RMSSD is defined
on the squared differences between successive intervals, a single such artifact
can inflate the metric several-fold within a short analysis window; correction
is therefore a prerequisite for any RR-derived HRV feature.

Artifacts were identified using a local-median criterion. Each interval was
first subjected to a hard physiological gate (intervals outside 300–2000 ms,
corresponding to 30–200 bpm, were treated as non-physiological). Each remaining
interval was then compared to the median of a five-interval neighbourhood
centred on it (excluding the interval itself); intervals deviating from this
local median by more than 20% were flagged as artifacts. Flagged intervals were
replaced by the local median of their valid neighbours, preserving the length
and time base of the series. This local-median approach is a standard method for
RR artifact correction and, unlike a fixed global range filter, adapts to slow
changes in heart rate over the recording.

To prevent windows with pervasive artifacts from contributing spurious feature
values, the fraction of intervals flagged in each analysis window was retained
as a quality indicator. Windows in which more than 30% of intervals required
correction were treated as unreliable and excluded from HRV feature computation
rather than reported as corrected values, since heavy correction can yield
plausible-looking but untrustworthy estimates.

---

## Validation note (for internal records, not necessarily the paper)

The correction procedure was validated against the specific artifacts observed
in pilot RR recordings. In a representative test series at approximately 97 bpm
(mean RR ≈ 620 ms) with four inserted artifacts matching the pilot's patterns
(a doubled interval at 1290 ms, a tripled interval at 2170 ms, and a
short-interval pair at 400 ms and 830 ms), uncorrected RMSSD was inflated to
approximately 390 ms, roughly sixteen times the value of the underlying clean
series (≈24 ms). After correction, RMSSD returned to approximately 18 ms, close
to the clean reference, and only the four inserted artifacts were flagged. This
confirms that the procedure recovers physiologically meaningful HRV from RR
series containing the beat-detection errors characteristic of this hardware.
