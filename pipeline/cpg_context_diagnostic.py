"""Replay real MANC sensory traces to isolate the size-reference assumption.

The original author graph is unchanged. This is a firing-rate model experiment
on MANC, not the MaleCNS animal used by the live service. Frozen feedback replay
is a controlled neural experiment, not a newly closed body loop or walking claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import find_peaks

from cpg_rate import RateNetwork

ROOT = Path(__file__).resolve().parents[1]
CORE_TYPES = ['DNg100', 'IN17A001', 'INXXX466', 'IN16B036']
CONDITIONS = {
    'native_replay': dict(reference='whole', feedback=True, silence=[]),
    'front_reference_replay': dict(reference='front', feedback=True, silence=[]),
    'native_no_feedback': dict(reference='whole', feedback=False, silence=[]),
    'front_reference_no_feedback': dict(reference='front', feedback=False, silence=[]),
    'front_reference_silence_LF_E1': dict(reference='front', feedback=True, silence=[10707]),
    'front_reference_silence_LF_E2': dict(reference='front', feedback=True, silence=[11751]),
    'front_reference_front_region': dict(reference='front', feedback=False, silence=[], front_only=True),
    'front_reference_front_region_silence_E1': dict(reference='front', feedback=False, silence=[10707], front_only=True),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rhythm(series, dt=.001):
    # Every condition uses the same analysis criteria, without frequency tuning.
    x = np.asarray(series, float)
    amplitude = float(np.ptp(x))
    prominence = max(.5, .1 * amplitude)
    peaks, _ = find_peaks(x, prominence=prominence, distance=10)
    periods = np.diff(peaks) * dt
    cv = float(np.std(periods) / np.mean(periods)) if len(periods) else None
    first = float(np.ptp(x[:len(x)//3]))
    last = float(np.ptp(x[-len(x)//3:]))
    sustained = (amplitude >= 2 and len(peaks) >= 5 and cv is not None
                 and cv <= .2 and last >= .5 * max(first, 1e-12))
    return dict(mean_hz=float(x.mean()), amplitude_hz=amplitude,
                peaks=len(peaks), period_cv=cv,
                frequency_hz=float(1 / np.median(periods)) if len(periods) else None,
                first_third_amplitude_hz=first, last_third_amplitude_hz=last,
                diagnostic_sustained_rhythm=bool(sustained))


def compare_pair(positive, negative):
    p, q = np.asarray(positive), np.asarray(negative)
    pr, qr = rhythm(p), rhythm(q)
    net = p - q
    state = np.where(net > .5, 1, np.where(net < -.5, -1, 0))
    previous = 0
    switches = 0
    start = 0
    for end in range(1, len(state) + 1):
        if end == len(state) or state[end] != state[start]:
            if end - start >= 10 and state[start] != 0:
                if previous and previous != state[start]:
                    switches += 1
                previous = state[start]
            start = end
    frequency_compatible = False
    phase = None
    if pr['diagnostic_sustained_rhythm'] and qr['diagnostic_sustained_rhythm']:
        frequency_compatible = abs(pr['frequency_hz'] / qr['frequency_hz'] - 1) <= .1
        if frequency_compatible:
            freq = (pr['frequency_hz'] + qr['frequency_hz']) / 2
            basis = np.exp(-2j * np.pi * freq * np.arange(len(p)) * .001)
            a, b = (p - p.mean()) @ basis, (q - q.mean()) @ basis
            phase = float(np.degrees(np.angle(b * np.conj(a))))
    return dict(positive=pr, negative=qr,
                net_mean_hz=float(net.mean()), net_min_hz=float(net.min()),
                net_max_hz=float(net.max()), dominance_reversals_at_least_10ms=switches,
                positive_dominance_fraction=float((net > .5).mean()),
                negative_dominance_fraction=float((net < -.5).mean()),
                common_frequency=bool(frequency_compatible), phase_offset_degrees=phase,
                alternating_neural_dominance=bool(frequency_compatible and switches >= 4))


def run_one(name, spec, n, w, old, sensor_ids, reference, out, front_ids):
    b = RateNetwork(w, n['size'], seed=1, dt=.0005, size_reference=reference)
    dl = np.flatnonzero(n.type.eq('DNg100'))
    motors = np.flatnonzero(n['class'].eq('motor neuron'))
    cores = np.flatnonzero(n.type.isin(CORE_TYPES))
    recorded = np.unique(np.r_[motors, cores])
    lookup = {int(v): i for i, v in enumerate(n.bodyId)}
    for body_id in spec['silence']:
        b.muted[lookup[body_id]] = True
    if spec.get('front_only'):
        b.muted |= ~n.bodyId.isin(front_ids).to_numpy()
    source_neuron = {ix: col for col, ix in enumerate(recorded)}
    trace = np.zeros((2000, len(recorded)))
    max_old_error = 0.
    identical = True
    core_inputs = []
    start = time.perf_counter()
    for tick in range(200):
        inp = old['feedback_inputs'][tick] if spec['feedback'] else np.zeros(len(sensor_ids))
        ids = np.r_[dl, sensor_ids]
        values = np.r_[np.full(len(dl), 250.), inp]
        for ms in range(10):
            b.advance(ids, values, 1)
            trace[tick * 10 + ms] = b.rates[recorded]
        if name == 'native_replay':
            identical = identical and np.array_equal(b.rates, old['rates'][tick])
            max_old_error = max(max_old_error, float(np.max(np.abs(b.rates - old['rates'][tick]))))
        ix = lookup[10707]
        dn_input = float((b.w[ix, dl] @ b.rates[dl]).item())
        total_internal = float((b.w[ix] @ b.rates).item())
        core_inputs.append(dict(t=(tick+1)/100, E1_body_id=10707,
                                direct_DNg100_input=dn_input,
                                all_internal_input=total_internal,
                                threshold=float(b.threshold[ix]),
                                rate_hz=float(b.rates[ix])))
    keep = np.arange(2000) >= 500
    neuron_stats = []
    for ix in recorded:
        stat = rhythm(trace[keep, source_neuron[ix]])
        neuron_stats.append(dict(bodyId=int(n.bodyId.iloc[ix]), type=n.type.iloc[ix],
                                 cell_class=n['class'].iloc[ix],
                                 side=n.somaSide.iloc[ix], neuromere=n.somaNeuromere.iloc[ix],
                                 **stat))
    pair_stats = []
    for segment, suffix in [('T1', 'F'), ('T2', 'M'), ('T3', 'H')]:
        for side in ['L', 'R']:
            leg_mask = n['class'].eq('motor neuron') & n.somaNeuromere.eq(segment) & n.somaSide.str.startswith(side)
            for joint, pm, qm in [('coxa', 'coxa swing', 'coxa stance'),
                                  ('trochanter', 'femur/tr flex', 'femur/tr extend'),
                                  ('tibia', 'tibia flex', 'tibia extend')]:
                pi = np.flatnonzero(leg_mask & n['motor module'].eq(pm))
                qi = np.flatnonzero(leg_mask & n['motor module'].eq(qm))
                if not len(pi) or not len(qi):
                    pair_stats.append(dict(leg=side+suffix, joint=joint, missing_group=True))
                    continue
                positive = trace[keep][:, [source_neuron[i] for i in pi]].mean(axis=1)
                negative = trace[keep][:, [source_neuron[i] for i in qi]].mean(axis=1)
                pair_stats.append(dict(leg=side+suffix, joint=joint, missing_group=False,
                                       positive_module=pm, negative_module=qm,
                                       positive_body_ids=n.bodyId.iloc[pi].astype(int).tolist(),
                                       negative_body_ids=n.bodyId.iloc[qi].astype(int).tolist(),
                                       **compare_pair(positive, negative)))
    pd.DataFrame(neuron_stats).to_csv(out/(name+'-neurons.csv'), index=False)
    pd.DataFrame(core_inputs).to_csv(out/(name+'-LF-E1-input.csv'), index=False)
    np.savez_compressed(out/(name+'.npz'), t=np.arange(1, 2001)/1000, rates=trace,
                        body_ids=n.bodyId.iloc[recorded].to_numpy(),
                        neuron_indices=recorded,
                        tau=b.tau, gain=b.a, threshold=b.threshold, cap=b.cap,
                        final_rates=b.rates)
    report = dict(condition=name, size_reference=reference, neurons=len(n),
                  directed_pairs=w.nnz, synapses=int(abs(w).sum()),
                  model_seconds=2., dt_seconds=.0005, recorded_dt_seconds=.001,
                  artificial_DN_body_ids=n.bodyId.iloc[dl].astype(int).tolist(),
                  artificial_DN_input=250., feedback_replayed=spec['feedback'],
                  silenced_body_ids=spec['silence'], wall_seconds=time.perf_counter()-start,
                  front_region_only=spec.get('front_only', False),
                  total_muted_cells=int(b.muted.sum()),
                  compared_with_saved_full_network=name=='native_replay',
                  old_trace_bitwise_identical=identical if name=='native_replay' else None,
                  old_trace_max_error_hz=max_old_error if name=='native_replay' else None,
                  rhythmic_motor_cells=sum(x['diagnostic_sustained_rhythm']
                                           for x in neuron_stats if x['cell_class']=='motor neuron'),
                  recorded_motor_cells=len(motors),
                  core_neurons=[x for x in neuron_stats if x['type'] in CORE_TYPES],
                  motor_pairs=pair_stats)
    (out/(name+'.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(name, 'rhythmic_MN', report['rhythmic_motor_cells'], 'alternating_pairs',
          [(x['leg'], x['joint']) for x in pair_stats if x.get('alternating_neural_dominance')],
          'old_error', report['old_trace_max_error_hz'], 'wall', report['wall_seconds'], flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditions', default=','.join(CONDITIONS),
                   help='Comma-separated cases; empty string only regenerates summaries.')
    p.add_argument('--output', type=Path, default=ROOT/'results/cpg-context-diagnostic')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: ROOT/path for name, path in dict(
        whole_neurons='data/cpg/manc-neurons.csv.gz',
        front_neurons='data/cpg/front-neurons.csv.gz',
        graph='data/cpg/manc-pre-post.npz',
        previous='results/cpg-walking/states.npz',
        sensory_mapping='results/cpg-walking/sensory-map.json').items()}
    n = pd.read_csv(paths['whole_neurons']).fillna('')
    front = pd.read_csv(paths['front_neurons'])
    w = sparse.load_npz(paths['graph'])
    with np.load(paths['previous']) as saved:
        old = {key: saved[key] for key in saved.files}
    assert np.array_equal(old['body_ids'], n.bodyId.to_numpy())
    lookup = {int(v): i for i, v in enumerate(n.bodyId)}
    sensor_map = json.loads(paths['sensory_mapping'].read_text())
    sensor_ids = np.array([lookup[int(x)] for group in sensor_map for x in group['body_ids']])
    assert old['feedback_inputs'].shape == (200, len(sensor_ids))
    references = dict(whole=float(n['size'].median()), front=float(front['size'].median()))
    # Export the actual inhibitory/excitatory sources from the exactly replayed
    # historical full-network trace. This is an input decomposition, not an
    # attribution of necessity to any individual presynaptic cell.
    e1 = lookup[10707]
    weights = w[:, e1].toarray().ravel()
    present = weights != 0
    sources = n.loc[present, ['bodyId','type','class','somaSide','somaNeuromere','predictedNt']].copy()
    sources['signed_synapses_to_E1'] = weights[present]
    sources['presynaptic_mean_rate_hz'] = old['rates'][50:].mean(axis=0)[present]
    sources['mean_model_input_to_E1'] = (sources.signed_synapses_to_E1 *
                                        sources.presynaptic_mean_rate_hz * .03)
    sources.sort_values('mean_model_input_to_E1').to_csv(
        out/'native-replay-LF-E1-presynaptic-drive.csv', index=False)
    reports = {name: json.loads((out/(name+'.json')).read_text())
               for name in CONDITIONS if (out/(name+'.json')).exists()}
    for name in filter(None, args.conditions.split(',')):
        spec = CONDITIONS[name]
        reports[name] = run_one(name, spec, n, w, old, sensor_ids,
                               references[spec['reference']], out, front.bodyId.to_numpy())
    record = dict(scope=__doc__, sources={name:dict(path=str(path), sha256=sha(path))
                                        for name, path in paths.items()},
                  graph_changed=False, full_model_parameters_drawn_once_per_condition_seed=1,
                  source_code_sha256={'cpg_rate.py':sha(ROOT/'cpg_rate.py'),
                                      'cpg_context_diagnostic.py':sha(Path(__file__))},
                  front_membership=dict(author_front_cells=len(front),
                      intersection_with_whole=int(n.bodyId.isin(front.bodyId).sum()),
                      absent_from_whole_body_ids=front.loc[~front.bodyId.isin(n.bodyId),'bodyId'].astype(int).tolist(),
                      interpretation='Front-region conditions mute cells outside the '
                      'author-selected front-leg motor/premotor/DN node set; not an '
                      'anatomically bounded brain region or a full-network success.'),
                  size_references=references,
                  parameter_change_front_over_whole=dict(
                      gain=references['front']/references['whole'],
                      threshold=references['whole']/references['front']),
                  analysis_window_seconds=[.501, 2.],
                  rhythm_criteria='>=2Hz peak-to-peak; >=5 peaks with >=max(.5Hz,10% range) '
                    'prominence, >=10ms peak distance; period CV<=.2; last-third range '
                    '>=half first-third range.',
                  alternation_criteria='Both annotated module-average traces meet rhythm '
                    'criteria; frequencies agree within10%; net dominance reverses >=4 '
                    'times, each side exceeds .5Hz for >=10ms. This is not a gait score.',
                  conditions=reports,
                  limitations=['Same graph and raw parameter draws; normalization is a '
                    'sensitivity assumption, not an experimentally fitted correction.',
                    'The 142 sensory inputs are prerecorded from the old failed body run, '
                    'not recomputed under changed neural output.',
                    'No new physical walking or coordinated six-leg behavior is established.',
                    'MANC is a distinct animal from MaleCNS. Body IDs are not interchangeable.'])
    (out/'report.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
