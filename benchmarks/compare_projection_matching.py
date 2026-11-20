"""Check real-particle refinement results against a baseline STAR file.

Compare identical inputs/search settings/batch sizes. Rotation differences use
geodesic distance rather than Euler component differences (wraps and equivalent
Euler representations are harmless). Exit nonzero on a regression. Tolerances
are fixed before inspecting the 1,000-particle results: 0.01 degree rotation,
0.01 Angstrom shift-vector distance, and 1e-4 confidence difference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import starfile


def compare_particles(baseline, candidate, expected_rows=1000,
                      angle_atol=0.01, shift_atol=0.01, confidence_atol=1e-4):
    angles = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    shifts = ['rlnOriginXAngst', 'rlnOriginYAngst']
    confidence = 'rlnParticleFigureOfMerit'
    columns = angles + shifts + [confidence]
    if len(baseline) != expected_rows or len(candidate) != expected_rows:
        raise ValueError('Particle count differs from expected workload')
    if not np.array_equal(baseline['rlnImageName'], candidate['rlnImageName']):
        raise ValueError('Particle identities/order differ')
    for frame in [baseline, candidate]:
        if not np.isfinite(frame[columns].to_numpy()).all():
            raise ValueError('Nonfinite pose, shift, or confidence')
        if not frame[confidence].between(0, 1).all():
            raise ValueError('Confidence outside [0, 1]')
    base_rot = Rotation.from_euler('ZYZ', baseline[angles].to_numpy(), degrees=True)
    candidate_rot = Rotation.from_euler('ZYZ', candidate[angles].to_numpy(), degrees=True)
    differences = dict(
        angle_degrees=np.rad2deg((candidate_rot * base_rot.inv()).magnitude()),
        shift_angstrom=np.linalg.norm(candidate[shifts].to_numpy() - baseline[shifts].to_numpy(), axis=1),
        confidence=np.abs(candidate[confidence].to_numpy() - baseline[confidence].to_numpy()))
    tolerances = dict(angle_degrees=angle_atol, shift_angstrom=shift_atol, confidence=confidence_atol)
    result = dict(particles=expected_rows,
                  bit_identical=bool(np.array_equal(baseline[columns], candidate[columns])),
                  metrics={})
    for name, values in differences.items():
        result['metrics'][name] = dict(maximum=float(values.max()),
                                      median=float(np.median(values)),
                                      percentile_95=float(np.percentile(values, 95)),
                                      tolerance=tolerances[name],
                                      exceeding_tolerance=int((values > tolerances[name]).sum()))
    result['passed'] = all(item['exceeding_tolerance'] == 0 for item in result['metrics'].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-rows', type=int, default=1000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare_particles(starfile.read(args.baseline)['particles'],
                               starfile.read(args.candidate)['particles'], args.expected_rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
