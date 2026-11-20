"""Print the counts that characterise a RegionDenseSAR v2 or v3 plan directory, for comparing two builds.

    python tools/sardet100k/plan_summary.py --plan-dir /path/to/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32

Reads only plan_metadata.json and plan_report.json. Two builds from the same train annotation, settings and code
should print the same values; small differences in feature-dependent counts would point to different image decoding.
"""
import argparse
import json
from pathlib import Path


def summary(plan_dir):
    plan_dir = Path(plan_dir)
    meta = json.loads((plan_dir / 'plan_metadata.json').read_text(encoding='utf-8'))
    report = json.loads((plan_dir / 'plan_report.json').read_text(encoding='utf-8'))
    epochs = sorted(report['epochs'], key=int)
    first, last = report['epochs'][epochs[0]], report['epochs'][epochs[-1]]
    out = dict(version=meta.get('version'), source_images=meta['source_images'],
               canvases_per_epoch=meta['output_canvases_per_epoch'], steps_per_epoch=meta['optimizer_steps_per_epoch'],
               settings=meta['settings'], planned_epochs=[int(epochs[0]), int(epochs[-1])],
               completed_source_passes=len(report['completed_source_passes']),
               every_pass_complete=all(p['every_source_exactly_once'] for p in report['completed_source_passes']),
               remaining_pass_sources_used=report['remaining_pass_sources_used'],
               total_source_exposures=report['total_source_exposures'],
               instance_exposure=[report['min_instance_exposure'], report['max_instance_exposure']],
               total_class_counts=report['total_class_counts'],
               feature_selection_fallbacks=report['feature_selection_fallbacks'],
               first_epoch_routes=first['routes'], first_epoch_gt=first['gt'],
               last_epoch_routes=last['routes'], last_epoch_gt=last['gt'])
    if 'feature_valid_regions' in meta:
        out['feature_valid_regions'] = meta['feature_valid_regions']
    if 'adaptive_feature_fallbacks' in report:
        out['adaptive_feature_fallbacks'] = report['adaptive_feature_fallbacks']
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan-dir', required=True)
    args = parser.parse_args(argv)
    result = summary(args.plan_dir)
    for key, value in result.items():
        print(f'{key}: {json.dumps(value, sort_keys=True)}')
    return result


if __name__ == '__main__':
    main()
