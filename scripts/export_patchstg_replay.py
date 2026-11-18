"""Export a standalone historical replay using actual frozen PatchSTG forecasts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.data.spatial import load_spatial_maps
from traffic_benchmark.evaluation.checkpoint import file_sha256, validate_checkpoint, restore_frozen_model
from traffic_benchmark.evaluation.replay import select_windows, replay_payload, render_replay
from traffic_benchmark.training.trainer import seed_everything


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/patchstg_sd_seed2023_50ep'))
    parser.add_argument('--evaluation-report', type=Path, default=Path('runs/patchstg_sd_seed2023_test_v1/report.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw/largest_sd'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/patchstg_replay_v1'))
    parser.add_argument('--start-window', type=int, default=0)
    parser.add_argument('--frames', type=int, default=96)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a new path.')
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Use --device cpu explicitly if needed.')
    training = json.loads((args.run_dir / 'report.json').read_text())
    checkpoint_path = args.run_dir / 'best.pt'
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    values, metadata = load_processed_sd(args.data_dir)
    splits = make_splits(len(values))
    scaler = GlobalStandardScaler.fit(values, splits['train'][1])
    hashes = {name: file_sha256(args.data_dir / name) for name in ('flowsd.npz', 'sd_meta.csv')}
    identity = validate_checkpoint(checkpoint, training, hashes, splits, scaler)
    cfg = identity['model_config']
    if cfg['history'] != 12 or cfg['nodes'] != 716 or cfg['slots_per_day'] != 96:
        raise ValueError('Replay requires the fixed SD 12-to-12, 716-sensor protocol.')
    evaluation = json.loads(args.evaluation_report.read_text())
    if (evaluation.get('status') != 'completed' or evaluation['checkpoint']['sha256'] != checkpoint_hash
            or evaluation['source_hashes'] != hashes or evaluation['config']['split'] != 'test'):
        raise ValueError('Evaluation report does not match this checkpoint and dataset.')
    torch.set_num_threads(2)
    seed_everything(training['config']['seed'])
    maps = load_spatial_maps(args.run_dir / 'spatial_maps.npz', identity['spatial'])
    model = restore_frozen_model(checkpoint, maps).to(args.device)
    dataset = TrafficWindows(values, splits['test'], scaler)
    indices = select_windows(dataset, args.start_window, args.frames, args.frame_stride)
    loader = DataLoader(Subset(dataset, indices), batch_size=training['config']['batch_size'],
                        shuffle=False, num_workers=0)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            forecast = model(batch['x'].to(args.device), batch['x_calendar'].to(args.device))
            predictions.append(scaler.inverse_transform(forecast).cpu().numpy())
    template_path = Path(__file__).resolve().parents[1] / 'demos' / 'patchstg_replay.html'
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'historical forecast replay; cached real model predictions; no live traffic or road simulation',
        'checkpoint_sha256': checkpoint_hash, 'epoch': checkpoint['epoch'],
        'seed': training['config']['seed'], 'source_sha256': hashes,
        'training_report_sha256': file_sha256(args.run_dir / 'report.json'),
        'evaluation_report_sha256': file_sha256(args.evaluation_report),
        'full_test_metrics': evaluation['results']['test']['overall_pooled'],
        'training_identity': identity,
        'selection': {'start_window': args.start_window, 'frames': args.frames, 'frame_stride': args.frame_stride},
        'device': args.device,
        'software': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
        'hardware': {'platform': platform.platform()},
        'git': {'revision': git_output('rev-parse', 'HEAD'), 'status': git_output('status', '--porcelain')},
        'implementation_sha256': {str(p): file_sha256(p) for p in (
            Path('scripts/export_patchstg_replay.py'), Path('src/traffic_benchmark/evaluation/replay.py'), template_path)},
        'note': 'Display forecasts rounded to 3 decimals. No new benchmark or tuning. Original artifacts unchanged.',
    }
    if args.device == 'cuda:0':
        props = torch.cuda.get_device_properties(0)
        provenance['hardware']['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                                         'compute_capability': list(torch.cuda.get_device_capability(0))}
    payload = replay_payload(values, metadata, dataset, indices, np.concatenate(predictions), provenance)
    page = render_replay(template_path.read_text(), payload)
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError('Checkpoint changed during replay export.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'index.html').write_text(page, encoding='utf-8')
    manifest = {**provenance, 'artifact': {'name': 'index.html', 'sha256': file_sha256(args.output_dir / 'index.html')},
                'sensors': len(metadata), 'first_issued_at': payload['frames'][0]['issued_at'],
                'last_issued_at': payload['frames'][-1]['issued_at']}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f"Exported {len(indices)} origins × {len(metadata)} sensors × 12 forecast leads.")
    print('Open:', args.output_dir / 'index.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
