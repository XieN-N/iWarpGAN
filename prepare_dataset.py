#!/usr/bin/env python3
"""Prepare CASIA-Iris-Thousand-Norm for iWarpGAN dataset_tool.py.

Usage:
    python prepare_dataset.py <source> <dest_staging>
Example:
    python prepare_dataset.py \
        /home/xxienn/Datasets/CASIA-Iris-Thousand-Norm \
        /home/xxienn/Datasets/CASIA-Staging-Ours
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def collect_images(source_dir, subj_start, subj_end, exclude_debug=True):
    """Yield (subject_id_str, src_path) for every non-debug image."""
    for subj in range(subj_start, subj_end + 1):
        subj_dir = os.path.join(source_dir, f'{subj:03d}')
        if not os.path.isdir(subj_dir):
            continue
        for eye in ('L', 'R'):
            eye_dir = os.path.join(subj_dir, eye)
            if not os.path.isdir(eye_dir):
                continue
            for fname in sorted(os.listdir(eye_dir)):
                if exclude_debug and '_debug' in fname:
                    continue
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                subject_id_str = f'{subj:03d}{eye}'
                src_path = os.path.join(eye_dir, fname)
                yield subject_id_str, src_path


def main():
    parser = argparse.ArgumentParser(description='Prepare CASIA dataset for iWarpGAN')
    parser.add_argument('source', help='CASIA-Iris-Thousand-Norm directory')
    parser.add_argument('dest', help='Output staging directory')
    parser.add_argument('--subj-start', type=int, default=50, help='First subject number (default: 50)')
    parser.add_argument('--subj-end', type=int, default=99, help='Last subject number (default: 99)')
    args = parser.parse_args()

    source = args.source
    dest = args.dest

    if not os.path.isdir(source):
        print(f'Error: source {source} not found')
        sys.exit(1)

    os.makedirs(dest, exist_ok=True)

    records = []
    for subject_id_str, src_path in collect_images(source, args.subj_start, args.subj_end):
        ext = os.path.splitext(src_path)[1]
        dst_name = f'{subject_id_str}_{Path(src_path).name}'
        dst_path = os.path.join(dest, dst_name)
        shutil.copy2(src_path, dst_path)
        records.append((dst_name, subject_id_str))

    # Write dataset_attributes.txt (CSV: filename,subject_id_string)
    attrs_path = os.path.join(dest, 'dataset_attributes.txt')
    with open(attrs_path, 'w') as f:
        f.write('filename,subject_id\n')
        for fname, sid in records:
            f.write(f'{fname},{sid}\n')

    unique_ids = sorted(set(sid for _, sid in records))
    print(f'Copied {len(records)} images to {dest}')
    print(f'Unique subject IDs: {len(unique_ids)} (from {unique_ids[0]} to {unique_ids[-1]})')
    print(f'dataset_attributes.txt written with header + {len(records)} entries')


if __name__ == '__main__':
    main()
