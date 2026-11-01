"""Copy completed Drive results into this checkout, without copying activations."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

FILES = ('concept_accuracy.png', 'concept_predictions.csv', 'config.json')
DRIVE_RESULTS = Path('/content/drive/MyDrive/LogicCBM++/looking-beyond-attention/three-operand-experiments/results')


def copy_checked(source, destination):
    if source.resolve() == destination.resolve():
        return
    digest = hashlib.sha256(source.read_bytes()).digest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).digest() != digest:
            raise ValueError(f'Destination differs; left both files unchanged: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if hashlib.sha256(destination.read_bytes()).digest() != digest:
        raise ValueError(f'Copy verification failed: {destination}')


def verify_results(folder, model, metadata):
    from PIL import Image
    config = json.loads((folder / 'config.json').read_text())
    expected_model = {'qwen3': 'Qwen/Qwen3-VL-4B-Instruct', 'qwen3_5': 'Qwen/Qwen3.5-4B'}[model]
    if config.get('model') != expected_model or config.get('task') != 'concept' or config.get('smoke_test') is not False:
        raise ValueError(f'Not a complete concept run for {model}: {folder}')
    metadata_bytes = metadata.read_bytes()
    if config.get('metadata_hash') != hashlib.sha256(metadata_bytes).hexdigest()[:16]:
        raise ValueError('Metadata differs from the run configuration')
    test = {r['filename']: ''.join(map(str, r['bits']))
            for r in json.loads(metadata_bytes)['records'] if r['variation'] >= 4}
    if len(test) != 16:
        raise ValueError('Expected the original 16 held-out images')
    with (folder / 'concept_predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(dict)
    for row in rows:
        key = (int(row['layer']), row['pool'], row['control'])
        name = row['filename']
        if name not in test or row['truth'] != test[name] or name in groups[key]:
            raise ValueError('Unexpected, duplicated, or incorrectly labelled test image')
        correct = int(row['correct'])
        if row['predicted'] not in set(test.values()) or correct != int(row['predicted'] == row['truth']):
            raise ValueError('Prediction and correctness disagree')
        if not 0 <= float(row['true_class_probability']) <= 1:
            raise ValueError('Invalid probability')
        groups[key][name] = correct
    layers = sorted({key[0] for key in groups})
    expected = {(layer, pool, control) for layer in layers
                for pool in ('visual_mean', 'text_mean', 'last')
                for control in ('measured', 'shuffled_train_labels')}
    if not layers or layers != list(range(1, max(layers) + 1)) or set(groups) != expected:
        raise ValueError('Missing layers, pools, or controls')
    if any(set(values) != set(test) for values in groups.values()):
        raise ValueError('Each layer/pool/control must contain the same 16 test images')
    with Image.open(folder / 'concept_accuracy.png') as image:
        if image.format != 'PNG':
            raise ValueError('Expected a PNG plot')
        image.verify()
    # Read the existing CSV; never fit a probe or regenerate a missing plot here.
    for pool in ('visual_mean', 'text_mean', 'last'):
        accuracy = max(sum(values.values()) / 16 for key, values in groups.items()
                       if key[1:] == (pool, 'measured'))
        print(f'{model} best {pool}: {accuracy:.2%} (best observed layer, 16 images)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-results', type=Path, default=DRIVE_RESULTS)
    parser.add_argument('--recover-from', type=Path,
                        help='Old results root containing probing/; copy all concept files to the correct Drive root first')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    for model in ('qwen3', 'qwen3_5'):
        relative = Path('probing') / model / 'concept'
        source = args.drive_results / relative
        if args.recover_from:
            old = args.recover_from / relative
            verify_results(old, model, project / 'data/metadata.json')
            # Keep all 48 activation checkpoints on Drive so extraction can resume.
            for path in sorted(old.iterdir()):
                if path.is_file() and (path.name in FILES or path.suffix == '.npz'):
                    copy_checked(path, source / path.name)
            print(f'Copied and hash-checked {model} Drive files; originals retained.')
        verify_results(source, model, project / 'data/metadata.json')
        destination = project / 'results/02-probing' / model / 'concept'
        for name in FILES:
            copy_checked(source / name, destination / name)
        verify_results(destination, model, project / 'data/metadata.json')
        print(f'Copied and checked three lightweight files: {destination}')


if __name__ == '__main__':
    main()
