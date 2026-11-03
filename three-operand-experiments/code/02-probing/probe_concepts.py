"""First probing stage: single-image concept extraction and held-out linear probes."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_paths import DEFAULT_RESULTS_ROOT, require_results_root

import numpy as np

POOLS = ('last', 'text_mean', 'visual_mean')


def concept_split(records):
    # All eight classes occur in each variation. No image crosses the split.
    if len(records) == 1000:
        train = [i for i, r in enumerate(records) if r['split'] == 'train']
        test = [i for i, r in enumerate(records) if r['split'] == 'test']
    else:
        train = [i for i, r in enumerate(records) if r['variation'] < 4]
        test = [i for i, r in enumerate(records) if r['variation'] >= 4]
    assert set(records[i]['filename'] for i in train).isdisjoint(
        records[i]['filename'] for i in test)
    return train, test


def decoder_layers(model):
    # Qwen3.5 mixes attention and linear-attention blocks; hook decoder outputs,
    # rather than assuming every layer has Qwen3-VL's attention internals.
    for path in ('model.language_model', 'model.language_model.model',
                 'model.text_model', 'model', 'language_model'):
        obj = model
        for name in path.split('.'):
            obj = getattr(obj, name, None)
        if hasattr(obj, 'layers'):
            return obj.layers
    raise ValueError('Cannot locate decoder layers; check the Transformers version.')


def capture(adapter, paths, prompt, model_name):
    import torch
    inputs = (adapter._prepare(paths, prompt) if model_name == 'qwen3'
              else adapter._inputs(paths, prompt))
    ids = inputs['input_ids'][0]
    image_id = adapter.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    visual = ids == image_id
    if not visual.any() or visual.all():
        raise ValueError('Expected both visual and text tokens in the prepared prompt.')
    vectors = {}

    def hook_for(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden[0]
            mask = visual.to(h.device)
            # Pool immediately so full sequence activations are not retained.
            vectors[layer] = torch.stack((h[-1], h[~mask].mean(0), h[mask].mean(0))).float().cpu().numpy()
        return hook

    handles = []
    try:
        for layer, block in enumerate(decoder_layers(adapter.model)):
            handles.append(block.register_forward_hook(hook_for(layer)))
        with torch.inference_mode():
            adapter.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return np.stack([vectors[i] for i in range(len(vectors))])


def score_concepts(features, records, out, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    train, test = concept_split(records)
    labels = np.array([''.join(map(str, r['bits'])) for r in records])
    rows = []
    for layer in range(features.shape[1]):
        for pool, name in enumerate(POOLS):
            x = features[:, layer, pool]
            for control in ('measured', 'shuffled_train_labels'):
                y = labels[train].copy()
                if control != 'measured':
                    np.random.default_rng(seed).shuffle(y)
                probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
                probe.fit(x[train], y)
                predicted = probe.predict(x[test])
                probabilities = probe.predict_proba(x[test])
                classes = list(probe.classes_)
                for j, index in enumerate(test):
                    rows.append(dict(layer=layer + 1, pool=name, control=control,
                                     filename=records[index]['filename'], truth=labels[index],
                                     predicted=predicted[j], correct=int(predicted[j] == labels[index]),
                                     true_class_probability=float(probabilities[j, classes.index(labels[index])])) )
    with (out / 'concept_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in POOLS:
        for control, style in (('measured', '-'), ('shuffled_train_labels', '--')):
            accuracy = [np.mean([r['correct'] for r in rows if r['layer'] == layer and r['pool'] == name and r['control'] == control]) for layer in range(1, features.shape[1] + 1)]
            plt.plot(range(1, len(accuracy) + 1), accuracy, style, label=f'{name}: {control}')
    plt.axhline(1 / 8, color='grey', linestyle=':', label='Chance (1/8)')
    plt.xlabel('Decoder layer output (1-based)')
    plt.ylabel('Held-out eight-class accuracy')
    plt.title('Concept information across decoder layers')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / 'concept_accuracy.png', dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'qwen3_5'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_RESULTS_ROOT,
                        help='Drive results directory; probing/<model>/concept is appended')
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--precision', choices=('4bit', 'fp16', 'bf16', 'fp32'), default='4bit')
    parser.add_argument('--smoke-test', action='store_true', help='Extract two images; do not fit probes.')
    parser.add_argument('--validate-only', action='store_true', help='Check data and split without loading weights.')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '01-operator-controls'))
    if args.model == 'qwen3':
        import operator_controls_qwen3 as experiment
    else:
        import operator_controls_qwen35 as experiment
    root = args.experiment_dir.expanduser().resolve()
    if not args.validate_only:
        root = require_results_root(root)
    out = root / 'probing' / args.model / ('concept_smoke' if args.smoke_test else 'concept')
    experiment.configure_data(args.data_dir, out)
    experiment.verify_sampling()
    records = sorted(experiment.records, key=lambda r: r['filename'])
    train, test = concept_split(records)
    if len(records) == 1000:
        assert len(train) == 800 and len(test) == 200
        out = out / 'dataset-v1'
    else:
        assert len(train) == 32 and len(test) == 16
    assert len({tuple(records[i]['bits']) for i in train}) == 8
    assert len({tuple(records[i]['bits']) for i in test}) == 8
    print(f'Concept data: {len(train)} training images, {len(test)} test images.')
    if args.validate_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    prompt = 'Describe the objects in this image.'
    image_hash = hashlib.sha256()
    for record in records:
        image_hash.update(record['filename'].encode())
        image_hash.update((experiment.IMAGE_DIR / record['filename']).read_bytes())
    config = dict(version=1, task='concept', model=experiment.MODEL_ID, precision=args.precision,
                  metadata_hash=experiment.DATA_HASH, image_hash=image_hash.hexdigest(), prompt=prompt,
                  seed=2026, pools=POOLS, smoke_test=args.smoke_test,
                  split='train variations 0-3; test variations 4-5')
    # Refuse to mix cached features from different data or settings.
    config = json.loads(json.dumps(config))
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing checkpoint settings differ. Use a different experiment directory.')
    config_path.write_text(json.dumps(config, indent=2))
    selected = records[:2] if args.smoke_test else records
    adapter = None
    features = []
    for i, record in enumerate(selected):
        path = out / (Path(record['filename']).stem + '.npz')
        if not path.exists():
            if adapter is None:
                if args.model == 'qwen3':
                    if args.upstream is None:
                        parser.error('--upstream is required for Qwen3-VL extraction')
                    code = experiment.resolve_upstream_code(args.upstream)
                    experiment.load_upstream_model(code, experiment.MODEL_ID, args.precision, 'auto', 384 * 28 * 28)
                else:
                    experiment.load_native_model(experiment.MODEL_ID, args.precision)
                adapter = experiment.vlm
            values = capture(adapter, [str(experiment.IMAGE_DIR / record['filename'])], prompt, args.model)
            assert values.ndim == 3 and values.shape[1] == 3 and np.isfinite(values).all()
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, features=values)
            temporary.replace(path)
        with np.load(path) as saved:
            features.append(saved['features'])
        print(f'Concept features {i + 1}/{len(selected)}: {record["filename"]}')
    if args.smoke_test:
        print('Extraction smoke test finished. No probe accuracy was measured.')
    else:
        score_concepts(np.stack(features), records, out, 2026)


if __name__ == '__main__':
    main()
