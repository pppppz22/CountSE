"""Convert official CARPK/SHTech Part A test annotations to CountSE COCO input."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def choose_root(root, dataset):
    candidates = [root / 'data', root] if dataset == 'carpk' else [root / 'part_A/test_data', root / 'test_data', root]
    image_dir = 'Images' if dataset == 'carpk' else 'images'
    annotation_dir = 'Annotations' if dataset == 'carpk' else 'ground_truth'
    for candidate in candidates:
        if (candidate / image_dir).is_dir() and (candidate / annotation_dir).is_dir():
            return candidate
    raise FileNotFoundError(f'Could not locate {image_dir}/ and {annotation_dir}/ under {root}')


def prepare(dataset, root, output):
    from PIL import Image
    root = choose_root(root.resolve(), dataset)
    output.mkdir(parents=True, exist_ok=True)
    if dataset == 'carpk':
        stems = [s.strip() for s in (root / 'ImageSets/test.txt').read_text().splitlines() if s.strip()]
        order = sorted(Path(s).stem + '.png' for s in stems)
    else:
        order = sorted(p.name for p in (root / 'images').glob('IMG_*.jpg'))
    expected = 459 if dataset == 'carpk' else 182
    if len(order) != expected or len(set(order)) != expected:
        raise ValueError(f'Expected {expected} unique official test images, got {len(order)}')
    image_dir = root / ('Images' if dataset == 'carpk' else 'images')
    annotation_dir = root / ('Annotations' if dataset == 'carpk' else 'ground_truth')
    category = 'car' if dataset == 'carpk' else 'people'
    coco = dict(images=[], annotations=[], categories=[dict(id=0, name=category)])
    per_image = []
    for image_id, name in enumerate(order, 1):
        assert Path(name).name == name
        image_path = image_dir / name
        with Image.open(image_path) as image:
            width, height = image.size
        coco['images'].append(dict(id=image_id, file_name=name, width=width, height=height))
        if dataset == 'carpk':
            boxes = []
            for line in (annotation_dir / (Path(name).stem + '.txt')).read_text().splitlines():
                if not line.strip():
                    continue
                fields = [float(v) for v in line.split()]
                if len(fields) not in (4, 5):
                    raise ValueError(f'Invalid CARPK annotation: {name}: {line}')
                x1, y1, x2, y2 = fields[:4]
                assert all(math.isfinite(v) for v in fields[:4]) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                boxes.append([x1, y1, x2-x1, y2-y1])
        else:
            from scipy.io import loadmat
            points = loadmat(annotation_dir / ('GT_' + Path(name).stem + '.mat'))['image_info'][0, 0]['location'][0, 0]
            assert points.ndim == 2 and points.shape[1] == 2
            boxes = []
            for px, py in points:
                assert math.isfinite(float(px)) and math.isfinite(float(py))
                # Dummy point boxes carry count annotations, not true head extents.
                # Clamping keeps every point valid in the COCO loader at boundaries.
                x = min(max(float(px), 0), width-1)
                y = min(max(float(py), 0), height-1)
                boxes.append([x, y, min(2., width-x), min(2., height-y)])
        for box in boxes:
            coco['annotations'].append(dict(id=len(coco['annotations'])+1, image_id=image_id, category_id=0,
                                           bbox=box, area=box[2]*box[3], iscrowd=0))
        per_image.append(dict(image=name, gt=len(boxes)))
    annotation_path = (output / 'annotations.json').resolve()
    annotation_path.write_text(json.dumps(coco))
    manifest = {'train': [], 'val': [dict(root=str(image_dir), anno=str(annotation_path), label_map=None,
                                       dataset_mode='coco')]}
    (output / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    audit = dict(dataset=dataset, images=len(coco['images']), objects=len(coco['annotations']),
                 category=category, appended_exemplars=0, order='Sorted filenames', per_image=per_image)
    (output / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k != 'per_image'}, indent=2))
    print('Dataset manifest:', output / 'dataset.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['carpk', 'sha'])
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.dataset_root, args.output_dir)


if __name__ == '__main__':
    main()
