"""
Fills the "Corrected results" block of docs/REPORT.md (section 7.4) from the JSON files that
option 9 of my_labeling.py writes to results/.

Usage (from the repository root):
    python src/update_report_tables.py

The block lives between these two markers in docs/REPORT.md and is overwritten each time:
    <!-- CORRECTED-RESULTS:START -->
    <!-- CORRECTED-RESULTS:END -->
"""
import json
import os
import sys

RESULTS_DIR = './results'
REPORT = './docs/REPORT.md'
START = '<!-- CORRECTED-RESULTS:START -->'
END = '<!-- CORRECTED-RESULTS:END -->'
METRICS = ['euclidean', 'manhattan', 'cosine']

# Values of the original submission (mean accuracy over K=2..10, raw features)
ORIGINAL_IMGS = {'euclidean': 86.36, 'manhattan': 85.68, 'cosine': 86.98}
ORIGINAL_CROPPED = {'euclidean': 20.43, 'manhattan': 21.48, 'cosine': 18.77}


def load(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def raw_means(stage):
    """metric -> mean accuracy (raw features) from a knn_<dataset>.json stage."""
    rows = stage['data']['summary']['by_combination']
    return {r['metric']: r['mean_accuracy'] for r in rows if r['features'] == 'raw'}


def raw_range(stage):
    accs = [r['accuracy'] for r in stage['data']['runs'] if r['features'] == 'raw']
    return min(accs), max(accs)


def fmt(v):
    return '–' if v is None else f'{v:.2f} %'


def build():
    out = []
    summary = load('summary')
    if summary and 'train_test_overlap_removed' in summary:
        ov = summary['train_test_overlap_removed']
        sizes = summary.get('dataset_sizes', {})
        out.append(
            f"**Overlap found and removed:** {ov.get('imgs', '?')} training images were also part of the "
            f"extended dataset (`imgs`, {sizes.get('imgs', '?')} images) and {ov.get('test_imgs', '?')} were "
            f"also part of `test_imgs` ({sizes.get('test_imgs', '?')} images). Corrected runs: "
            f"protocol `{summary.get('protocol', '?')}`.\n")

    # --- Table 1: imgs, original vs corrected
    st = load('knn_imgs')
    out.append('**Table 7.1 – KNN on `imgs`, raw features, mean accuracy over K = 2…10**\n')
    if st:
        cor = raw_means(st)
        lo, hi = raw_range(st)
        best = st['data']['summary']['best_single_run']
        out += ['| Metric | Original (with overlap) | Corrected |', '|---|---|---|']
        out += [f"| {m} | {fmt(ORIGINAL_IMGS[m])} | {fmt(cor.get(m))} |" for m in METRICS]
        out.append(f"\nBest single run, corrected: K={best['k']}, {best['metric']}, {best['features']}: "
                   f"{best['accuracy']:.2f} % (original: K=2, 100.00 %). Raw-feature accuracy over all K and "
                   f"metrics ranges from {lo:.2f} % to {hi:.2f} %. Training set: {st['data']['train_size']} "
                   f"images, test set: {st['data']['test_size']}.\n")
    else:
        out.append('_Not generated yet (run option 9)._\n')

    # --- Table 2: cross-validation full vs cropped
    cv = load('knn_crossval_imgs')
    out.append('**Table 7.2 – Full vs cropped images, like-for-like '
               '(stratified cross-validation on the extended dataset, raw features, mean accuracy over K = 2…10)**\n')
    if cv:
        n_folds = cv['data']['n_folds']
        n_img = cv['data']['n_images']
        conds = cv['data']['conditions']
        table = {(r['condition'], r['metric']): r['mean_accuracy']
                 for r in cv['data']['summary'] if r['features'] == 'raw'}
        out += [f'| Metric | ' + ' | '.join(conds) + ' |', '|---|' + '---|' * len(conds)]
        out += [f"| {m} | " + ' | '.join(fmt(table.get((c, m))) for c in conds) + ' |' for m in METRICS]
        out.append(f"\n{n_img} images, {n_folds}-fold cross-validation: a test image is never in the training folds. "
                   "`full->cropped` trains on full images and queries with the cropped version of the held-out ones.\n")
    else:
        out.append('_Not generated yet (run option 9)._\n')

    # --- Table 3: train full -> query cropped, original vs corrected
    st = load('knn_imgs-cropped')
    out.append('**Table 7.3 – KNN on cropped `imgs`: training on all full training images (overlap removed), '
               'querying with cropped images; raw features, mean accuracy over K = 2…10**\n')
    if st:
        cor = raw_means(st)
        out += ['| Metric | Original (broken cropped training set) | Corrected |', '|---|---|---|']
        out += [f"| {m} | {fmt(ORIGINAL_CROPPED[m])} | {fmt(cor.get(m))} |" for m in METRICS]
        out.append('')
    else:
        out.append('_Not generated yet (run option 9)._\n')

    # --- Table 4: test_imgs sanity check
    st = load('knn_test-imgs')
    out.append('**Table 7.4 – KNN on `test_imgs` (separate folder, not affected), raw features**\n')
    if st:
        cor = raw_means(st)
        lo, hi = raw_range(st)
        out += ['| Metric | Mean accuracy over K = 2…10 |', '|---|---|']
        out += [f"| {m} | {fmt(cor.get(m))} |" for m in METRICS]
        out.append(f"\nAcross every K and metric the accuracy ranges from {lo:.2f} % to {hi:.2f} %.\n")
    else:
        out.append('_Not generated yet (run option 9)._\n')

    return '\n'.join(out)


def main():
    if not os.path.exists(REPORT):
        sys.exit(f'{REPORT} not found (run from the repository root).')
    with open(REPORT, encoding='utf-8') as f:
        text = f.read()
    if START not in text or END not in text:
        sys.exit(f'Markers {START} / {END} not found in {REPORT}.')
    a, b = text.index(START) + len(START), text.index(END)
    new_text = text[:a] + '\n\n' + build() + '\n' + text[b:]
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print(f'Updated {REPORT} from {RESULTS_DIR}/*.json')


if __name__ == '__main__':
    main()
