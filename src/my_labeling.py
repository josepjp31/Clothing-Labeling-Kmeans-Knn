__authors__ = ['1710404','1714533','1713947']
__group__ = '74'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, Plot3DCloud, visualize_k_means
from KNN import KNN
from Kmeans import KMeans, get_colors
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from PIL import Image
import random
import os
import json
import traceback
import hashlib

import contextlib
from datetime import datetime

# === OUTPUT FOLDERS (figures and JSON results) ===
FIGURES_DIR = './figures'
RESULTS_DIR = './results'

# Evaluation protocol of the saved results. Stages whose JSON carries an older/missing
# protocol are recomputed when resuming. v2 = no train/test overlap + valid crop handling
# (see report, section 7).
PROTOCOL = 'v2-no-overlap'


def remove_overlap(train_imgs, train_labels, other_imgs, name='other'):
    """
    Removes from the training set every image whose pixels are identical to an image in
    `other_imgs`. Without this, a test image finds itself in the training set at distance 0
    and KNN accuracy is inflated.
    Returns (clean_train_imgs, clean_train_labels, n_removed).
    """
    def h(im):
        return hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest()
    other_hashes = {h(im) for im in other_imgs}
    keep = np.array([h(im) not in other_hashes for im in train_imgs], dtype=bool)
    n_removed = int((~keep).sum())
    print(f"🔎 Train / {name} overlap: {n_removed} of {len(train_imgs)} training images also appear "
          f"in '{name}' -> removed from the training set ({int(keep.sum())} left)")
    return np.asarray(train_imgs)[keep], np.asarray(train_labels)[keep], n_removed



def to_jsonable(obj):
    """Recursively converts numpy types / defaultdicts / tuples into JSON-friendly objects."""
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(name, data):
    """Saves `data` as results/<name>.json and returns the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved: {path}")
    return path


@contextlib.contextmanager
def capture_figures(prefix, names=None):
    """
    While active, plt.show() saves every open figure to figures/<prefix>_<name>.png
    (instead of opening a window) and closes it. `names` are used in order; extra
    figures are called fig1, fig2, ... Works for any function that ends in plt.show(),
    including the ones in utils_data.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = list(names or [])
    saved = []
    plt.close('all')   # start clean: figures left open by earlier code must not be saved under these names
    original_show = plt.show

    def _save_show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            i = len(saved)
            name = names[i] if i < len(names) else f"fig{i + 1}"
            path = os.path.join(FIGURES_DIR, f"{prefix}_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved.append(path)
            print(f"🖼️  Figure saved: {path}")
        plt.close('all')

    plt.show = _save_show
    try:
        yield saved
    finally:
        plt.show = original_show
        plt.close('all')

# === QUALITATIVE FUNCTIONS ===

def retrieval_by_color(images, color_preds, query_colors, color_percentages=None, return_scores=False):
    """
    Search for images with matching color labels and optionally sort them by match percentage.

    Args:
        images (list): list of images.
        color_preds (list of list of str): color labels per image.
        query_colors (list or str): colors to search for.
        color_percentages (list of list of float, optional): percentage of each color per image.
        return_scores (bool): if True, returns tuples (img, score). If False, only the images.

    Returns:
        list: images (or tuples) containing any of the specified colors, ordered.
    """
    query_colors = [c.lower() for c in (query_colors if isinstance(query_colors, list) else [query_colors])]

    result = []

    for i, (img, labels) in enumerate(zip(images, color_preds)):
        labels_lower = [c.lower() for c in labels]

        match_indices = [j for j, c in enumerate(labels_lower) if c in query_colors]

        if match_indices:
            if color_percentages:
                match_score = sum(color_percentages[i][j] for j in match_indices)
            else:
                match_score = 1

            result.append((img, match_score))

    result.sort(key=lambda x: x[1], reverse=True)

    if return_scores:
        return result  # [(img, score), ...]
    else:
        return [img for img, _ in result]
    


def retrieval_by_shape(images, shape_preds, query_shape, shape_votes=None, return_scores=False):
    """
    Search for images with a given shape, optionally ordered by neighbor percentage.

    Args:
        images (list): list of images.
        shape_preds (list of str): shape labels per image.
        query_shape (str): the shape to look for.
        shape_votes (list of float, optional): percentage of KNN neighbors voting for the shape.
        return_scores (bool): if True, returns tuples (img, score). If False, only the images.

    Returns:
        list: images (or tuples) with the specified shape, ordered by percentage if given.
    """
    result = []

    for i, (img, label) in enumerate(zip(images, shape_preds)):
        if label == query_shape:
            score = shape_votes[i] if shape_votes else 1
            result.append((img, score))

    # Sort by percentage if provided
    result.sort(key=lambda x: x[1], reverse=True)

    if return_scores:
        return result  # [(img, score), ...]
    else:
        return [img for img, _ in result]

def retrieval_combined(images, shape_preds, color_preds, query_shape, query_colors,
                       shape_votes=None, color_percentages=None, return_scores=False):
    """
    Search for images that match both shape and color, ordered by combined score if provided.

    Args:
        images (list): list of images.
        shape_preds (list of str): shape predictions.
        color_preds (list of list of str): color predictions.
        query_shape (str): shape to search for.
        query_colors (list or str): colors to search for.
        shape_votes (list of float, optional): percentage of neighbors by shape.
        color_percentages (list of list of float, optional): percentages per color.
        return_scores (bool): if True, returns (img, score).

    Returns:
        list: images (or tuples) matching both shape and color, ordered by score.
    """
    query_colors = [c.lower() for c in (query_colors if isinstance(query_colors, list) else [query_colors])]
    result = []

    for i, (img, shape_label, color_labels) in enumerate(zip(images, shape_preds, color_preds)):
        # Shape match
        if shape_label != query_shape:
            continue

        # Color match
        color_labels_lower = [c.lower() for c in color_labels]
        match_indices = [j for j, c in enumerate(color_labels_lower) if c in query_colors]
        if not match_indices:
            continue

        # Shape score
        shape_score = shape_votes[i] if shape_votes else 1

        # Color score
        if color_percentages:
            color_score = sum(color_percentages[i][j] for j in match_indices)
        else:
            color_score = 1

        # Combined score (simple average)
        combined_score = (shape_score + color_score) / 2

        result.append((img, combined_score))

    result.sort(key=lambda x: x[1], reverse=True)

    if return_scores:
        return result
    else:
        return [img for img, _ in result]



# === QUANTITATIVE FUNCTIONS ===

def get_shape_accuracy(preds, gts, return_errors=False):
    """
    Calculate the accuracy of the obtained labels and optionally return the errors.

    Args:
        preds (list): labels predicted by KNN.
        gts (list): ground-truth of the image set.
        return_errors (bool): if True, also returns the incorrect labels.

    Returns:
        float: percentage of correct labels.
        dict (optional): mapping between incorrect and correct labels.
    """
    correct = 0
    total = len(gts)
    errors = {}

    for pred, gt in zip(preds, gts):
        if pred == gt:
            correct += 1
        elif return_errors:
            if (gt, pred) not in errors:
                errors[(gt, pred)] = 1
            else:
                errors[(gt, pred)] += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    if return_errors:
        return accuracy, errors
    else:
        return accuracy

def get_color_accuracy(pred_labels, gt_labels):
    """
    pred_labels: list of sets of predicted labels per image (e.g. [{0, 2}, {1}, ...])
    gt_labels: list of sets of ground-truth labels per image (same format)

    Returns: average F1-score across all images
    """
    assert len(pred_labels) == len(gt_labels), "Incompatible lengths between predictions and ground-truth."

    f1_scores = []
    for pred, gt in zip(pred_labels, gt_labels):
        pred = set(pred)
        gt = set(gt)

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)
        
        

    return (sum(f1_scores) / len(f1_scores)) *100


def kmeans_analysis(images, gts, Kmax=10):
    """
    Runs KMeans for K from 2 to Kmax with different initializations and heuristics
    """

    # Configurations to test
    init_methods = ['first', 'random', 'custom', 'kmeans++', 'pca']
    heuristics = [
        ('WCD', lambda m: m.withinClassDistance()),
    ]
    
    # Dictionaries to store results
    results = {
        'criteria': defaultdict(list),
        'accuracy': defaultdict(list),
        'time': defaultdict(list),
        'iterations': defaultdict(list)
    }
    init_stats = {
    'time': defaultdict(list),
    'accuracy': defaultdict(list)
    }
    
    # Different colors for each combination
    colors = plt.cm.tab20.colors

    # Main analysis loop
    for init in init_methods:
        for (heur_name, heur_fn) in heuristics:
            print(f"\nAnalyzing: Initialization={init}, Heuristic={heur_name}")
            
            Ks = list(range(2, Kmax+1))
            color_idx = 0
            
            for K in Ks:
                t0 = time.time()
                
                # 1. Run a global KMeans using the heuristic as the criterion
                model = KMeans(images, K=K, options={
                    'km_init': init,
                    'fitting': heur_name.lower()  # Use the heuristic as the criterion
                })


                model.fit()
                val = heur_fn(model)
                elapsed = time.time() - t0
                
                # 2. Predict colors using the same heuristic
                color_preds = []
                for img in images:
                    kmeans = KMeans(img, K, options={
                        'km_init': init,
                        'fitting': heur_name.lower()
                    })
                    kmeans.fit()
                    centroids = kmeans.centroids
                    labels = get_colors(centroids)
                    color_preds.append(labels)
                
                # Calculate accuracy
                acc = get_color_accuracy(color_preds, gts) if len(color_preds) == len(gts) else 0
                
                # Store results
                key = f"{init}_{heur_name}"
                results['criteria'][key].append(val)
                results['accuracy'][key].append(acc)
                results['time'][key].append(elapsed)
                results['iterations'][key].append(model.num_iter)
                # Store to compute averages
                init_stats['time'][init].append(elapsed)
                init_stats['accuracy'][init].append(acc)
                
                print(f"K={K} | {heur_name}={val:.2f} | Acc={acc:.2f}% | Time={elapsed:.2f}s | Iters={model.num_iter}")
    # Show averages per initialization
    print("\nAverages by initialization method:")
    for init in init_methods:
        avg_time = np.mean(init_stats['time'][init]) if init_stats['time'][init] else 0
        avg_acc = np.mean(init_stats['accuracy'][init]) if init_stats['accuracy'][init] else 0
        print(f"{init}: Average time = {avg_time:.2f}s, Average accuracy = {avg_acc:.2f}%")

    # Individual plot generation
    Ks = list(range(2, Kmax+1))
    
    # 1. Criteria Value vs K plot
    plt.figure(figsize=(10, 6))
    color_idx = 0
    for key in results['criteria']:
        init, heur = key.split('_')
        plt.plot(Ks, results['criteria'][key], marker='o', 
                label=f"{init} ({heur})", color=colors[color_idx])
        color_idx = (color_idx + 1) % len(colors)
    plt.xlabel('K')
    plt.ylabel('Heuristic Value')
    plt.title('Heuristic Evolution by K')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 2. Accuracy vs K plot
    plt.figure(figsize=(10, 6))
    color_idx = 0
    for key in results['accuracy']:
        init, heur = key.split('_')
        plt.plot(Ks, results['accuracy'][key], marker='o', 
                label=f"{init} ({heur})", color=colors[color_idx])
        color_idx = (color_idx + 1) % len(colors)
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy in Color Prediction by K')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 3. Time vs K plot
    plt.figure(figsize=(10, 6))
    color_idx = 0
    for key in results['time']:
        init, heur = key.split('_')
        plt.plot(Ks, results['time'][key], marker='o', 
                label=f"{init} ({heur})", color=colors[color_idx])
        color_idx = (color_idx + 1) % len(colors)
    plt.xlabel('K')
    plt.ylabel('Time (s)')
    plt.title('Execution Time by K')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 4. Iterations vs K plot
    plt.figure(figsize=(10, 6))
    color_idx = 0
    for key in results['iterations']:
        init, heur = key.split('_')
        plt.plot(Ks, results['iterations'][key], marker='o', 
                label=f"{init} ({heur})", color=colors[color_idx])
        color_idx = (color_idx + 1) % len(colors)
    plt.xlabel('K')
    plt.ylabel('Number of Iterations')
    plt.title('Iterations to Converge by K')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'K_values': Ks,
        'criteria': dict(results['criteria']),
        'accuracy': dict(results['accuracy']),
        'time': dict(results['time']),
        'iterations': dict(results['iterations']),
        'averages_by_init': {
            init: {
                'mean_time_s': float(np.mean(init_stats['time'][init])) if init_stats['time'][init] else 0.0,
                'mean_accuracy': float(np.mean(init_stats['accuracy'][init])) if init_stats['accuracy'][init] else 0.0,
            } for init in init_methods
        },
    }


# === OTHER FUNCTIONS ===

def resize_images(images, target_shape):
    resized = []
    height, width = target_shape[:2]
    for img in images:
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((width, height))
        resized.append(np.array(img_resized))
    return np.array(resized)


def knn_systematic_analysis(train_imgs, train_labels, test_imgs, test_labels):
    print("\n=== SYSTEMATIC KNN ANALYSIS ===")

    # Configurations to test
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics = ['euclidean', 'manhattan', 'cosine']
    feature_types = ['raw', 'mean_rgb', 'mean_halves', 'grayscale_mean']

    results = []
    aggregate_accuracy = defaultdict(list)  # (metric, feature) -> [accuracies...]
    aggregate_time = defaultdict(list)      # (metric, feature) -> [times...]

    for k in k_values:
        for metric in metrics:
            for feat_type in feature_types:
                print(f"\nEvaluating K={k}, distance={metric}, features={feat_type}")

                knn = KNN(train_imgs, train_labels)
                knn.distance = metric
                knn.feature_type = feat_type

                start_time = time.time()
                predicted = knn.predict(test_imgs, k=k)
                elapsed_time = time.time() - start_time

                accuracy = get_shape_accuracy(predicted, test_labels)

                results.append({
                    'k': k,
                    'metric': metric,
                    'features': feat_type,
                    'accuracy': accuracy,
                    'time': elapsed_time
                })

                key = (metric, feat_type)
                aggregate_accuracy[key].append(accuracy)
                aggregate_time[key].append(elapsed_time)

                print(f"Accuracy: {accuracy:.2f}% | Time: {elapsed_time:.2f} s")

    # Show detailed summary
    print("\nSummary of individual results:")
    for r in sorted(results, key=lambda x: -x['accuracy']):
        print(f"K={r['k']}, {r['metric']}, {r['features']}: "
              f"{r['accuracy']:.2f}% | Time: {r['time']:.2f} s")

    # Show aggregated averages
    print("\n📊 Average accuracy and time by metric/feature combination:")
    media_results = []
    for key in aggregate_accuracy:
        accs = aggregate_accuracy[key]
        times = aggregate_time[key]
        avg_acc = sum(accs) / len(accs)
        avg_time = sum(times) / len(times)
        metric, feat_type = key
        media_results.append((avg_acc, avg_time, metric, feat_type))

    for avg_acc, avg_time, metric, feat_type in sorted(media_results, reverse=True):
        print(f"{metric}, {feat_type}: {avg_acc:.2f}% average accuracy | {avg_time:.2f} s average time")

    # === Average accuracy plot ===
    labels = [f"{m}\n{f}" for _, _, m, f in sorted(media_results, reverse=True)]
    accuracies = [acc for acc, _, _, _ in sorted(media_results, reverse=True)]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, accuracies, color='skyblue')
    plt.xlabel("Average accuracy (%)")
    plt.title("Average accuracy by metric/feature combination")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # === Accuracy vs K plots ===
    plt.figure(figsize=(12, 6))
    for key in aggregate_accuracy:
        metric, feat_type = key
        plt.plot(k_values, aggregate_accuracy[key], label=f"{metric} - {feat_type}")

    plt.xlabel("K")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs K by combination")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === Time vs K plots ===
    plt.figure(figsize=(12, 6))
    for key in aggregate_time:
        metric, feat_type = key
        plt.plot(k_values, aggregate_time[key], label=f"{metric} - {feat_type}")

    plt.xlabel("K")
    plt.ylabel("Time (s)")
    plt.title("Prediction time vs K by combination")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === Average time per combination ===
    labels = [f"{m}\n{f}" for _, _, m, f in sorted(media_results, reverse=True)]
    times = [t for _, t, _, _ in sorted(media_results, reverse=True)]

    plt.figure(figsize=(12, 6))
    plt.barh(labels, times, color='salmon')
    plt.xlabel("Average time (s)")
    plt.title("Average time by metric/feature combination")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return results

def visualizarRetrievalConKMeans(test_imgs,
                                 predicted_shapes=None,
                                 color_preds=None,
                                 color_percentages=None,
                                 shape_votes=None,
                                 query_shape=None,
                                 query_colors=None,
                                 color=True,
                                 forma=False,
                                 ambos=False,
                                 n=20):
    """
    Visualizes retrieval using labels and percentages that were already generated.

    Args:
        test_imgs (list): Images to display.
        predicted_shapes (list): Predicted shapes for each image (optional).
        color_preds (list): Predicted colors for each image.
        color_percentages (list): Percentage of colors per image.
        shape_votes (list): Confidence vote for shape per image.
        query_shape (str): Shape to search for.
        query_colors (list): List of colors to search for.
        color (bool): Whether to enable color search.
        forma (bool): Whether to enable shape search.
        ambos (bool): Whether to enable combined search.
        n (int): Number of images to display.
    """

    print("\n🔍 Advanced retrieval visualization...")

    # === Prediction evaluation ===
    ok = None
    ok_shape = None
    ok_color = None

    # Correct shapes
    if forma or ambos:
        if query_shape is not None and predicted_shapes is not None:
            ok_shape = [query_shape.lower().strip() == shape.lower().strip() for shape in predicted_shapes]

    # Correct colors (main or secondary color > 0.2)
    if color or ambos:
        if query_colors is not None and color_preds is not None and color_percentages is not None:
            # Convert everything to lowercase for case-insensitive comparison
            query_colors_lower = [qc.lower().strip() for qc in query_colors]

            ok_color = []
            for pred_colors, percents in zip(color_preds, color_percentages):
                pred_colors_lower = [c.lower().strip() for c in pred_colors]

                # 1. Main color matches
                if pred_colors_lower[0] in query_colors_lower:
                    ok_color.append(True)
                # 2. Any secondary color matches with > 20%
                elif any(c in query_colors_lower and p > 0.2 for c, p in zip(pred_colors_lower, percents)):
                    ok_color.append(True)
                else:
                    ok_color.append(False)

    # === Retrieval by mode ===
    if ambos:
        if predicted_shapes is None or color_preds is None:
            print("❌ Shape and color predictions are required for combined retrieval.")
            return
        print("→ Retrieving by shape + color...")
        retrieved = retrieval_combined(
            test_imgs, predicted_shapes, color_preds,
            query_shape=query_shape,
            query_colors=query_colors,
            shape_votes=shape_votes,
            color_percentages=color_percentages,
            return_scores=True
        )
        title = f"{query_shape} + {'/'.join(query_colors)}"
        ok = [s and c for s, c in zip(ok_shape, ok_color)] if ok_shape and ok_color else None

    elif forma:
        if predicted_shapes is None:
            print("❌ No shape predictions were provided.")
            return
        print("→ Retrieving by shape only...")
        retrieved = retrieval_by_shape(
            test_imgs,
            predicted_shapes,
            query_shape,
            shape_votes=shape_votes,
            return_scores=True
        )
        title = f"Shape: {query_shape}"
        ok = ok_shape

    elif color:
        if color_preds is None:
            print("❌ No color predictions were provided.")
            return
        print("→ Retrieving by color only...")
        retrieved = retrieval_by_color(
            test_imgs,
            color_preds,
            query_colors,
            color_percentages=color_percentages,
            return_scores=True
        )
        title = f"Color: {'/'.join(query_colors)}"
        ok = ok_color

    else:
        print("❌ No retrieval type has been selected.")
        return

    # === Prepare data for visualization ===
    retrieved_imgs = [img for img, _ in retrieved]
    retrieved_scores = [f"{score*100:.1f}%" for _, score in retrieved]

    # Get indices of retrieved images to verify ground truth
    retrieved_indices = []
    for img in retrieved_imgs:
        idx = next((i for i, timg in enumerate(test_imgs) if np.array_equal(timg, img)), None)
        retrieved_indices.append(idx)

    # Verify ground truth for retrieved images
    if ok is not None and retrieved_indices:
        retrieved_ok = [ok[idx] for idx in retrieved_indices if idx is not None]
    else:
        retrieved_ok = None

    # === Final visualization ===
    mostrar_retrieval(retrieved, n=n, title=title, ok=retrieved_ok)


def mostrar_retrieval(retrieved_with_scores, n, title='', ok=None):
    topN = min(n, len(retrieved_with_scores))
    print(f"Recovered images: {len(retrieved_with_scores)}")
    if topN > 0:
        top_imgs = [img for img, _ in retrieved_with_scores[:topN]]
        top_scores = [f"{score * 100:.1f}%" for _, score in retrieved_with_scores[:topN]]
        
        # If 'ok' is not defined, create a list of True values (to avoid errors)
        if ok is None:
            ok = [True] * len(retrieved_with_scores)
        top_ok = ok[:topN]
       
        visualize_retrieval(np.array(top_imgs), topN=topN, info=top_scores, ok=top_ok, title=title)
    else:
        print("No images were found for the query.")


def kmeans_interactivo(original_images, cropped_images, color_labels):
    # Image type selection section
    print("\n=== KMEANS IMAGE TYPE SELECTION ===")
    print("1. Use original images")
    if cropped_images is not None:
        print("2. Use cropped images")
    print("3. Exit")

    seleccion = input("Select an option (1-3): ").strip()
    while seleccion not in ['1', '2', '3'] or (seleccion == '2' and cropped_images is None):
        print("Invalid input (cropped images are only available for the extended dataset).")
        seleccion = input("Select an option (1-3): ").strip()

    if seleccion == '3':
        print("Exiting...")
        return
    elif seleccion == '1':
        images = original_images
        origen = "original"
        print("✅ Using original images.")
    else:
        images = cropped_images
        origen = "cropped"
        print("✅ Using cropped images.")

    print(f"\n🖼️ Total images loaded: {len(images)}")

    # Initialization method selection
    inicializaciones = {
        '1': 'first',
        '2': 'random',
        '3': 'custom',
        '4': 'kmeans++',
        '5': 'pca',
        '6': 'exit'
    }
    print("\nSelect the initialization method:")
    for key, value in inicializaciones.items():
        print(f"{key}. {value}")
    seleccion = input("Select (1-6): ").strip()
    while seleccion not in inicializaciones or seleccion == '6':
        if seleccion == '6':
            print("Exiting...")
            return
        print("Invalid input.")
        seleccion = input("Select (1-6): ").strip()
    init_method = inicializaciones[seleccion]
    print(f"✅ Initialization selected: {init_method}")

    # Evaluation method selection
    metodos = {
        '1': 'WCD',
        '2': 'BCD',
        '3': 'Fisher',
        '4': 'exit'
    }
    print("\nSelect the evaluation method:")
    for key, value in metodos.items():
        print(f"{key}. {value}")
    seleccion = input("Select (1-4): ").strip()
    while seleccion not in metodos or seleccion == '4':
        if seleccion == '4':
            print("Exiting...")
            return
        print("Invalid input.")
        seleccion = input("Select (1-4): ").strip()
    metodo = metodos[seleccion]
    print(f"✅ You selected: {metodo}")

    # K configuration
    k = None
    if metodo == 'WCD':
        opcion_k = input("\nDo you want to enter K manually (1) or find the best k automatically (2)? (1/2): ").strip()
        while opcion_k not in ['1', '2']:
            print("Invalid input.")
            opcion_k = input("Enter 1 or 2: ").strip()
        if opcion_k == '1':
            k = int(input("Enter the value of K (2-10): "))
            while k < 2 or k > 10:
                print("K must be between 2 and 10")
                k = int(input("Enter the value of K (2-10): "))
        else:
            modo_busqueda = input("Select the method to find the best K:\n"
                                  "1. By drop threshold (percentage)\n"
                                  "2. Elbow method\n"
                                  "Select (1/2): ").strip()
            while modo_busqueda not in ['1', '2']:
                print("Invalid input.")
                modo_busqueda = input("Select (1/2): ").strip()

            X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
            print("✅ Data prepared (reduced X)")
            km = KMeans(X, options={'km_init': init_method})

            if modo_busqueda == '1':
                threshold = input("Enter the threshold value (20% is standard): ") or "20"
                threshold = float(threshold)
                print(f"🔍 Searching for the best K with threshold={threshold}%...")
                km.find_bestK(max_K=10, method='threshold', threshold=threshold)
            else:
                print("📉 Searching for the best K using the elbow method...")
                km.find_bestK(max_K=10, method='elbow')

            k = km.K
            print(f"⚙️ Automatically selected K: {k}")
    else:
        opcion_k = input("\nDo you want to enter K manually (1) or search for the best k automatically (2)? (1/2): ").strip()
        while opcion_k not in ['1', '2']:
            print("Invalid input.")
            opcion_k = input("Enter 1 or 2: ").strip()
        if opcion_k == '1':
            k = int(input("Enter the value of K (2-10): "))
            while k < 2 or k > 10:
                print("K must be between 2 and 10")
                k = int(input("Enter the value of K (2-10): "))
        else:
            print(f"🔍 Searching for the best K for {metodo}...")
            X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
            print("✅ Data prepared (reduced X)")
            km = KMeans(X, options={'km_init': init_method})
            km.find_bestK(max_K=10, method=metodo)
            k = km.K
            print(f"⚙️ Automatically selected K: {k}")

    # Execute KMeans on each image
    print(f"\n🎨 Processing images with KMeans (K={k}, {init_method}, {metodo})...")

    color_preds = []
    color_percentages = []
    kmeans_objects = []

    for img in images:
        kmeans = KMeans(img, K=k, options={
            'km_init': init_method,
            'fitting': metodo.lower()
        })
        kmeans.fit()
        centroids = kmeans.centroids
        labels = get_colors(centroids)
        counts = np.bincount(kmeans.labels)
        percentages = counts / counts.sum()
        color_preds.append(labels)
        color_percentages.append(percentages.tolist())
        kmeans_objects.append((kmeans, img.shape))

    # Color accuracy
    accuracy = get_color_accuracy(color_preds, color_labels)
    print(f"\n🎯 Color prediction accuracy: {accuracy:.2f}%")

    # Search by color
    hacer_busqueda = input("\nDo you want to do a color search? (y/n): ").strip().lower()
    while hacer_busqueda == 'y':
        colores_disponibles = [
            'Red', 'Orange', 'Yellow', 'Green', 'Blue',
            'Purple', 'Pink', 'Brown', 'Black', 'White', 'Grey'
        ]
        print("\nAvailable colors:")
        for i, color in enumerate(colores_disponibles, 1):
            print(f"{i}. {color}")
        print("0. Exit")

        seleccion = input("Enter the color numbers separated by commas (e.g. 1,3,5): ").strip()
        if seleccion == '0':
            print("Exiting the color search.")
            break

        seleccionados = seleccion.split(',')
        colores_elegidos = []
        for s in seleccionados:
            if s.strip().isdigit():
                idx = int(s.strip()) - 1
                if 0 <= idx < len(colores_disponibles):
                    colores_elegidos.append(colores_disponibles[idx])
        if not colores_elegidos:
            print("❌ No valid color was selected.")
            continue

        print(f"\n🔎 Retrieving images with colors: {', '.join(colores_elegidos)}")

        visualizarRetrievalConKMeans(
            test_imgs=images,
            color_preds=color_preds,
            color_percentages=color_percentages,
            query_colors=colores_elegidos,
            color=True,
            n=8
        )

        for color_objetivo in colores_elegidos:
            visualizar_color = input(f"\nDo you want to visualize K-means for representative images with {color_objetivo}? (y/n): ").strip().lower()
            if visualizar_color != 'y':
                continue

            imagenes_con_color = []
            for i, (preds, percentages) in enumerate(zip(color_preds, color_percentages)):
                if color_objetivo in preds:
                    idx_color = preds.index(color_objetivo)
                    imagenes_con_color.append((i, percentages[idx_color]))

            if not imagenes_con_color:
                print(f"No images with color {color_objetivo} were found")
                continue

            imagenes_con_color.sort(key=lambda x: x[1], reverse=True)
            indices_mostrar = []
            if len(imagenes_con_color) >= 1:
                indices_mostrar.append(imagenes_con_color[0][0])
            if len(imagenes_con_color) >= 2:
                mid_idx = len(imagenes_con_color) // 2
                indices_mostrar.append(imagenes_con_color[mid_idx][0])
            if len(imagenes_con_color) >= 3:
                indices_mostrar.append(imagenes_con_color[-1][0])

            print(f"\n🖼️ Showing K-means for representative images with {color_objetivo}:")
            for idx in indices_mostrar:
                porcentaje = next(p for i, p in imagenes_con_color if i == idx)
                print(f"\nImage {idx+1} - Percentage of {color_objetivo}: {porcentaje*100:.1f}%")
                kmeans_obj, img_shape = kmeans_objects[idx]
                visualize_k_means(kmeans_obj, img_shape)
                plt.suptitle(f"Image {idx+1} - Color: {color_objetivo}\n"
                             f"Percentage: {porcentaje*100:.1f}% | K={k}, {init_method}, {metodo}")
                plt.show()

        hacer_busqueda = input("\nDo you want to search for other colors? (y/n): ").strip().lower()

    # Execution summary
    print("\n📋 EXECUTION SUMMARY:")
    print(f" - Images used: {origen}")
    print(f" - Evaluation method: {metodo}")
    if metodo == 'WCD' and 'threshold' in locals():
        print(f" - Threshold used: {threshold}%")
    print(f" - K value used: {k}")
    print(f" - Initialization: {init_method}")
    print(f" - Color accuracy: {accuracy:.2f}%")

def knn_interactivo(train_imgs, train_class_labels, test_imgs, test_class_labels,
                   cropped_train=None, cropped_test=None):
    """
    Interactive KNN function with the option to use original or cropped images.

    Args:
        train_imgs: Original training images
        train_class_labels: Training labels
        test_imgs: Original test images
        test_class_labels: Test labels
        cropped_train: (optional) Cropped training images
        cropped_test: (optional) Cropped test images
    """

    # 1. Select image type
    print("\n=== KNN IMAGE TYPE SELECTION ===")
    print("1. Use original images")
    if cropped_train is not None and cropped_test is not None:
        print("2. Use cropped images (cropped)")
    print("3. Exit")

    seleccion = input("Select an option (1-3): ").strip()
    while seleccion not in ['1', '2', '3'] or (seleccion == '2' and (cropped_train is None or cropped_test is None)):
        print("Invalid input.")
        seleccion = input("Select an option (1-3): ").strip()

    if seleccion == '3':
        print("Exiting...")
        return
    elif seleccion == '1':
        X_train = train_imgs
        X_test = test_imgs
        origen = "original"
        print("✅ Using original images.")
    else:
        X_train = cropped_train
        X_test = cropped_test
        origen = "cropped"
        print("ℹ️ Training on full images and querying with cropped ones (crop windows only exist for the extended dataset)")
        print("✅ Using cropped images (cropped).")

    print("\n🖼️ Total images loaded:")
    print(f" - Training: {len(X_train)}")
    print(f" - Test: {len(X_test)}")
    print(f" - Total: {len(X_train) + len(X_test)}")

    # 2. K value
    k = int(input("\nEnter the value of k (2-10): "))
    while k < 2 or k > 10:
        print("k must be between 2 and 10")
        k = int(input("Enter the value of k (2-10): "))

    # 3. Distance metric
    metricas = {'1': 'euclidean', '2': 'manhattan', '3': 'cosine', '4': 'exit'}
    print("\nSelect the distance metric:")
    for key, value in metricas.items():
        print(f"{key}. {value}")
    seleccion = input("Select (1-4): ").strip()
    if seleccion == '4':
        print("Exiting...")
        return
    metrica = metricas.get(seleccion, 'euclidean')

    # 4. Feature type
    tipos = {'1': 'raw', '2': 'mean_rgb', '3': 'mean_halves', '4': 'grayscale_mean', '5': 'exit'}
    print("\nSelect the feature type:")
    for key, value in tipos.items():
        print(f"{key}. {value}")
    seleccion = input("Select (1-5): ").strip()
    if seleccion == '5':
        print("Exiting...")
        return
    tipo = tipos.get(seleccion, 'raw')

    # 5. KNN classification
    print("\n🧠 Classifying images with KNN...")
    knn = KNN(X_train, train_class_labels)
    knn.distance = metrica
    knn.feature_type = tipo

    predicted = knn.predict(X_test, k=k)

    # 6. Confidence (shape_votes)
    shape_votes = []
    for i in range(len(X_test)):
        neighbors = knn.neighbors[i]
        pred = predicted[i]
        vote_ratio = list(neighbors).count(pred) / len(neighbors)
        shape_votes.append(vote_ratio)

    # 7. Evaluation
    accuracy, errors = get_shape_accuracy(predicted, test_class_labels, return_errors=True)
    print(f"\n🎯 Classification accuracy: {accuracy:.2f}%")

    if errors:
        print("❌ Most frequent errors (GT → Prediction):")
        for (gt, pred), count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
            print(f" - {gt} → {pred}: {count} times")
    else:
        print("✅ No errors found.")

    # 8. Shape search
    hacer_busqueda = input("\nDo you want to search by shape? (y/n): ").strip().lower()
    while hacer_busqueda == 'y':
        formas_disponibles = {
            '1': 'Dresses', '2': 'Flip Flops', '3': 'Jeans', '4': 'Sandals',
            '5': 'Shirts', '6': 'Shorts', '7': 'Socks', '8': 'Handbags', '9': 'Heels', 'q': 'exit'
        }

        print("\nSelect the shape you want to search for:")
        for key, value in formas_disponibles.items():
            print(f"{key}. {value}")
        seleccion_forma = input("Enter the corresponding number (1-9): ").strip()
        if seleccion_forma == 'q':
            print("Exiting the shape search.")
            return

        forma_objetivo = formas_disponibles.get(seleccion_forma)
        if not forma_objetivo:
            print("Invalid shape.")
            return

        print(f"\n🔎 Retrieving images with shape: {forma_objetivo}...")

        visualizarRetrievalConKMeans(
            test_imgs=X_test,
            predicted_shapes=predicted,
            query_shape=forma_objetivo,
            shape_votes=shape_votes,
            forma=True,
            n=8
        )
        hacer_busqueda = input("\nDo you want to search for other shapes? (y/n): ").strip().lower()

    # 9. Final summary
    print("\n📋 EXECUTION SUMMARY:")
    print(f" - Image type: {origen}")
    print(f" - K value: {k}")
    print(f" - Distance metric used: {metrica}")
    print(f" - Feature type: {tipo}")
    print(f" - Shape accuracy: {accuracy:.2f}%")
    
    
def buscador_interactivo(test_imgs, cropped_test_resized, test_color_labels, test_class_labels,
                         train_imgs, train_class_labels, 
                         cropped_train_resized,
                         es_test_set=False):
    print("\n=== INTERACTIVE SEARCH ===")

    # === Step 1: Cropped? ===
    usar_cropped = input("Do you want to use cropped images (cropped)? (y/n): ").strip().lower() == 'y'
    if usar_cropped and cropped_test_resized is None:
        print("ℹ️ No crop windows exist for the test set; using the full images.")
        usar_cropped = False

    images = cropped_test_resized if usar_cropped else test_imgs
    train_images = cropped_train_resized if usar_cropped else train_imgs

    # === Step 2: Search type ===
    print("\nSelect search type:")
    print("1. By color")
    print("2. By shape")
    print("3. By both (color + shape)")
    tipo_busqueda = input("Option (1-3): ").strip()
    while tipo_busqueda not in ['1', '2', '3']:
        tipo_busqueda = input("Invalid input. Option (1-3): ").strip()

    usar_color = tipo_busqueda in ['1', '3']
    usar_forma = tipo_busqueda in ['2', '3']

    # === Step 3: Color selection ===
    colores_disponibles = [
        'Red', 'Orange', 'Yellow', 'Green', 'Blue',
        'Purple', 'Pink', 'Brown', 'Black', 'White', 'Grey'
    ]
    colores = []
    if usar_color:
        print("\nAvailable colors:")
        for i, color in enumerate(colores_disponibles, 1):
            print(f"{i}. {color}")
        seleccionados = input("Enter color numbers separated by commas (e.g. 1,3,5): ").strip().split(',')
        for s in seleccionados:
            if s.strip().isdigit():
                idx = int(s.strip()) - 1
                if 0 <= idx < len(colores_disponibles):
                    colores.append(colores_disponibles[idx])
        if not colores:
            print("❌ No valid color was selected.")
            return

    # === Step 4: Shape selection ===
    formas_disponibles = {
        '1': 'Dresses', '2': 'Flip Flops', '3': 'Jeans', '4': 'Sandals',
        '5': 'Shirts', '6': 'Shorts', '7': 'Socks', '8': 'Handbags',
        '9': 'Heels'
    }
    forma = None
    if usar_forma:
        print("\nAvailable shapes:")
        for key, val in formas_disponibles.items():
            print(f"{key}. {val}")
        seleccion = input("Select a shape (1-9): ").strip()
        forma = formas_disponibles.get(seleccion)
        if not forma:
            print("❌ Invalid shape.")
            return

    # === Step 5: Advanced configuration ===
    avanzada = input("Do you want to use advanced configuration? (y/n): ").strip().lower() == 'y'

    color_preds = None
    color_percentages = None
    shape_preds = None
    shape_votes = None
    k_final = 3  # default value

    if avanzada:
        # KMeans initialization
        inicializaciones = {
            '1': 'first', '2': 'random', '3': 'custom',
            '4': 'kmeans++', '5': 'pca'
        }
        print("\nSelect the KMeans initialization method:")
        for key, val in inicializaciones.items():
            print(f"{key}. {val}")
        sel_init = input("Option (1-5): ").strip()
        while sel_init not in inicializaciones:
            sel_init = input("Invalid input. Option (1-5): ").strip()
        init_method = inicializaciones[sel_init]

        # Evaluation
        metodos = {
            '1': 'WCD', '2': 'BCD', '3': 'Fisher'
        }
        print("\nSelect the KMeans evaluation method:")
        for key, val in metodos.items():
            print(f"{key}. {val}")
        sel_eval = input("Option (1-3): ").strip()
        while sel_eval not in metodos:
            sel_eval = input("Invalid input. Option (1-3): ").strip()
        metodo = metodos[sel_eval]

        # K value
        opcion_k = input("\nDo you want to enter K manually (1) or use automatic search (2)? (1/2): ").strip()
        while opcion_k not in ['1', '2']:
            opcion_k = input("Enter 1 or 2: ").strip()
        if opcion_k == '1':
            k_final = int(input("Enter the value of K (2-10): "))
            while k_final < 2 or k_final > 10:
                k_final = int(input("K must be between 2 and 10: "))
        else:
            if es_test_set:
                print("\nℹ️ Test dataset detected — using fixed K=3 to avoid heavy computations.")
                k_final = 3
            else:
                print("\n🔍 Running find_bestK...")
                X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
                km_tmp = KMeans(X, options={'km_init': init_method})

                if metodo == 'wcd':
                    km_tmp.find_bestK(max_K=10, method='WCD', threshold=20)
                else:
                    km_tmp.find_bestK(max_K=10, method=metodo)

                k_final = km_tmp.K
                print(f"✅ Automatically detected K: {k_final}")

        # Run KMeans if needed
        if usar_color:
            color_preds = []
            color_percentages = []
            for img in images:
                km = KMeans(img, K=k_final, options={'km_init': init_method, 'fitting': metodo})
                km.fit()
                centroids = km.centroids
                labels = get_colors(centroids)
                counts = np.bincount(km.labels)
                percentages = counts / counts.sum()
                color_preds.append(labels)
                color_percentages.append(percentages.tolist())

        # Run KNN if needed
        if usar_forma:
            print(f"\n🧠 Running KNN with k={k_final}...")
            metricas = {'1': 'euclidean', '2': 'manhattan', '3': 'cosine'}
            tipos = {'1': 'raw', '2': 'mean_rgb', '3': 'mean_halves', '4': 'grayscale_mean'}

            print("\nSelect the distance metric for KNN:")
            for key, val in metricas.items():
                print(f"{key}. {val}")
            sel_met = input("Option (1-3): ").strip()
            metrica = metricas.get(sel_met, 'euclidean')

            print("\nSelect the feature type:")
            for key, val in tipos.items():
                print(f"{key}. {val}")
            sel_tipo = input("Option (1-4): ").strip()
            tipo = tipos.get(sel_tipo, 'raw')

            knn = KNN(train_images, train_class_labels)
            knn.distance = metrica
            knn.feature_type = tipo
            shape_preds = knn.predict(images, k=k_final)
            shape_votes = []
            for i in range(len(images)):
                neighbors = knn.neighbors[i]
                pred = shape_preds[i]
                vote_ratio = list(neighbors).count(pred) / len(neighbors)
                shape_votes.append(vote_ratio)

    else:
        # Default mode with find_bestK (color) + KNN with same K
        if usar_color:
            if es_test_set:
                print("\nℹ️ Test dataset detected — using fixed K=3 to avoid heavy computations.")
                k_final = 3
            else:
                print("\n🔍 Running default KMeans with find_bestK (WCD)...")
                X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
                km_global = KMeans(X, options={'km_init': 'first'})
                km_global.find_bestK(max_K=10, method='WCD', threshold=20)
                k_final = km_global.K
                print(f"✅ Automatically detected K: {k_final}")

            color_preds = []
            color_percentages = []
            for img in images:
                km = KMeans(img, K=k_final, options={'km_init': 'first', 'fitting': 'wcd'})
                km.fit()
                centroids = km.centroids
                labels = get_colors(centroids)
                counts = np.bincount(km.labels)
                percentages = counts / counts.sum()
                color_preds.append(labels)
                color_percentages.append(percentages.tolist())

        if usar_forma:
            print(f"\n🧠 Ejecutando KNN con k={k_final} por defecto...")
            knn = KNN(train_images, train_class_labels)
            knn.distance = 'euclidean'
            knn.feature_type = 'raw'
            shape_preds = knn.predict(images, k=k_final)
            shape_votes = []
            for i in range(len(images)):
                neighbors = knn.neighbors[i]
                pred = shape_preds[i]
                vote_ratio = list(neighbors).count(pred) / len(neighbors)
                shape_votes.append(vote_ratio)

    # === Final visualization ===
    visualizarRetrievalConKMeans(
        test_imgs=images,
        predicted_shapes=shape_preds,
        color_preds=color_preds,
        color_percentages=color_percentages,
        shape_votes=shape_votes,
        query_shape=forma,
        query_colors=colores,
        color=usar_color,
        forma=usar_forma,
        ambos=(tipo_busqueda == '3'),
        n=4
    )

def compare_find_bestK_heuristics(images, max_K=10, color_labels=[]):
    """
    Compare find_bestK across different heuristics (WCD, BCD, Fisher, Elbow),
    measure search time, run KMeans with the best K, and compute the accuracy.

    Args:
        images (list): List of images
        max_K (int): Maximum K value to test
        color_labels (list): Ground-truth color labels (optional)
    """

    heuristics = ['WCD', 'BCD', 'Fisher', 'elbow']
    X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
    results = {}

    heuristic_scores = {}

    for heur in heuristics:
        print(f"\n➡️ Finding the best K using {heur}...")
        start = time.time()

        km = KMeans(X, options={'km_init': 'first'})
        if heur == 'WCD':
            km.find_bestK(max_K=max_K, method='WCD', threshold=20)
        else:
            km.find_bestK(max_K=max_K, method=heur)

        elapsed = time.time() - start
        best_K = km.K
        print(f"   ✅ Best K = {best_K} (time: {elapsed:.2f}s)")

        # Store heuristic values by K for plotting
        k_values = list(range(2, max_K + 1))
        scores = []
        for K in k_values:
            model = KMeans(X, K=K, options={'km_init': 'first'})
            model.fit()

            if heur == 'WCD' or heur == 'elbow':
                value = model.withinClassDistance()
            elif heur == 'Fisher':
                value = model.fisher_score()
            elif heur == 'BCD':
                value = model.betweenClassDistance()

            scores.append(value)
        heuristic_scores[heur] = (k_values, scores)

        # Run KMeans with the best K for each image
        color_preds = []
        color_percentages = []

        for img in images:
            kmeans = KMeans(img, K=best_K, options={
                'km_init': 'first'
            })
            kmeans.fit()
            centroids = kmeans.centroids
            labels = get_colors(centroids)
            counts = np.bincount(kmeans.labels)
            percentages = counts / counts.sum()
            color_preds.append(labels)
            color_percentages.append(percentages.tolist())

        # 5. Color accuracy
        accuracy = get_color_accuracy(color_preds, color_labels)
        print(f"\n🎯 Predicted color accuracy: {accuracy:.2f}%")

        results[heur] = {
            'best_K': best_K,
            'time': elapsed,
            'accuracy': accuracy,
            'scores': {'K_values': k_values, 'values': scores}
        }

    # Final summary
    print("\n📋 FINAL SUMMARY:")
    for heur, res in results.items():
        print(f"- {heur.upper():7}: K = {res['best_K']}, Time = {res['time']:.2f}s, Accuracy = {res['accuracy']:.2f}%")
    # --------------------------------------------------
    # Linear chart: heuristic vs number of clusters
    # --------------------------------------------------
    plt.figure(figsize=(10, 6))
    for heur, (k_vals, scores) in heuristic_scores.items():
        plt.plot(k_vals, scores, marker='o', label=heur)

    plt.title('Comparison of heuristics vs number of clusters K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Heuristic value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # --------------------------------------------------
    # Second chart: Comparison of metrics (time and accuracy)
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))

    methods = list(results.keys())
    times = [results[heur]['time'] for heur in methods]
    accuracies = [results[heur]['accuracy'] for heur in methods]

    plt.subplot(1, 2, 1)
    plt.bar(methods, times, color=['blue', 'green', 'red', 'purple'])
    plt.title('Execution time by heuristic')
    plt.ylabel('Time (s)')

    plt.subplot(1, 2, 2)
    plt.bar(methods, accuracies, color=['blue', 'green', 'red', 'purple'])
    plt.title('Color accuracy by heuristic')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()

    return results


def _stage_is_current(path, require_protocol):
    """True if the JSON exists and (when required) was produced with the current PROTOCOL."""
    if not os.path.exists(path):
        return False
    if not require_protocol:
        return True
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('protocol') == PROTOCOL
    except Exception:
        return False


def _run_stage(name, fn, skip_existing, summary, require_protocol=False):
    """Runs one stage of the full report; a failure never loses the previous stages."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if skip_existing and _stage_is_current(path, require_protocol):
        print(f"\n⏭️  Skipping '{name}' (results/{name}.json already exists)")
        summary['stages'][name] = {'status': 'skipped'}
        return
    if skip_existing and os.path.exists(path):
        print(f"\n♻️  '{name}': existing JSON was produced with an older evaluation protocol -> recomputing")
    print(f"\n{'=' * 60}\n▶ {name}\n{'=' * 60}")
    t0 = time.time()
    try:
        data = fn()
        elapsed = time.time() - t0
        save_json(name, {'stage': name, 'protocol': PROTOCOL, 'elapsed_s': elapsed,
                         'generated_at': datetime.now().isoformat(timespec='seconds'),
                         'data': data})
        summary['stages'][name] = {'status': 'ok', 'elapsed_s': round(elapsed, 1)}
    except Exception as e:
        traceback.print_exc()
        plt.close('all')
        summary['stages'][name] = {'status': 'failed', 'error': repr(e)}
    save_json('summary', summary)


def _knn_summary(results):
    """Aggregates the per-K results of knn_systematic_analysis by (metric, features)."""
    groups = defaultdict(lambda: {'acc': [], 'time': []})
    for r in results:
        groups[(r['metric'], r['features'])]['acc'].append(r['accuracy'])
        groups[(r['metric'], r['features'])]['time'].append(r['time'])
    rows = [{'metric': m, 'features': f,
             'mean_accuracy': float(np.mean(v['acc'])),
             'mean_time_s': float(np.mean(v['time']))} for (m, f), v in groups.items()]
    rows.sort(key=lambda r: -r['mean_accuracy'])
    best = max(results, key=lambda r: r['accuracy'])
    return {'by_combination': rows, 'best_single_run': best}


def _retrieval_examples(images, train_images, train_labels, tag, knn_k=2):
    """
    Saves the qualitative retrieval figures (color, shape and combined queries)
    as figures/retrieval_<tag>_<query>.png. Color labels come from KMeans (K chosen
    with find_bestK / WCD, 'first' init); shape labels come from KNN (raw, euclidean).
    """
    X = np.vstack([img.reshape(-1, 3) for img in images]).astype(np.float32)
    km_global = KMeans(X, options={'km_init': 'first'})
    km_global.find_bestK(max_K=10, method='WCD', threshold=20)
    K = km_global.K

    color_preds, color_percentages = [], []
    for img in images:
        km = KMeans(img, K=K, options={'km_init': 'first', 'fitting': 'wcd'})
        km.fit()
        color_preds.append(get_colors(km.centroids))
        counts = np.bincount(km.labels)
        color_percentages.append((counts / counts.sum()).tolist())

    knn = KNN(train_images, train_labels)
    knn.distance = 'euclidean'
    knn.feature_type = 'raw'
    shape_preds = knn.predict(images, k=knn_k)
    shape_votes = [list(knn.neighbors[i]).count(shape_preds[i]) / len(knn.neighbors[i])
                   for i in range(len(images))]

    queries = [
        ('color-blue',            dict(color=True,  query_colors=['Blue']), 8),
        ('color-black',           dict(color=True,  query_colors=['Black']), 20),
        ('shape-flip-flops',      dict(color=False, forma=True, query_shape='Flip Flops'), 20),
        ('shape-handbags',        dict(color=False, forma=True, query_shape='Handbags'), 8),
        ('combined-shorts-blue',  dict(color=True,  ambos=True, query_shape='Shorts', query_colors=['Blue']), 4),
    ]
    for name, kwargs, n in queries:
        with capture_figures(f"retrieval_{tag}", names=[name]):
            visualizarRetrievalConKMeans(
                test_imgs=images, predicted_shapes=shape_preds, color_preds=color_preds,
                color_percentages=color_percentages, shape_votes=shape_votes, n=n, **kwargs)

    return {'kmeans_K': K, 'knn_k': knn_k, 'knn_metric': 'euclidean', 'knn_features': 'raw',
            'knn_training': 'full training images, images overlapping the query set removed',
            'queries': [q[0] for q in queries], 'n_images': len(images)}


def knn_crossval_analysis(full_imgs, cropped_imgs, labels, n_folds=5, k_values=None, seed=0):
    """
    Like-for-like comparison of FULL vs CROPPED images for shape classification.

    Crop windows only exist for the 180 images of the extended dataset, so the classifier is
    trained and tested on those 180 images with stratified n-fold cross-validation (a test
    image is never in the training folds). Three conditions:
      - 'full->full'         train and query with full images
      - 'cropped->cropped'   train and query with cropped images
      - 'full->cropped'      train with full images, query with cropped ones
    Returns a dict with every (condition, metric, features, K) accuracy and a summary.
    """
    k_values = list(k_values or range(2, 11))
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.RandomState(seed)

    fold_of = np.zeros(n, dtype=int)                      # stratified fold assignment
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[i] = j % n_folds

    conditions = {
        'full->full': (full_imgs, full_imgs),
        'cropped->cropped': (cropped_imgs, cropped_imgs),
        'full->cropped': (full_imgs, cropped_imgs),
    }
    metrics = ['euclidean', 'manhattan', 'cosine']
    feature_types = ['raw', 'mean_rgb', 'mean_halves', 'grayscale_mean']
    kmax = max(k_values)

    correct = defaultdict(int)                            # (cond, metric, feat, k) -> hits
    for cond, (tr_src, te_src) in conditions.items():
        tr_src, te_src = np.asarray(tr_src), np.asarray(te_src)
        for metric in metrics:
            for feat in feature_types:
                print(f"CV {cond}: {metric}, {feat}")
                for fold in range(n_folds):
                    te = np.where(fold_of == fold)[0]
                    tr = np.where(fold_of != fold)[0]
                    knn = KNN(tr_src[tr], labels[tr])
                    knn.distance = metric
                    knn.feature_type = feat
                    knn.get_k_neighbours(te_src[te], kmax)   # neighbours sorted by distance
                    for k in k_values:
                        # same voting / tie-breaking as KNN.get_class, restricted to the first k
                        preds = np.array([max(x[:k], key=list(x[:k]).count) for x in knn.neighbors])
                        correct[(cond, metric, feat, k)] += int((preds == labels[te]).sum())

    results = [{'condition': c, 'metric': m, 'features': f, 'k': k, 'accuracy': 100.0 * v / n}
               for (c, m, f, k), v in correct.items()]

    groups = defaultdict(list)
    for r in results:
        groups[(r['condition'], r['metric'], r['features'])].append(r['accuracy'])
    summary = [{'condition': c, 'metric': m, 'features': f, 'mean_accuracy': float(np.mean(v))}
               for (c, m, f), v in groups.items()]
    summary.sort(key=lambda r: (r['condition'], -r['mean_accuracy']))

    # --- Figure 1: raw + euclidean, accuracy vs K for the three conditions
    plt.figure(figsize=(9, 5.5))
    for cond in conditions:
        ys = [next(r['accuracy'] for r in results if r['condition'] == cond and r['metric'] == 'euclidean'
                   and r['features'] == 'raw' and r['k'] == k) for k in k_values]
        plt.plot(k_values, ys, marker='o', label=cond)
    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Full vs cropped images ({n_folds}-fold CV, raw features, euclidean)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: mean accuracy (over K) by metric/features and condition
    combos = [(m, f) for f in feature_types for m in metrics]
    width = 0.27
    plt.figure(figsize=(13, 6))
    for i, cond in enumerate(conditions):
        vals = [next(s['mean_accuracy'] for s in summary
                     if s['condition'] == cond and s['metric'] == m and s['features'] == f) for m, f in combos]
        plt.bar(np.arange(len(combos)) + (i - 1) * width, vals, width, label=cond)
    plt.xticks(np.arange(len(combos)), [f"{m}\n{f}" for m, f in combos], rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean accuracy over K (%)')
    plt.title(f'Full vs cropped images: mean accuracy by metric/features ({n_folds}-fold CV)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return {'n_images': int(n), 'n_folds': n_folds, 'k_values': k_values,
            'conditions': list(conditions), 'results': results, 'summary': summary}


def generate_full_report(imgs, class_labels, color_labels, cropped_resized,
                         train_ext, train_labels_ext, train_tst, train_labels_tst,
                         test_imgs, test_class_labels, test_color_labels,
                         overlap=None, skip_existing=False):
    """
    Option 9: runs EVERY analysis and saves
      - all figures  -> figures/*.png
      - all results  -> results/*.json  (+ results/summary.json)
    It can take hours. Each stage is saved as soon as it finishes, and with
    skip_existing=True a stage whose JSON already exists is not recomputed
    (stages affected by the evaluation correction are recomputed if their JSON is older).

    train_ext / train_tst: training set WITHOUT the images that also appear in `imgs` /
    `test_imgs` (see remove_overlap).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    overlap = overlap or {}
    t_start = time.time()
    summary = {'started_at': datetime.now().isoformat(timespec='seconds'),
               'protocol': PROTOCOL,
               'train_test_overlap_removed': overlap,
               'dataset_sizes': {'train_without_imgs_overlap': len(train_ext),
                                 'train_without_test_overlap': len(train_tst),
                                 'imgs': len(imgs), 'test_imgs': len(test_imgs),
                                 'imgs_cropped': len(cropped_resized)},
               'stages': {}}
    save_json('summary', summary)

    # (tag, kmeans images, kmeans GT, knn train, knn train labels, knn queries, knn query labels)
    # 'imgs-cropped' for KNN = train on FULL images, query with the cropped ones (no crop windows
    # exist for the training set); the like-for-like comparison is the cross-validation stage.
    datasets = [
        ('imgs',         imgs,            color_labels,      train_ext, train_labels_ext, imgs,            class_labels),
        ('imgs-cropped', cropped_resized, color_labels,      train_ext, train_labels_ext, cropped_resized, class_labels),
        ('test-imgs',    test_imgs,       test_color_labels, train_tst, train_labels_tst, test_imgs,       test_class_labels),
    ]

    # Dataset sample grid
    def _samples():
        idx = random.Random(0).sample(range(len(imgs)), min(16, len(imgs)))
        with capture_figures('dataset', names=['samples']):
            visualize_retrieval(np.array(imgs)[idx], topN=len(idx))
        return {'n_samples': len(idx)}
    _run_stage('dataset_samples', _samples, skip_existing, summary)

    for tag, km_imgs, km_gt, tr_x, tr_y, te_x, te_y in datasets:
        # --- KMeans systematic analysis ---
        def _km(km_imgs=km_imgs, km_gt=km_gt, tag=tag):
            with capture_figures(f"kmeans_{tag}", names=['heuristic', 'accuracy', 'time', 'iterations']):
                return kmeans_analysis(km_imgs, km_gt, Kmax=10)
        _run_stage(f"kmeans_{tag}", _km, skip_existing, summary)

        # --- KNN systematic analysis ---
        def _knn(tr_x=tr_x, tr_y=tr_y, te_x=te_x, te_y=te_y, tag=tag):
            with capture_figures(f"knn_{tag}", names=['avg-accuracy', 'accuracy-vs-k', 'time-vs-k', 'avg-time']):
                results = knn_systematic_analysis(tr_x, tr_y, te_x, te_y)
            return {'runs': results, 'summary': _knn_summary(results),
                    'train_size': int(len(tr_x)), 'test_size': int(len(te_x))}
        needs_protocol = (tag != 'test-imgs') or overlap.get('test_imgs', 0) > 0
        _run_stage(f"knn_{tag}", _knn, skip_existing, summary, require_protocol=needs_protocol)

    # --- Full vs cropped, like-for-like (cross-validation on the 180 extended images) ---
    def _cv():
        with capture_figures('knn_crossval_imgs', names=['raw-euclidean-vs-k', 'avg-accuracy']):
            return knn_crossval_analysis(imgs, cropped_resized, class_labels)
    _run_stage('knn_crossval_imgs', _cv, skip_existing, summary, require_protocol=True)

    # --- find_bestK heuristics comparison (extended dataset, full and cropped) ---
    # (add ('test-imgs', test_imgs, test_color_labels) to this list to also run it on the test set)
    for tag, im, gt in [('imgs', imgs, color_labels), ('imgs-cropped', cropped_resized, color_labels)]:
        def _bk(im=im, gt=gt, tag=tag):
            with capture_figures(f"find-bestk_{tag}", names=['heuristics-vs-k', 'time-and-accuracy']):
                return compare_find_bestK_heuristics(im, max_K=10, color_labels=gt)
        _run_stage(f"find-bestk_{tag}", _bk, skip_existing, summary)

    # --- Qualitative retrieval examples (full vs cropped); KNN is always trained on full images ---
    _run_stage('retrieval_full',
               lambda: _retrieval_examples(imgs, train_ext, train_labels_ext, 'full'),
               skip_existing, summary, require_protocol=True)
    _run_stage('retrieval_cropped',
               lambda: _retrieval_examples(cropped_resized, train_ext, train_labels_ext, 'cropped'),
               skip_existing, summary, require_protocol=True)

    summary['finished_at'] = datetime.now().isoformat(timespec='seconds')
    summary['total_elapsed_s'] = round(time.time() - t_start, 1)
    save_json('summary', summary)
    failed = [k for k, v in summary['stages'].items() if v['status'] == 'failed']
    print(f"\n✅ Done in {summary['total_elapsed_s'] / 60:.1f} min. Figures -> {FIGURES_DIR}/ , results -> {RESULTS_DIR}/")
    if failed:
        print(f"⚠️ Stages that failed: {', '.join(failed)} (see results/summary.json). Re-run option 9 with 'skip' = y to retry only those.")

if __name__ == '__main__':

    print("\n🔃 Loading dataset...")
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existing classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    # NOTE: the crop windows (upper/lower) only exist for the extended dataset (imgs), so cropping
    # is only valid for it. (Earlier versions also cropped train_imgs / test_imgs with these windows,
    # which paired every image with the window of a different one: see report, section 7.)
    cropped_images = crop_images(imgs, upper, lower)

    print("🖼️ Resizing images...")
    cropped_resized = resize_images(cropped_images, (80, 60, 3))
    cropped_extended_resized = cropped_resized   # same array; kept as an alias for the menu code

    # The extended dataset (imgs) is read from images/train, so its images are also in the KNN
    # training set. Remove them from the training set before using imgs as a test set.
    train_imgs_ext, train_class_labels_ext, n_overlap_ext = remove_overlap(
        train_imgs, train_class_labels, imgs, 'imgs')
    train_imgs_tst, train_class_labels_tst, n_overlap_tst = remove_overlap(
        train_imgs, train_class_labels, test_imgs, 'test_imgs')

    
    print(f"Number of images: {len(imgs)}")
    print(f"Number of test images: {len(test_imgs)}")
    print(f"Number of cropped images: {len(cropped_images)}")
    print(f"Number of resized cropped images: {len(cropped_resized)}")

    while True:
        print("\n=== MAIN MENU ===")
        print("1. Run systematic KMeans (full analysis)")
        print("2. Run systematic KNN (full analysis)")
        print("3. Run interactive KMeans")
        print("4. Run interactive KNN")
        print("5. Search")
        print("6. Visualize dataset images")
        print("7. Compare find_bestK methods")
        print("8. Exit")
        print("9. Generate ALL figures and results (saved to figures/ and results/)")
        
        opcion = input("\nSelect an option (1-9): ").strip()
        
        if opcion == '1':
            print("\n=== SYSTEMATIC KMEANS ANALYSIS ===")
            print("⚠️ WARNING: This process may take several minutes")
            print("It will analyze multiple configurations for K from 2 to 10")
            confirmacion = input("Continue? (y/n): ").strip().lower()
            
            if confirmacion == 'y':
                usar_crop = input("Use cropped images (cropped) (only applicable to imgs)? (y/n): ").strip().lower() == 'y'
                
                print("\n1. Use extended dataset")
                print("2. Use test set")
                sub_op = input("Select (1-2): ").strip()
        
                if sub_op == '1':
                    imgs_input = cropped_resized if usar_crop else imgs
                    kmeans_analysis(imgs_input, color_labels, Kmax=10)
                elif sub_op == '2':
                    kmeans_analysis(test_imgs, test_color_labels, Kmax=10)
                else:
                    print("❌ Invalid option")
            else:
                print("❌ Analysis cancelled")
                
        elif opcion == '2':
            print("\n=== SYSTEMATIC KNN ANALYSIS ===")
            print("⚠️ WARNING: This process may take several minutes")
            confirmacion = input("Continue? (y/n): ").strip().lower()
            
            if confirmacion == 'y':
                usar_crop = input("Use cropped images (cropped) (only applicable to imgs)? (y/n): ").strip().lower() == 'y'
        
                print("\n1. Use full dataset (imgs)")
                print("2. Use only the test set (test_imgs)")
                sub_op = input("Select (1-2): ").strip()
        
                if sub_op == '1':
                    # train on full images (overlap removed); cropped = query with cropped images
                    Xtest = cropped_extended_resized if usar_crop else imgs
                    knn_systematic_analysis(train_imgs_ext, train_class_labels_ext, Xtest, class_labels)

                elif sub_op == '2':
                    if usar_crop:
                        print("ℹ️ No crop windows exist for the test set; using the full images.")
                    knn_systematic_analysis(train_imgs_tst, train_class_labels_tst, test_imgs, test_class_labels)
                else:
                    print("❌ Invalid option")
            else:
                print("❌ Analysis cancelled")
                
        elif opcion == '3':
            print("\n=== INTERACTIVE KMEANS ===")
            print("1. Use extended dataset (imgs)")
            print("2. Use test set (test_imgs)")
            sub_op = input("Select (1-2): ").strip()
            
            if sub_op == '1':
                kmeans_interactivo(imgs, cropped_resized, color_labels)
            elif sub_op == '2':
                kmeans_interactivo(test_imgs, None, test_color_labels)
            else:
                print("Invalid option, returning to the main menu")
                
        elif opcion == '4':
            print("\n=== INTERACTIVE KNN ===")
            print("1. Use extended dataset (imgs)")
            print("2. Use test set (test_imgs)")
            sub_op = input("Select (1-2): ").strip()
            
            if sub_op == '1':
                knn_interactivo(
                    train_imgs_ext, train_class_labels_ext,
                    imgs, class_labels,
                    train_imgs_ext, cropped_extended_resized   # cropped option: train full, query cropped
                )
            elif sub_op == '2':
                knn_interactivo(
                    train_imgs_tst, train_class_labels_tst,
                    test_imgs, test_class_labels,
                    None, None                                  # no crop windows for the test set
                )
            else:
                print("Invalid option, returning to the main menu")
                
        elif opcion == '5':
            print("1. Search in extended dataset (imgs)")
            print("2. Search in test set (test_imgs)")
            sub_op = input("Select (1-2): ").strip()
            
            if sub_op == '1':
                buscador_interactivo(
                    test_imgs=imgs,
                    cropped_test_resized=cropped_resized,
                    test_color_labels=color_labels,
                    test_class_labels=class_labels,
                    train_imgs=train_imgs_ext,
                    train_class_labels=train_class_labels_ext,
                    cropped_train_resized=train_imgs_ext,   # cropped: train full, query cropped
                    es_test_set=False
                )
            elif sub_op == '2':
                buscador_interactivo(
                    test_imgs=test_imgs,
                    cropped_test_resized=None,               # no crop windows for the test set
                    test_color_labels=test_color_labels,
                    test_class_labels=test_class_labels,
                    train_imgs=train_imgs_tst,
                    train_class_labels=train_class_labels_tst,
                    cropped_train_resized=None,
                    es_test_set=True
                )
            else:
                print("Invalid option, returning to the main menu")

        elif opcion == '6':
            print("\n=== DATASET VISUALIZATION ===")
            print("1. Visualize extended dataset (imgs)")
            print("2. Visualize test set (test_imgs)")
            sub_op = input("Select (1-2): ").strip()
            n = int(input("\nTotal images to display: "))
            
            if sub_op == '1':
                print(f"\nTotal extended images: {len(imgs)}")
                sample = random.sample(range(len(imgs)), min(n, len(imgs)))
                visualize_retrieval(np.array(imgs)[sample], topN=len(sample))
            elif sub_op == '2':
                print(f"\nTotal test images: {len(test_imgs)}")
                sample = random.sample(range(len(test_imgs)), min(n, len(test_imgs)))
                visualize_retrieval(np.array(test_imgs)[sample], topN=len(sample))
            else:
                print("Invalid option, returning to the main menu")
                
        elif opcion == '7':
            print("\n=== COMPARISON OF find_bestK METHODS ===")
            print("1. Use extended dataset (imgs)")
            print("2. Use test set (test_imgs)")
            sub_op = input("Select (1-2): ").strip()

            if sub_op == '1':
                compare_find_bestK_heuristics(imgs, max_K=10, color_labels=color_labels)
            elif sub_op == '2':
                compare_find_bestK_heuristics(test_imgs, max_K=10, color_labels=test_color_labels)
            else:
                print("Invalid option")

        elif opcion == '8':
            print("Exiting the program...")
            break

        elif opcion == '9':
            print("\n=== GENERATE ALL FIGURES AND RESULTS ===")
            print("⚠️ WARNING: this runs every analysis (KMeans, KNN, find_bestK, retrieval examples)")
            print("   on all datasets and can take SEVERAL HOURS. Nothing is shown on screen:")
            print(f"   figures are saved to {FIGURES_DIR}/ and results to {RESULTS_DIR}/ (JSON).")
            confirmacion = input("Continue? (y/n): ").strip().lower()
            if confirmacion == 'y':
                skip = input("Skip stages whose JSON already exists (resume a previous run)? (y/n): ").strip().lower() == 'y'
                generate_full_report(
                    imgs, class_labels, color_labels, cropped_resized,
                    train_imgs_ext, train_class_labels_ext,
                    train_imgs_tst, train_class_labels_tst,
                    test_imgs, test_class_labels, test_color_labels,
                    overlap={'imgs': n_overlap_ext, 'test_imgs': n_overlap_tst},
                    skip_existing=skip
                )
            else:
                print("❌ Cancelled")

        else:
            print("Invalid option. Please select an option from 1 to 9.")
   

