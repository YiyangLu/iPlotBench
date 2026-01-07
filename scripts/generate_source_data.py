#!/usr/bin/env python
"""Generate source data for iPlotBench (FigureQA-compatible format).

This is a Python 3 port of FigureQA's source_data_generation.py.
Both data generation AND question generation match the original FigureQA.

Question types (15 total):
  - Bar/Pie (0-5): is_min, is_max, less_than, greater_than, low_median, high_median
  - Line (6-14): auc_min, auc_max, smoothest, roughest, lowest_value, highest_value,
                 less_than, greater_than, intersect
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import auc

# ============ Configuration (from FigureQA YAML) ============

# Data generation config (from color_scheme1_source_data.yaml)
DATA_CONFIG = {
    'vbar_categorical': {
        'y_range': [0, 99],
        'n_points_range': [2, 10],
        'x_distn': ['linear'],
        'shape': ['random', 'random', 'random', 'random', 'linear_inc', 'linear_dec', 'cluster'],
    },
    'hbar_categorical': {
        'y_range': [0, 99],
        'n_points_range': [2, 10],
        'x_distn': ['linear'],
        'shape': ['random', 'random', 'random', 'random', 'linear_inc', 'linear_dec', 'cluster'],
    },
    'line': {
        'x_range': [0, 100],
        'y_range': [0, 100],
        'n_points_range': [5, 20],
        'x_distn': ['linear'],
        'shape': ['linear', 'linear_with_noise', 'quadratic'],
        'n_classes_range': [2, 7],
    },
    'dot_line': {
        'x_range': [0, 100],
        'y_range': [0, 100],
        'n_points_range': [5, 20],
        'x_distn': ['linear'],
        'shape': ['linear', 'linear_with_noise', 'quadratic'],
        'n_classes_range': [2, 7],
    },
    'pie': {
        'n_classes_range': [2, 7],
    },
}

# Paths
COLORS_FILE = Path("/home/ylu21/proj/FigureQA/resources/x11_colors_refined.txt")
OUTPUT_DIR = Path(__file__).parent.parent / "data"


def load_colors(path: Path = COLORS_FILE) -> list[tuple[str, str]]:
    """Load color name -> hex mapping."""
    colors = []
    with open(path) as f:
        for line in f:
            if ',' in line:
                name, hex_code = line.strip().split(',')
                colors.append((name.strip(), hex_code.strip()))
    return colors


# ============ Data Generation (ported from FigureQA) ============

def generate_data_by_shape(x_range: tuple, y_range: tuple, n: int, x_distn: str, shape: str):
    """Generate x, y data points based on shape.

    Ported from FigureQA source_data_generation.py
    """
    # Generate x values based on distribution
    if x_distn == "random":
        x = (x_range[1] - x_range[0]) * np.random.random(n) + x_range[0]
    elif x_distn == "linear":
        x = np.linspace(x_range[0], x_range[1], n)
    elif x_distn == "normal":
        mean = (x_range[1] - x_range[0]) * np.random.random(1) + x_range[0]
        points = (x_range[1] - x_range[0]) * np.random.normal(0, 1/6.0, 3*n) + mean
        final_points = []
        for point in points:
            if x_range[0] <= point <= x_range[1]:
                final_points.append(point)
            if len(final_points) == n:
                break
        x = final_points
    else:
        x = np.linspace(x_range[0], x_range[1], n)

    x = sorted(x)
    max_slope = (y_range[1] - y_range[0]) / float(x_range[1] - x_range[0]) if x_range[1] != x_range[0] else 1

    # Generate y values based on shape
    if shape == "random":
        y = (y_range[1] - y_range[0]) * np.random.random(n) + y_range[0]

    elif shape == "linear":
        slope_direction = 1 if np.random.random() > 0.5 else -1
        offset = y_range[0] if slope_direction >= 0 else y_range[1]
        y = np.clip(slope_direction * max_slope * np.random.random() * np.array(x) + offset,
                    y_range[0], y_range[1])

    elif shape == "linear_with_noise":
        slope_direction = 1 if np.random.random() > 0.5 else -1
        offset = y_range[0] if slope_direction >= 0 else y_range[1]
        y = np.clip(slope_direction * max_slope * np.random.random() * np.array(x) + offset,
                    y_range[0], y_range[1]).tolist()
        # Add noise
        noise_multiplier = 0.05 * (y_range[1] - y_range[0])
        for i in range(len(y)):
            y[i] += noise_multiplier * (2 * np.random.random() - 1)
        y = np.clip(y, y_range[0], y_range[1])

    elif shape == "linear_inc":
        y = np.clip(max_slope * np.random.random() * np.array(x) + y_range[0],
                    y_range[0], y_range[1])

    elif shape == "linear_dec":
        y = np.clip(-max_slope * np.random.random() * np.array(x) + y_range[1],
                    y_range[0], y_range[1])

    elif shape == "cluster":
        mean = (y_range[1] - y_range[0]) * np.random.random() + y_range[0]
        final_points = []
        while len(final_points) < n:
            points = (y_range[1] - y_range[0]) * np.random.normal(0, 1/6.0, n) + mean
            for point in points:
                if y_range[0] <= point <= y_range[1]:
                    final_points.append(point)
                if len(final_points) == n:
                    break
        y = final_points

    elif shape == "quadratic":
        # Use vertex form: y = a(x-h)^2 + k
        h = (x_range[1] - x_range[0]) / 2 * np.random.random() + x_range[0]
        k = (y_range[1] - y_range[0]) / 2 * np.random.random() + y_range[0]
        dist_from_mid = np.abs((y_range[1] - y_range[0]) / 2 + y_range[0])

        # Decide a direction based on k
        if k < (y_range[1] - y_range[0]) / 2 + y_range[0]:
            a = -1 * dist_from_mid
        else:
            a = 1 * dist_from_mid

        a *= np.random.random() * 0.00005
        y = np.clip([a * (xx - h)**2 + k for xx in x], y_range[0], y_range[1])

    else:
        y = (y_range[1] - y_range[0]) * np.random.random(n) + y_range[0]

    # Convert to lists
    if not isinstance(x, list):
        x = list(x) if hasattr(x, '__iter__') else [x]
    if not isinstance(y, list):
        y = y.tolist() if hasattr(y, 'tolist') else list(y)

    return x, y


def _generate_scatter_data_continuous(config: dict, fix_x_range: bool = False):
    """Generate continuous scatter/line data.

    Ported from FigureQA source_data_generation.py
    """
    x_range = tuple(config['x_range'])
    y_range = tuple(config['y_range'])

    s, e = config['n_classes_range']
    n_classes = np.random.randint(s, e + 1)

    s, e = config['n_points_range']
    n_points = np.random.randint(s, e + 1)

    point_sets = []
    for i in range(n_classes):
        x_distn = np.random.choice(config['x_distn'])
        shape = np.random.choice(config['shape'])

        x, y = generate_data_by_shape(x_range, y_range, n_points, x_distn, shape)

        point_sets.append({'class': i, 'x': x, 'y': y})

    return {'type': 'scatter_base', 'data': point_sets, 'n_points': n_points, 'n_classes': n_classes}


def _generate_scatter_data_categorical(config: dict):
    """Generate categorical scatter data (for bar charts).

    Ported from FigureQA source_data_generation.py
    """
    y_range = tuple(config['y_range'])

    s, e = config['n_points_range']
    n_points = np.random.randint(s, e + 1)

    # Pick and randomize labels by index
    all_labels = np.random.permutation(n_points).tolist()

    x_distn = np.random.choice(config['x_distn'])
    shape = np.random.choice(config['shape'])

    x, y = generate_data_by_shape([0, n_points - 1], y_range, n_points, x_distn, shape)

    # Round x to discretize
    x = np.array(np.around(x), dtype=np.int32)

    # Deduplicate
    dedupe_x, dedupe_y = [x[0]], [y[0]]
    last_x = x[0]
    for i in range(1, len(x)):
        if x[i] != last_x:
            last_x = x[i]
            dedupe_x.append(x[i])
            dedupe_y.append(y[i])

    x, y = dedupe_x, dedupe_y
    labels = [all_labels[xx] for xx in x]

    if not isinstance(y, list):
        y = y.tolist() if hasattr(y, 'tolist') else list(y)

    return {'type': 'scatter_categorical_base', 'data': [{'class': 0, 'x': labels, 'y': y}], 'n_points': n_points}


# ============ Figure Generation Functions ============

def generate_line_data() -> dict:
    """Generate line plot data (FigureQA format)."""
    config = DATA_CONFIG['line']
    data = _generate_scatter_data_continuous(config, fix_x_range=True)

    colors = load_colors()
    selected = random.sample(colors, len(data['data']))

    models = []
    for i, point_set in enumerate(data['data']):
        name, color = selected[i]
        models.append({
            'name': name,
            'label': name,
            'color': color,
            'x': point_set['x'],
            'y': point_set['y'],
        })

    return {
        'type': 'line',
        'models': models,
    }


def generate_dot_line_data() -> dict:
    """Generate dot-line plot data."""
    config = DATA_CONFIG['dot_line']
    data = _generate_scatter_data_continuous(config, fix_x_range=True)

    colors = load_colors()
    selected = random.sample(colors, len(data['data']))

    models = []
    for i, point_set in enumerate(data['data']):
        name, color = selected[i]
        models.append({
            'name': name,
            'label': name,
            'color': color,
            'x': point_set['x'],
            'y': point_set['y'],
        })

    return {
        'type': 'dot_line',
        'models': models,
    }


def generate_vbar_data() -> dict:
    """Generate vertical bar chart data (FigureQA format)."""
    config = DATA_CONFIG['vbar_categorical']
    data = _generate_scatter_data_categorical(config)

    colors = load_colors()
    n_bars = len(data['data'][0]['x'])
    selected = random.sample(colors, n_bars)

    # Map label indices to actual color names
    labels = []
    bar_colors = []
    values = data['data'][0]['y']

    for label_idx in data['data'][0]['x']:
        labels.append(selected[label_idx][0])
        bar_colors.append(selected[label_idx][1])

    models = [{
        'name': 'bars',
        'labels': labels,
        'colors': bar_colors,
        'x': labels,
        'y': values,
    }]

    return {
        'type': 'vbar_categorical',
        'models': models,
    }


def generate_hbar_data() -> dict:
    """Generate horizontal bar chart data (FigureQA format)."""
    config = DATA_CONFIG['hbar_categorical']
    data = _generate_scatter_data_categorical(config)

    colors = load_colors()
    n_bars = len(data['data'][0]['x'])
    selected = random.sample(colors, n_bars)

    labels = []
    bar_colors = []
    values = data['data'][0]['y']

    for label_idx in data['data'][0]['x']:
        labels.append(selected[label_idx][0])
        bar_colors.append(selected[label_idx][1])

    models = [{
        'name': 'bars',
        'labels': labels,
        'colors': bar_colors,
        'x': values,  # values on x-axis for horizontal
        'y': labels,
    }]

    return {
        'type': 'hbar_categorical',
        'models': models,
    }


def generate_pie_data() -> dict:
    """Generate pie chart data (FigureQA format)."""
    config = DATA_CONFIG['pie']

    s, e = config['n_classes_range']
    n_classes = np.random.randint(s, e + 1)

    # Generate random widths
    widths = np.array([np.random.random() + 0.05 for _ in range(n_classes)])
    widths_radians = 2 * np.pi * widths / np.sum(widths)

    # Compute start/end angles
    starts = [0.0]
    for i in range(n_classes - 1):
        starts.append(starts[i] + widths_radians[i])
    ends = starts[1:] + [2 * np.pi]

    colors = load_colors()
    selected = random.sample(colors, n_classes)

    models = []
    for i, (name, color) in enumerate(selected):
        models.append({
            'name': name,
            'label': name,
            'color': color,
            'span': float(widths_radians[i]),
            'start': starts[i],
            'end': ends[i],
        })

    return {
        'type': 'pie',
        'models': models,
    }


# ============ QA Generation (ported from FigureQA) ============

NONE_COLOR = "--None--"


def _get_min_max_non(tuples: list[tuple]) -> dict:
    """Get min, max, and non-min/max elements from (label, value) tuples."""
    sorted_tuples = sorted(tuples, key=lambda x: x[1])
    min_tup = sorted_tuples[0]
    max_tup = sorted_tuples[-1]

    q_data = {'min': min_tup[0], 'max': max_tup[0]}

    if min_tup[1] != max_tup[1]:
        not_min, not_max = None, None
        for i in range(1, len(tuples)):
            if not_min and not_max:
                break
            if tuples[i][1] < sorted_tuples[-1][1]:
                not_max = tuples[i]
            if tuples[i][1] > sorted_tuples[0][1]:
                not_min = tuples[i]

        if not_min:
            q_data['not_min'] = not_min[0]
        if not_max:
            q_data['not_max'] = not_max[0]

    return q_data


def _calculate_roughness(x: list, y: list) -> float:
    """Calculate roughness (sum of absolute slope changes)."""
    x = np.array(x)
    y = np.array(y)
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    differences = slopes[1:] - slopes[:-1]
    return float(np.sum(np.abs(differences)))


def _is_strictly_greater_than(series_a: list, series_b: list) -> bool:
    return all(np.array(series_a) > np.array(series_b))


def _is_strictly_less_than(series_a: list, series_b: list) -> bool:
    return all(np.array(series_a) < np.array(series_b))


def generate_qa_pairs(data: dict) -> list[dict]:
    """Generate QA pairs for a figure."""
    fig_type = data['type']
    models = data['models']

    if fig_type in ['vbar_categorical', 'hbar_categorical']:
        return _generate_categorical_qa(models, is_bar=True)
    elif fig_type == 'pie':
        return _generate_categorical_qa(models, is_bar=False)
    elif fig_type in ['line', 'dot_line']:
        return _generate_line_qa(models)

    return []


def _generate_categorical_qa(models: list, is_bar: bool = True) -> list[dict]:
    """Generate QA for bar/pie charts (FigureQA format)."""
    qa_pairs = []

    if is_bar:
        bars = models[0]
        labels = bars['labels']
        values = bars.get('y', bars.get('x'))
        if isinstance(values[0], str):
            values = bars['x']
    else:
        labels = [m['label'] for m in models]
        values = [m['span'] for m in models]

    data = list(zip(labels, values))
    sorted_categories = sorted(data, key=lambda b: b[1])

    min_category = sorted_categories[0]
    max_category = sorted_categories[-1]

    qa_pairs.append({
        'question_string': f'Is {min_category[0]} the minimum?',
        'question_id': 0,
        'color1_name': min_category[0],
        'color2_name': NONE_COLOR,
        'answer': 1
    })
    qa_pairs.append({
        'question_string': f'Is {max_category[0]} the maximum?',
        'question_id': 1,
        'color1_name': max_category[0],
        'color2_name': NONE_COLOR,
        'answer': 1
    })

    if min_category[1] != max_category[1]:
        not_min_category, not_max_category = None, None
        greater, less = None, None

        for i in range(1, len(data)):
            if not_min_category and not_max_category and greater and less:
                break

            if data[i][1] < sorted_categories[-1][1]:
                not_max_category = data[i][0]
            if data[i][1] > sorted_categories[0][1]:
                not_min_category = data[i][0]

            if data[i - 1][1] < data[i][1]:
                less = data[i - 1][0]
                greater = data[i][0]
            elif data[i - 1][1] > data[i][1]:
                less = data[i][0]
                greater = data[i - 1][0]

        if not_min_category:
            qa_pairs.append({
                'question_string': f'Is {not_min_category} the minimum?',
                'question_id': 0,
                'color1_name': not_min_category,
                'color2_name': NONE_COLOR,
                'answer': 0
            })

        if not_max_category:
            qa_pairs.append({
                'question_string': f'Is {not_max_category} the maximum?',
                'question_id': 1,
                'color1_name': not_max_category,
                'color2_name': NONE_COLOR,
                'answer': 0
            })

        if less and greater:
            qa_pairs.extend([
                {'question_string': f'Is {greater} greater than {less}?', 'question_id': 3,
                 'color1_name': greater, 'color2_name': less, 'answer': 1},
                {'question_string': f'Is {less} less than {greater}?', 'question_id': 2,
                 'color1_name': less, 'color2_name': greater, 'answer': 1},
                {'question_string': f'Is {less} greater than {greater}?', 'question_id': 3,
                 'color1_name': less, 'color2_name': greater, 'answer': 0},
                {'question_string': f'Is {greater} less than {less}?', 'question_id': 2,
                 'color1_name': greater, 'color2_name': less, 'answer': 0},
            ])
    else:
        qa_pairs.extend([
            {'question_string': f'Is {min_category[0]} greater than {max_category[0]}?', 'question_id': 3,
             'color1_name': min_category[0], 'color2_name': max_category[0], 'answer': 0},
            {'question_string': f'Is {max_category[0]} less than {min_category[0]}?', 'question_id': 2,
             'color1_name': max_category[0], 'color2_name': min_category[0], 'answer': 0},
        ])

    # Median questions
    if len(sorted_categories) % 2 == 1:
        median_low_index = len(sorted_categories) // 2
        median_high_index = median_low_index
    else:
        median_high_index = len(sorted_categories) // 2
        median_low_index = median_high_index - 1

    median_low = sorted_categories[median_low_index][0]
    median_high = sorted_categories[median_high_index][0]

    other_indices = list(range(median_low_index)) + list(range(median_low_index + 1, len(sorted_categories)))
    not_median_low = sorted_categories[random.choice(other_indices)][0]

    other_indices = list(range(median_high_index)) + list(range(median_high_index + 1, len(sorted_categories)))
    not_median_high = sorted_categories[random.choice(other_indices)][0]

    qa_pairs.extend([
        {'question_string': f'Is {median_high} the high median?', 'question_id': 5,
         'color1_name': median_high, 'color2_name': NONE_COLOR, 'answer': 1},
        {'question_string': f'Is {median_low} the low median?', 'question_id': 4,
         'color1_name': median_low, 'color2_name': NONE_COLOR, 'answer': 1},
        {'question_string': f'Is {not_median_high} the high median?', 'question_id': 5,
         'color1_name': not_median_high, 'color2_name': NONE_COLOR, 'answer': 0},
        {'question_string': f'Is {not_median_low} the low median?', 'question_id': 4,
         'color1_name': not_median_low, 'color2_name': NONE_COLOR, 'answer': 0},
    ])

    return qa_pairs


def _generate_line_qa(models: list) -> list[dict]:
    """Generate QA for line plots (FigureQA format)."""
    qa_pairs = []

    aucs = {}
    roughnesses = {}
    global_mins = {}
    global_maxes = {}

    for model in models:
        label = model['label']
        x, y = model['x'], model['y']
        aucs[label] = auc(x, y)
        roughnesses[label] = _calculate_roughness(x, y)
        global_mins[label] = min(y)
        global_maxes[label] = max(y)

    # AUC questions
    auc_q_data = _get_min_max_non(list(aucs.items()))
    qa_pairs.extend([
        {'question_string': f'Does {auc_q_data["min"]} have the minimum area under the curve?',
         'question_id': 6, 'color1_name': auc_q_data['min'], 'color2_name': NONE_COLOR, 'answer': 1},
        {'question_string': f'Does {auc_q_data["max"]} have the maximum area under the curve?',
         'question_id': 7, 'color1_name': auc_q_data['max'], 'color2_name': NONE_COLOR, 'answer': 1},
    ])
    if 'not_min' in auc_q_data:
        qa_pairs.append({
            'question_string': f'Does {auc_q_data["not_min"]} have the minimum area under the curve?',
            'question_id': 6, 'color1_name': auc_q_data['not_min'], 'color2_name': NONE_COLOR, 'answer': 0
        })
    if 'not_max' in auc_q_data:
        qa_pairs.append({
            'question_string': f'Does {auc_q_data["not_max"]} have the maximum area under the curve?',
            'question_id': 7, 'color1_name': auc_q_data['not_max'], 'color2_name': NONE_COLOR, 'answer': 0
        })

    # Roughness questions
    roughness_q_data = _get_min_max_non(list(roughnesses.items()))
    qa_pairs.extend([
        {'question_string': f'Is {roughness_q_data["min"]} the smoothest?',
         'question_id': 8, 'color1_name': roughness_q_data['min'], 'color2_name': NONE_COLOR, 'answer': 1},
        {'question_string': f'Is {roughness_q_data["max"]} the roughest?',
         'question_id': 9, 'color1_name': roughness_q_data['max'], 'color2_name': NONE_COLOR, 'answer': 1},
    ])
    if 'not_min' in roughness_q_data:
        qa_pairs.append({
            'question_string': f'Is {roughness_q_data["not_min"]} the smoothest?',
            'question_id': 8, 'color1_name': roughness_q_data['not_min'], 'color2_name': NONE_COLOR, 'answer': 0
        })
    if 'not_max' in roughness_q_data:
        qa_pairs.append({
            'question_string': f'Is {roughness_q_data["not_max"]} the roughest?',
            'question_id': 9, 'color1_name': roughness_q_data['not_max'], 'color2_name': NONE_COLOR, 'answer': 0
        })

    # Lowest/highest value questions
    global_min_data = _get_min_max_non(list(global_mins.items()))
    qa_pairs.append({
        'question_string': f'Does {global_min_data["min"]} have the lowest value?',
        'question_id': 10, 'color1_name': global_min_data['min'], 'color2_name': NONE_COLOR, 'answer': 1
    })
    if 'not_min' in global_min_data:
        qa_pairs.append({
            'question_string': f'Does {global_min_data["not_min"]} have the lowest value?',
            'question_id': 10, 'color1_name': global_min_data['not_min'], 'color2_name': NONE_COLOR, 'answer': 0
        })

    global_max_data = _get_min_max_non(list(global_maxes.items()))
    qa_pairs.append({
        'question_string': f'Does {global_max_data["max"]} have the highest value?',
        'question_id': 11, 'color1_name': global_max_data['max'], 'color2_name': NONE_COLOR, 'answer': 1
    })
    if 'not_max' in global_max_data:
        qa_pairs.append({
            'question_string': f'Does {global_max_data["not_max"]} have the highest value?',
            'question_id': 11, 'color1_name': global_max_data['not_max'], 'color2_name': NONE_COLOR, 'answer': 0
        })

    # Strict comparison and intersection
    strictness_map = {
        'AltB': None, 'AgtB': None, 'AintB': None,
        'not_AltB': None, 'not_AgtB': None, 'not_AintB': None
    }

    all_labels = [m['label'] for m in models]
    model_map = {m['label']: m for m in models}

    all_perms = list(itertools.combinations(all_labels, 2))
    all_perms = [(a, b) for a, b in all_perms] + [(b, a) for a, b in all_perms]
    all_perms = [(a, b) for a, b in all_perms if a != b]
    random.shuffle(all_perms)

    for a, b in all_perms:
        if all(v is not None for v in strictness_map.values()):
            break

        a_lt_b = _is_strictly_less_than(model_map[a]['y'], model_map[b]['y'])
        a_gt_b = _is_strictly_greater_than(model_map[a]['y'], model_map[b]['y'])

        if a_lt_b and not strictness_map['AltB'] and not strictness_map['not_AintB']:
            strictness_map['AltB'] = (a, b)
            strictness_map['not_AintB'] = (a, b)

        if a_gt_b and not strictness_map['AgtB']:
            strictness_map['AgtB'] = (a, b)

        if not a_lt_b and not a_gt_b and not strictness_map['AintB']:
            strictness_map['AintB'] = (a, b)
            strictness_map['not_AltB'] = (a, b)
            strictness_map['not_AgtB'] = (a, b)

    if strictness_map['AltB']:
        a, b = strictness_map['AltB']
        qa_pairs.append({
            'question_string': f'Is {a} less than {b}?', 'question_id': 12,
            'color1_name': a, 'color2_name': b, 'answer': 1
        })

    if strictness_map['AgtB']:
        a, b = strictness_map['AgtB']
        qa_pairs.append({
            'question_string': f'Is {a} greater than {b}?', 'question_id': 13,
            'color1_name': a, 'color2_name': b, 'answer': 1
        })

    if strictness_map['AintB']:
        a, b = strictness_map['AintB']
        qa_pairs.append({
            'question_string': f'Does {a} intersect {b}?', 'question_id': 14,
            'color1_name': a, 'color2_name': b, 'answer': 1
        })

    if strictness_map['not_AltB']:
        a, b = strictness_map['not_AltB']
        qa_pairs.append({
            'question_string': f'Is {a} less than {b}?', 'question_id': 12,
            'color1_name': a, 'color2_name': b, 'answer': 0
        })

    if strictness_map['not_AgtB']:
        a, b = strictness_map['not_AgtB']
        qa_pairs.append({
            'question_string': f'Is {a} greater than {b}?', 'question_id': 13,
            'color1_name': a, 'color2_name': b, 'answer': 0
        })

    if strictness_map['not_AintB']:
        a, b = strictness_map['not_AintB']
        qa_pairs.append({
            'question_string': f'Does {a} intersect {b}?', 'question_id': 14,
            'color1_name': a, 'color2_name': b, 'answer': 0
        })

    return qa_pairs


# ============ Question Balancing (ported from FigureQA) ============

NUM_DISTINCT_QS = 15


def balance_questions_by_qid(all_data: list, all_qa: list) -> list:
    """Balance questions by question_id to achieve ~50/50 Yes/No per QID.

    Ported from FigureQA questions/utils.py balance_questions_by_qid()

    Args:
        all_data: List of figure annotations (with image_index)
        all_qa: List of all QA pairs (with image_index)

    Returns:
        Balanced list of QA pairs
    """
    # Group QA pairs by image_index
    qa_by_image = {}
    for qa in all_qa:
        img_idx = qa['image_index']
        if img_idx not in qa_by_image:
            qa_by_image[img_idx] = []
        qa_by_image[img_idx].append(qa)

    # Count Yes/No per question_id across all figures
    qid_counts = {qid: [0, 0] for qid in range(NUM_DISTINCT_QS)}
    for qa in all_qa:
        qid_counts[qa['question_id']][qa['answer']] += 1

    # Balance by discarding excess Yes/No per QID
    total_qa_pairs = 0
    total_qa_pairs_lost = 0
    balanced_qa = []

    for img_idx in sorted(qa_by_image.keys()):
        qa_pairs = qa_by_image[img_idx]
        new_qa_pairs = []

        for i, qa in enumerate(qa_pairs):
            # Can't discard everything - keep at least one QA per figure
            if i == len(qa_pairs) - 1 and len(new_qa_pairs) == 0:
                new_qa_pairs.append(qa)
                continue

            # Attempt to balance by qid
            diff = qid_counts[qa['question_id']][1] - qid_counts[qa['question_id']][0]

            if diff > 0 and qa['answer'] == 1:
                # Too many Yes answers, discard this one
                qid_counts[qa['question_id']][1] -= 1
                continue
            elif diff < 0 and qa['answer'] == 0:
                # Too many No answers, discard this one
                qid_counts[qa['question_id']][0] -= 1
                continue

            # Keep this question
            new_qa_pairs.append(qa)

        total_qa_pairs_lost += len(qa_pairs) - len(new_qa_pairs)
        total_qa_pairs += len(new_qa_pairs)
        balanced_qa.extend(new_qa_pairs)

    # Print balancing stats
    print(f"Question balancing: kept {total_qa_pairs}, discarded {total_qa_pairs_lost} "
          f"({100*total_qa_pairs_lost/(total_qa_pairs+total_qa_pairs_lost):.1f}%)")

    return balanced_qa


# ============ Main ============

def generate_dataset(
    output_file: Path,
    seed: int = 42,
    vbar: int = 200,
    hbar: int = 200,
    pie: int = 200,
    line: int = 200,
    dot_line: int = 200,
):
    """Generate full dataset."""
    random.seed(seed)
    np.random.seed(seed)

    generators = [
        ('vbar_categorical', generate_vbar_data, vbar),
        ('hbar_categorical', generate_hbar_data, hbar),
        ('pie', generate_pie_data, pie),
        ('line', generate_line_data, line),
        ('dot_line', generate_dot_line_data, dot_line),
    ]

    all_data = []
    all_qa = []

    for fig_type, gen_func, count in generators:
        print(f"Generating {count} {fig_type}...")
        for i in range(count):
            data = gen_func()
            data['image_index'] = len(all_data)

            qa_pairs = generate_qa_pairs(data)
            for qa in qa_pairs:
                qa['image_index'] = data['image_index']

            all_data.append(data)
            all_qa.extend(qa_pairs)

    # Balance questions by question_id (like FigureQA)
    all_qa = balance_questions_by_qid(all_data, all_qa)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotations_file = output_file.parent / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    qa_file = output_file.parent / "qa_pairs.json"
    with open(qa_file, 'w') as f:
        json.dump({'qa_pairs': all_qa}, f, indent=2)

    print(f"Generated {len(all_data)} figures")
    print(f"Generated {len(all_qa)} QA pairs (after balancing)")
    print(f"Saved to: {annotations_file}")
    print(f"Saved to: {qa_file}")

    counts = {}
    for d in all_data:
        t = d['type']
        counts[t] = counts.get(t, 0) + 1
    print(f"Counts: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Generate iPlotBench source data (FigureQA format)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "source.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vbar", type=int, default=200)
    parser.add_argument("--hbar", type=int, default=200)
    parser.add_argument("--pie", type=int, default=200)
    parser.add_argument("--line", type=int, default=200)
    parser.add_argument("--dot-line", type=int, default=200, dest="dot_line")
    parser.add_argument("--total", type=int, default=None,
                        help="Total figures (overrides individual counts)")

    args = parser.parse_args()

    requested_total = args.total
    if args.total:
        per_type = args.total // 5
        actual_total = per_type * 5
        args.vbar = args.hbar = args.pie = args.line = args.dot_line = per_type

        # Show rounding message if needed
        if actual_total != requested_total:
            print(f"Note: Requested {requested_total} figures, generating {actual_total}")
            print(f"      (rounded to {per_type} per type x 5 types)")
            print("")

    generate_dataset(
        output_file=args.output,
        seed=args.seed,
        vbar=args.vbar,
        hbar=args.hbar,
        pie=args.pie,
        line=args.line,
        dot_line=args.dot_line,
    )


if __name__ == "__main__":
    main()
