"""Mine identical data, check equivalence and save real repeated timings."""
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from generate_data import ATTRIBUTES, N, ROOT, SEED, VALUES, generate

MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.0  # Strict greater-than for retained rules.
MAX_LENGTH = 3
TRIALS = 20
ORDER_SEED = 431
EPSILON = 1e-12


def eclat(transactions, min_support=MIN_SUPPORT, max_length=MAX_LENGTH):
    """Vertical TID sets, lexicographic suffixes and depth-first intersection."""
    n = len(transactions)
    if not n:
        raise ValueError('Transactions must not be empty')
    minimum_count = math.ceil(min_support * n - EPSILON)
    vertical = {}
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            vertical.setdefault(item, set()).add(tid)
    suffix = [(item, tids) for item, tids in sorted(vertical.items())
              if len(tids) >= minimum_count]
    found = {}

    def visit(prefix, candidates):
        for index, (item, tids) in enumerate(candidates):
            extended = prefix + (item,)
            found[frozenset(extended)] = len(tids) / n
            if len(extended) >= max_length:
                continue
            next_candidates = []
            for other, other_tids in candidates[index + 1:]:
                common = tids & other_tids
                if len(common) >= minimum_count:
                    next_candidates.append((other, common))
            visit(extended, next_candidates)

    visit((), suffix)
    return found


def as_mapping(frame):
    return dict(zip(frame['itemsets'], frame['support']))


def make_rules(itemsets, min_confidence=MIN_CONFIDENCE, min_lift=MIN_LIFT):
    """One shared generator: every nonempty proper split of frequent itemsets."""
    rules = []
    for union, support in sorted(itemsets.items(), key=lambda pair: sorted(pair[0])):
        for size in range(1, len(union)):
            for antecedent in itertools.combinations(sorted(union), size):
                left = frozenset(antecedent)
                right = union - left
                confidence = support / itemsets[left]
                lift = confidence / itemsets[right]
                if confidence + EPSILON >= min_confidence and lift > min_lift:
                    rules.append(dict(antecedent=' & '.join(sorted(left)),
                                      consequent=' & '.join(sorted(right)),
                                      support=support, confidence=confidence, lift=lift))
    return sorted(rules, key=lambda r: (-r['lift'], -r['confidence'],
                                       -r['support'], r['antecedent'], r['consequent']))


def write_csv(name, rows, fields):
    with (ROOT / 'results' / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def brute_force(transactions, min_support, max_length):
    """Independent tiny-data reference, also checks the full 18-item sample."""
    all_items = sorted(set().union(*transactions))
    result = {}
    for size in range(1, max_length + 1):
        for combination in itertools.combinations(all_items, size):
            key = frozenset(combination)
            support = sum(key.issubset(t) for t in transactions) / len(transactions)
            if support + EPSILON >= min_support:
                result[key] = support
    return result


def verify(transactions, outputs, rules, frame):
    checks = []
    assert len(frame) == N and list(frame.columns) == ATTRIBUTES
    assert frame.isin(VALUES).all().all() and not frame.isna().any().any()
    assert all(len(t) == len(ATTRIBUTES) for t in transactions)
    assert frame.to_dict('records') == generate()
    checks.append('200 rows, six valid categories per row, fixed-seed regeneration agrees')
    reference = brute_force(transactions, MIN_SUPPORT, MAX_LENGTH)
    for name, found in outputs.items():
        assert found.keys() == reference.keys(), name
        assert all(abs(found[k] - reference[k]) <= EPSILON for k in found), name
    checks.append('All three complete itemset maps agree with exhaustive reference to 1e-12')
    assert all(len(k) <= MAX_LENGTH for k in reference)
    for rule in rules:
        left = frozenset(rule['antecedent'].split(' & '))
        right = frozenset(rule['consequent'].split(' & '))
        joint = sum((left | right).issubset(t) for t in transactions)
        left_count = sum(left.issubset(t) for t in transactions)
        right_count = sum(right.issubset(t) for t in transactions)
        assert abs(rule['support'] - joint / N) <= EPSILON
        assert abs(rule['confidence'] - joint / left_count) <= EPSILON
        assert abs(rule['lift'] - joint * N / (left_count * right_count)) <= EPSILON
        assert rule['support'] + EPSILON >= MIN_SUPPORT
        assert rule['confidence'] + EPSILON >= MIN_CONFIDENCE and rule['lift'] > MIN_LIFT
    checks.append('Every retained rule independently recounted and metric formulas verified')
    tiny = [frozenset('ab'), frozenset('ab'), frozenset('a'), frozenset('c')]
    tiny_sets = eclat(tiny, 0.5, 3)
    assert tiny_sets == {frozenset('a'): .75, frozenset('b'): .5, frozenset('ab'): .5}
    tiny_rules = make_rules(tiny_sets)
    ab = next(r for r in tiny_rules if r['antecedent'] == 'a')
    assert math.isclose(ab['confidence'], 2/3) and math.isclose(ab['lift'], 4/3)
    assert make_rules({frozenset('a'): 1., frozenset('b'): .5,
                       frozenset('ab'): .5}) == []  # Lift=1 must be excluded.
    checks.append('Four-transaction manual example and strict lift boundary verified')
    rng = random.Random(99)
    for _ in range(12):
        tiny = [frozenset(x for x in 'abcde' if rng.random() < .5) for _ in range(9)]
        for length in [1, 2, 3]:
            assert eclat(tiny, .25, length) == brute_force(tiny, .25, length)
    checks.append('ECLAT agrees with exhaustive counts on 12 additional seeded small cases')
    return checks


def make_figures(comparison, selected):
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout='constrained')
    ax.bar([r['algorithm'] for r in comparison], [r['median_ms'] for r in comparison],
           color='#646464')
    for i, row in enumerate(comparison):
        ax.text(i, row['median_ms'], f"{row['median_ms']:.3f}", ha='center', va='bottom')
    ax.set(ylabel='Median mining time (ms)', title=f'Mining time on {N} synthetic transactions ({TRIALS} trials)')
    ax.margins(y=.2)
    fig.savefig(ROOT / 'figures/algorithm_runtime.png', dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout='constrained')
    ax.barh([r['rule_id'] for r in selected][::-1],
            [100*r['confidence'] for r in selected][::-1], color='#646464')
    for i, r in enumerate(selected[::-1]):
        ax.text(100*r['confidence'] + 1, i, f"{100*r['confidence']:.1f}%", va='center')
    ax.set(xlabel='Confidence (%)', xlim=(0, 112),
           title='Selected synthetic rules (R1 to R5 match Table 5)')
    ax.axvline(60, color='black', linestyle='--', linewidth=.8)
    fig.savefig(ROOT / 'figures/rule_confidence.png', dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 1.45), layout='constrained')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    labels = ['Generate 200\nsynthetic rows', 'Encode six\nitems per row', 'Mine with three\nalgorithms',
              'Check equality\nand form rules', 'Compare times\nand interpret']
    for i, label in enumerate(labels):
        x = .1 + i * .2
        ax.text(x, .52, label, ha='center', va='center', fontsize=9,
                bbox={'boxstyle': 'square,pad=.5', 'facecolor': 'white', 'edgecolor': 'black'})
        if i < 4:
            ax.annotate('', xy=(x+.13, .52), xytext=(x+.075, .52),
                        arrowprops={'arrowstyle': '->', 'color': 'black'})
    ax.axis('off')
    fig.savefig(ROOT / 'figures/methodology_workflow.png', dpi=250)
    plt.close(fig)


def environment():
    processor = platform.processor()
    try:
        p = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Processor | Select-Object -First 1).Name'],
                           capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and p.stdout.strip():
            processor = p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return dict(python=sys.version.split()[0], operating_system=platform.platform(),
                processor=processor, logical_cpus=os.cpu_count(),
                libraries={p: importlib.metadata.version(p) for p in
                           ['numpy', 'pandas', 'mlxtend', 'matplotlib', 'scikit-learn', 'scipy']})


def main():
    for directory in ['results', 'figures']:
        (ROOT / directory).mkdir(exist_ok=True)
    path = ROOT / 'data/teaching_evaluation_synthetic.csv'
    frame = pd.read_csv(path, dtype=str)
    transactions = [frozenset(f'{a}={row[a]}' for a in ATTRIBUTES)
                    for row in frame.to_dict('records')]
    encoder = TransactionEncoder()
    one_hot = pd.DataFrame(encoder.fit_transform(transactions), columns=encoder.columns_, dtype=bool)
    miners = {
        'Apriori': lambda: apriori(one_hot, min_support=MIN_SUPPORT,
                                   max_len=MAX_LENGTH, use_colnames=True),
        'FP-Growth': lambda: fpgrowth(one_hot, min_support=MIN_SUPPORT,
                                      max_len=MAX_LENGTH, use_colnames=True),
        'ECLAT': lambda: eclat(transactions),
    }
    # One unmeasured warm-up per implementation. Timer includes mining/output creation,
    # and ECLAT vertical-set construction, but excludes shared CSV reading/encoding,
    # result canonicalization, checks, rule generation and plotting.
    for miner in miners.values():
        miner()
    timings = {name: [] for name in miners}
    raw = []
    rng = random.Random(ORDER_SEED)
    for trial in range(1, TRIALS + 1):
        order = list(miners)
        rng.shuffle(order)
        for position, name in enumerate(order, 1):
            start = time.perf_counter()
            output = miners[name]()
            elapsed = (time.perf_counter() - start) * 1000
            timings[name].append(elapsed)
            raw.append(dict(trial=trial, position=position, algorithm=name, runtime_ms=elapsed))
    outputs = {name: miners[name]() for name in miners}
    outputs = {k: v if isinstance(v, dict) else as_mapping(v) for k, v in outputs.items()}
    common = outputs['Apriori']
    rules = make_rules(common)
    checks = verify(transactions, outputs, rules, frame)
    comparison = [dict(algorithm=name, frequent_itemsets=len(found), rules=len(rules),
                       median_ms=statistics.median(timings[name]),
                       q1_ms=float(np.percentile(timings[name], 25)),
                       q3_ms=float(np.percentile(timings[name], 75)),
                       min_ms=min(timings[name]), max_ms=max(timings[name]), trials=TRIALS)
                  for name, found in outputs.items()]
    # Deterministic educational subset: single-item consequents, ranked by lift,
    # confidence, support, then lexical order. All rules retained separately.
    selected = [dict(rule_id=f'R{i}', **r) for i, r in enumerate(
        [r for r in rules if len(r['consequent'].split(' & ')) == 1][:5], 1)]
    assert len(selected) == 5, 'Not enough selected rules at unchanged thresholds'
    summary = [dict(attribute=a, **{v: int((frame[a] == v).sum()) for v in VALUES}) for a in ATTRIBUTES]
    for name, found in outputs.items():
        item_rows = [dict(items=' & '.join(sorted(k)), length=len(k), support=v,
                          count=round(v * N)) for k, v in sorted(found.items(), key=lambda p:(len(p[0]), sorted(p[0])))]
        write_csv('itemsets_' + name.lower().replace('-', '_') + '.csv', item_rows,
                  ['items', 'length', 'support', 'count'])
    write_csv('algorithm_comparison.csv', comparison, list(comparison[0]))
    write_csv('runtime_trials.csv', raw, ['trial', 'position', 'algorithm', 'runtime_ms'])
    write_csv('association_rules.csv', rules, ['antecedent', 'consequent', 'support', 'confidence', 'lift'])
    write_csv('selected_rules.csv', selected, ['rule_id', 'antecedent', 'consequent', 'support', 'confidence', 'lift'])
    write_csv('dataset_summary.csv', summary, ['attribute', 'Low', 'Medium', 'High'])
    make_figures(comparison, selected)
    experiment = dict(dataset_rows=N, seed=SEED, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, lift_strictly_greater_than=MIN_LIFT,
                      max_length=MAX_LENGTH, trials=TRIALS, order_seed=ORDER_SEED,
                      unique_items=len(one_hot.columns), itemsets_by_length=dict(Counter(len(k) for k in common)),
                      comparison=comparison, selected_rules=selected, dataset_summary=summary,
                      environment=environment(), checks=checks, synthetic=True)
    (ROOT / 'results/experiment.json').write_text(json.dumps(experiment, indent=2), encoding='utf-8')
    print(json.dumps(experiment, indent=2))


if __name__ == '__main__':
    main()
