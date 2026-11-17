"""Create 200 fictional evaluations; never represents an actual survey."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 4312026
N = 200
ATTRIBUTES = ['Teaching_Clarity', 'Subject_Knowledge', 'Student_Interaction',
              'Feedback_Timeliness', 'Learning_Resources', 'Overall_Satisfaction']
VALUES = ['Low', 'Medium', 'High']


def generate(n=N, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        # First five factors sampled separately, without duplicated ratings.
        ratings = [rng.choices(VALUES, weights=w, k=1)[0] for w in
                   [(0.20, 0.30, 0.50), (0.12, 0.28, 0.60),
                    (0.25, 0.30, 0.45), (0.25, 0.40, 0.35),
                    (0.20, 0.40, 0.40)]]
        level = dict(zip(VALUES, [0, 1, 2]))
        # Simulated satisfaction propensity; clarity/interaction get larger weights.
        quality = sum(w * level[r] for w, r in zip(
            [0.35, 0.15, 0.30, 0.10, 0.10], ratings)) / 2
        if quality >= 0.70:
            probabilities = [0.05, 0.15, 0.80]
        elif quality >= 0.45:
            probabilities = [0.15, 0.55, 0.30]
        else:
            probabilities = [0.70, 0.25, 0.05]
        ratings.append(rng.choices(VALUES, weights=probabilities, k=1)[0])
        rows.append(dict(zip(ATTRIBUTES, ratings)))
    return rows


def main():
    path = ROOT / 'data/teaching_evaluation_synthetic.csv'
    path.parent.mkdir(exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=ATTRIBUTES)
        writer.writeheader()
        writer.writerows(generate())
    print(f'Generated {N} synthetic rows with seed {SEED}: {path}')


if __name__ == '__main__':
    main()
