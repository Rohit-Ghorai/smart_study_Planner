# Experimental Results

This report records one reproducible smoke experiment, not a statistically
conclusive algorithm ranking.

## Configuration

- Subjects: Math (difficulty 4, confidence 2), Physics (4, 3), English (2, 4)
- Episodes: 200 per algorithm
- Alpha: 0.1; gamma: 0.9; initial epsilon: 1.0; decay: 0.995; minimum: 0.05
- Seed: 42
- State count: 7,290; action count: 15

## Results

| Metric | Q-learning | SARSA |
|---|---:|---:|
| Mean training reward | 23.0340 | 21.2860 |
| Population standard deviation | 7.3831 | 6.2557 |
| Best episode | 33.7000 | 32.7000 |
| Worst episode | -4.5000 | -4.5000 |
| Greedy evaluation reward | 33.7000 | 23.0000 |

Generated schedules were `Math_Mock, Physics, Math_Deep` for Q-learning and
`Math_Mock, Physics_Deep` for SARSA. These single-run results are evidence that
the pipeline executes, not proof that one algorithm is universally superior.
