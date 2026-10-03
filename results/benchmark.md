# K?t qu? benchmark offline

## Standard Benchmark

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 901 |                     11850 |                  0.000 |              0.000 |                       0 |             0 |
| Advanced |                 915 |                     19296 |                  1.000 |              1.000 |                     324 |             0 |

## Long-Context Stress Benchmark

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 174 |                     21444 |                  0.000 |              0.000 |                       0 |             0 |
| Advanced |                 205 |                     13115 |                  1.000 |              1.000 |                     265 |             2 |
