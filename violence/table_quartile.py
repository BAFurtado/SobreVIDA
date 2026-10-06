"""Build the deprivation table — policy arms by AREAP wage quartile.

Every number is read from `output/scenarios_panel_<metro><suffix>.csv`, so the table
cannot drift from the runs that produced it.

WHY THE PANEL AND NOT `scenarios_wage_quartile_<metro>.csv`. The published quartile CSV
carries counts pooled over replicates, so a cell has a level but no dispersion, and a
quartile delta cannot be told from noise. The panel keeps `run`, so the delta is formed
WITHIN replicate — the same common-random-numbers pairing Table 1 uses — and the spread
across replicates is the standard error. It reads a ~100 MB file per city.

WHY QUARTILES AND NOT MUNICIPALITIES. `measure_municipality` flags cells the run cannot
support (`has_power`, scenarios.py). At 150 replicates only 1 of 23 municipalities clears
it in Porto Alegre and 2 of 23 in Belo Horizonte: the capital carries almost the whole
population, so a per-municipality table would be one real row and twenty-two noise rows.
Pooling AREAPs by wage instead puts ~42 areas in every cell, which resolves. The cut is
deprivation, not geography: the model's network is non-spatial.

    python3 violence/table_quartile.py                  # markdown, both cities
    python3 violence/table_quartile.py --latex          # LaTeX booktabs body
    python3 violence/table_quartile.py -m "BELO HORIZONTE"
    python3 violence/table_quartile.py --suffix _final  # read a specific run

--suffix must match the `-o` given to scenarios.py. It matters: the table renders just as
happily from an older run as from the current one, so leaving it at the default after a
new run rebuilds the previous table without any sign that it did.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scenarios import wage_quartiles  # noqa: E402

OUTPUT = pathlib.Path(__file__).resolve().parent.parent / 'output'

QUARTILES = ['Q1', 'Q2', 'Q3', 'Q4']

# The arms the table reports. `no_mpu_no_patrol` reproduces `no_mpu` exactly once no
# order exists, and `inert_mpu`/`no_arrest` are city-wide contrasts that do not resolve
# at this cell size; they stay in the run and out of the table.
ARMS = [
    ('no_mpu', 'No protective order'),
    ('no_patrol', 'No patrol'),
    ('full_patrol', 'Full patrol coverage'),
    ('more_reporting', 'Reporting at the FBSP ceiling'),
]

DEFAULT_SUFFIX = '_final'


def paired(metro, suffix=DEFAULT_SUFFIX):
    """Prevalence per quartile, and each arm's paired delta with its standard error."""
    path = OUTPUT / f'scenarios_panel_{metro}{suffix}.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found.\nThe suffix is part of the filename scenarios.py wrote '
            f'(-o). Pass --suffix to point at the run you mean, e.g. --suffix _final.')
    panel = pd.read_csv(path)
    panel['AREAP'] = panel.AREAP.astype(str)
    groups, _ = wage_quartiles(panel.AREAP.unique())
    panel['q'] = panel.AREAP.map(groups)
    panel = panel.dropna(subset=['q'])

    # Prevalence is a ratio of sums; sum the parts inside the replicate before dividing,
    # so a large area weighs more than a small one, as in scenarios.measure_wage_groups.
    cell = panel.groupby(['arm', 'q', 'run'], observed=True)[['Attacked', 'Females']].sum()
    cell['prev'] = cell.Attacked / cell.Females
    prev = cell.prev.unstack('arm')

    rows = []
    for q in QUARTILES:
        sub = prev.xs(q, level='q')
        row = {'quartile': q, 'n_rep': len(sub), 'baseline': sub.baseline.mean() * 100}
        for arm, _ in ARMS:
            if arm not in sub:
                continue
            delta = (sub[arm] - sub.baseline) * 100
            se = delta.std(ddof=1) / len(delta) ** 0.5
            row[arm] = (delta.mean(), se, delta.mean() / se if se else float('nan'))
        rows.append(row)
    return path, rows


def gradient(metro, arm='more_reporting', suffix=DEFAULT_SUFFIX):
    """The Q1-minus-Q4 contrast, paired within replicate.

    Monotonicity across four cells is suggestive but is not a test; this is the test.
    """
    path = OUTPUT / f'scenarios_panel_{metro}{suffix}.csv'
    panel = pd.read_csv(path)
    panel['AREAP'] = panel.AREAP.astype(str)
    groups, _ = wage_quartiles(panel.AREAP.unique())
    panel['q'] = panel.AREAP.map(groups)
    panel = panel.dropna(subset=['q'])
    cell = panel.groupby(['arm', 'q', 'run'], observed=True)[['Attacked', 'Females']].sum()
    cell['prev'] = cell.Attacked / cell.Females
    prev = cell.prev.unstack('arm')
    d = {q: (prev.xs(q, level='q')[arm] - prev.xs(q, level='q').baseline) * 100
         for q in ('Q1', 'Q4')}
    diff = d['Q1'] - d['Q4']
    se = diff.std(ddof=1) / len(diff) ** 0.5
    return diff.mean(), se, diff.mean() / se


def _fmt(entry):
    delta, se, z = entry
    return f'{delta:+.2f} ({se:.2f})  z*={z:+.1f}'


def markdown(metro, suffix=DEFAULT_SUFFIX):
    path, rows = paired(metro, suffix)
    head = ['Wage quartile', 'Prevalence (%)'] + [label for _, label in ARMS]
    lines = [f'### {metro} — {rows[0]["n_rep"]} replicates',
             '', '| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    for row in rows:
        cells = [row['quartile'], f'{row["baseline"]:.3f}']
        cells += [_fmt(row[arm]) if arm in row else '---' for arm, _ in ARMS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    g, gse, gz = gradient(metro, suffix=suffix)
    lines += ['', f'Q1 minus Q4 on the reporting arm: {g:+.2f} pp (SE {gse:.2f}), '
                  f'z* = {gz:+.2f}.',
              f'Source: `output/{path.name}`. Deltas are percentage points of prevalence, '
              f'paired within replicate under common random numbers.']
    return '\n'.join(lines)


def latex(metro, suffix=DEFAULT_SUFFIX):
    path, rows = paired(metro, suffix)
    body = []
    for row in rows:
        cells = [row['quartile'], f'{row["baseline"]:.3f}']
        cells += [_fmt(row[arm]).replace('*', '$^*$') if arm in row else '---'
                  for arm, _ in ARMS]
        body.append(' & '.join(cells))
    head = ' & '.join(['Quartile', '\\textbf{Prevalence (\\%)}']
                      + [f'\\textbf{{{label}}}' for _, label in ARMS])
    g, gse, gz = gradient(metro, suffix=suffix)
    return (f'% {metro} — {rows[0]["n_rep"]} replicates, from output/{path.name}\n'
            f'% Q1-Q4 on more_reporting: {g:+.2f} pp (SE {gse:.2f}), z* = {gz:+.2f}\n'
            f'\\begin{{tabular}}{{l c cccc}}\n\\toprule\n{head} \\\\\n\\midrule\n'
            + ' \\\\\n'.join(body) + f' \\\\\n\\bottomrule\n\\end{{tabular}}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--metro', action='append',
                        help='metropolis; repeatable, defaults to both')
    parser.add_argument('--latex', action='store_true', help='emit a LaTeX tabular')
    parser.add_argument('--suffix', default=DEFAULT_SUFFIX,
                        help=f'suffix of the scenarios run to read, i.e. the -o given to '
                             f'scenarios.py (default {DEFAULT_SUFFIX!r}). Point this at the '
                             f'current run or the table silently rebuilds an older one.')
    args = parser.parse_args()

    for metro in args.metro or ['PORTO ALEGRE', 'BELO HORIZONTE']:
        print(latex(metro, args.suffix) if args.latex else markdown(metro, args.suffix))
        print()


if __name__ == '__main__':
    main()
