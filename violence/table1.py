"""Build Table 1 of the paper — policy arms against the four published outcomes.

Every number is read from `output/scenarios_<metro><suffix>.csv`, so the table cannot drift
from the runs that produced it. Deltas are PAIRED by common random numbers: replicate k
of every arm shares the seed of replicate k of the baseline, which is what makes a 1%
effect resolvable at 40 replicates.

    python3 violence/table1.py                       # markdown, both cities
    python3 violence/table1.py --latex               # LaTeX booktabs body
    python3 violence/table1.py -m "BELO HORIZONTE"
    python3 violence/table1.py --suffix _final       # read a specific run

--suffix must match the `-o` given to scenarios.py. It matters: the table renders just as
happily from an older run as from the current one, so leaving it at the default after a
new run rebuilds the previous table without any sign that it did.

`no_mpu_no_patrol` is dropped by default: with `P_MPU_REQUESTED = 0` no order exists, so
patrol coverage has nothing to act on and the arm reproduces `no_mpu` exactly. It is kept
in the runs as a consistency check, not as a scenario. Pass --all to show it.
"""

import argparse
import csv
import pathlib

OUTPUT = pathlib.Path(__file__).resolve().parent.parent / 'output'

# (column, label, decimals) — the four outcomes the paper publishes.
OUTCOMES = [
    ('prevalence', 'Prevalence', 3),
    ('assaults_per_woman_year', 'Assaults per woman-year', 4),
    ('non_compliance_rate', 'Non-compliance per order', 3),
    ('femicide_rate_expected', 'Expected femicide (per 100k)', 2),
]

ARM_LABEL = {
    'baseline': 'Baseline',
    'no_mpu': 'No protective order',
    'inert_mpu': 'Order granted but inert',
    'no_arrest': 'No arrest on breach',
    'no_patrol': 'No patrol',
    'full_patrol': 'Full patrol coverage',
    'no_mpu_no_patrol': 'No order, no patrol',
    'more_reporting': 'Reporting at the FBSP ceiling',
}

REDUNDANT = 'no_mpu_no_patrol'

# Which scenarios run to read, i.e. the `-o` given to scenarios.py. A default rather than a
# hard-coded name because the table is regenerated at every replicate count, and reading
# the wrong run is silent: the table renders perfectly, from stale numbers.
DEFAULT_SUFFIX = '_09'


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read(metro, suffix=DEFAULT_SUFFIX):
    path = OUTPUT / f'scenarios_{metro}{suffix}.csv'
    if not path.exists():
        raise SystemExit(
            f'{path} not found.\nThe suffix is part of the filename scenarios.py wrote '
            f'(-o). Pass --suffix to point at the run you mean, e.g. --suffix _final.')
    with path.open() as handle:
        return path, list(csv.DictReader(handle))


def cell(row, base, key, decimals):
    """Baseline prints level and standard error; every other arm prints level, paired
    percentage change and z*. A blank paired delta means the arm has no denominator for
    that outcome — `no_mpu` grants no order, so non-compliance is undefined rather than
    zero, and printing 0 there would be a different claim."""
    level = _num(row[key])
    if level is None:
        return '---'
    if row['arm'] == 'baseline':
        return f"{level:.{decimals}f} ({_num(row['se_' + key]):.{decimals}f})"
    delta = _num(row.get('paired_d_' + key))
    se = _num(row.get('paired_se_' + key))
    if delta is None or not se:
        return f"{level:.{decimals}f}"
    return f"{level:.{decimals}f}  {100 * delta / _num(base[key]):+.1f}%  z*={delta / se:+.1f}"


def build(metro, keep_all, suffix=DEFAULT_SUFFIX):
    path, rows = read(metro, suffix)
    base = next(r for r in rows if r['arm'] == 'baseline')
    rows = [r for r in rows if keep_all or r['arm'] != REDUNDANT]
    table = [[ARM_LABEL.get(r['arm'], r['arm'])]
             + [cell(r, base, key, dec) for key, _, dec in OUTCOMES]
             for r in rows]
    return path, base, table


def markdown(metro, keep_all, suffix=DEFAULT_SUFFIX):
    path, base, table = build(metro, keep_all, suffix)
    head = ['Arm'] + [label for _, label, _ in OUTCOMES]
    lines = [f'### {metro} — {base["runs"]} replicates, {base["steps"]} steps',
             '', '| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in table]
    lines += ['', f'Source: `output/{path.name}`.']
    return '\n'.join(lines)


def latex(metro, keep_all, suffix=DEFAULT_SUFFIX):
    path, base, table = build(metro, keep_all, suffix)
    body = ' \\\\\n'.join(' & '.join(c.replace('%', r'\%') for c in row) for row in table)
    head = ' & '.join(['Arm'] + [f'\\textbf{{{label}}}' for _, label, _ in OUTCOMES])
    return (f'% {metro} — {base["runs"]} replicates, from output/{path.name}\n'
            f'\\begin{{tabular}}{{l cccc}}\n\\toprule\n{head} \\\\\n\\midrule\n'
            f'{body} \\\\\n\\bottomrule\n\\end{{tabular}}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--metro', action='append',
                        help='metropolis; repeatable, defaults to both')
    parser.add_argument('--latex', action='store_true', help='emit a LaTeX tabular')
    parser.add_argument('--suffix', default=DEFAULT_SUFFIX,
                        help=f'suffix of the scenarios run to read, i.e. the -o given to '
                             f'scenarios.py (default {DEFAULT_SUFFIX!r}). Point this at the '
                             f'current run or the table silently rebuilds an older one.')
    parser.add_argument('--all', action='store_true',
                        help=f'keep the redundant {REDUNDANT} arm')
    args = parser.parse_args()

    for metro in args.metro or ['PORTO ALEGRE', 'BELO HORIZONTE']:
        print(latex(metro, args.all, args.suffix) if args.latex
              else markdown(metro, args.all, args.suffix))
        print()


if __name__ == '__main__':
    main()
