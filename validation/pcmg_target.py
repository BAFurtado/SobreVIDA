"""The Belo Horizonte held-out target: police occurrences per victim-year.

Polícia Civil de Minas Gerais, 230,205 occurrences with women victims, all of them in
Belo Horizonte, 2010-2023, WITH a person identifier (`IDvítima`) — the column no Porto
Alegre source has, and the reason BH can be an out-of-sample test rather than an
assertion of transfer.

The construct is `denounce_times` among women who denounced at least once in the year:
the model's `denounce` IS a police record by definition, so nothing is converted between
flow and stock or between temporal aggregations. Restricted to the natures the model's
assault stands for (AMEACA, LESAO CORPORAL, VIAS DE FATO) and to intimate-partner
relations, over the model's window (2010-2021).

Nothing here feeds calibration. The moment is validation only.

Usage (from project root):
    python validation/pcmg_target.py

Writes output/pcmg_target_BELO HORIZONTE.csv — one row for the pooled moment plus one
row per year, which is where the tolerance band comes from.
"""
import argparse

import pandas as pd

SOURCE = 'validation/shapes/PCMG.csv'
OUT = 'output/pcmg_target_BELO HORIZONTE.csv'

# The natures the model's assault represents, the same three the Porto Alegre
# denominator uses.
NATURES = ['AMEACA', 'LESAO CORPORAL', 'VIAS DE FATO / AGRESSAO']

# Intimate partner, current or former. PASSIONAL and RELACIONAMENTO EXTRA-CONJUGAL are
# left out: both are relations the model has no counterpart for, and including them moves
# the moment by 0.0006, reported below as a robustness line.
RELATIONS = ['CONJUGE / COMPANHEIRO', 'EX-CONJUGE / EX-COMPANHEIRO',
             'NAMORADO (A)', 'CONJUGE', 'EX-CONJUGE']
RELATIONS_WIDE = RELATIONS + ['PASSIONAL (NAMORO/CASO/COMPANHEIRO(A))',
                              'RELACIONAMENTO EXTRA-CONJUGAL']

# The model runs 144 steps from 2010 (violence/homicide.py carries municipal series only
# to 2021), so the target window is the model's window.
YEAR_MIN, YEAR_MAX = 2010, 2021

# Frequency bands, matched to the ones the FBSP publishes so the two shapes can be read
# side by side.
BANDS = [(0, 1, '1'), (1, 2, '2'), (2, 6, '3-6'), (6, 10 ** 6, '7+')]


def victim_years(source=SOURCE, relations=RELATIONS):
    """Occurrences per (victim, year), the unit the moment is a mean over."""
    df = pd.read_csv(source, low_memory=False)
    keep = (df['DescriçãoSubclasseNatPrincipal'].isin(NATURES)
            & df['RelaçãoVítimaAutor'].isin(relations)
            & df['AnoFato'].between(YEAR_MIN, YEAR_MAX))
    df = df[keep]
    return df.groupby(['IDvítima', 'AnoFato']).size(), df['IDvítima'].nunique()


def shape(counts):
    """Share of victim-years in each frequency band."""
    out = {}
    for lo, hi, label in BANDS:
        out[f'share_{label}'] = float(((counts > lo) & (counts <= hi)).mean())
    return out


def run(source=SOURCE, out_path=OUT):
    counts, victims = victim_years(source)
    per_year = counts.groupby(level=1).mean()

    pooled = {'window': f'{YEAR_MIN}-{YEAR_MAX}',
              'victim_years': int(len(counts)),
              'victims': int(victims),
              'occurrences': int(counts.sum()),
              'occ_per_victim_year': float(counts.mean()),
              # The band is the dispersion of the source across its own years, not a
              # standard error.
              'year_min': float(per_year.min()),
              'year_max': float(per_year.max()),
              'half_band': float((per_year.max() - per_year.min()) / 2)}
    pooled.update(shape(counts))

    rows = [pooled]
    for year, mean in per_year.items():
        n = counts.xs(int(year), level=1)
        row = {'window': str(int(year)), 'victim_years': int(len(n)),
               'victims': int(len(n)), 'occurrences': int(n.sum()),
               'occ_per_victim_year': float(mean)}
        row.update(shape(n))
        rows.append(row)

    # Same moment with the two ambiguous relation codes added, so the reader can see the
    # filter is not carrying the result.
    wide, _ = victim_years(source, RELATIONS_WIDE)
    rows.append({'window': 'wide_relations', 'victim_years': int(len(wide)),
                 'occurrences': int(wide.sum()),
                 'occ_per_victim_year': float(wide.mean()), **shape(wide)})

    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"TARGET  occurrences per victim-year = {pooled['occ_per_victim_year']:.4f}"
          f"  +- {pooled['half_band']:.4f}")
    print(f"        over {pooled['victim_years']} victim-years, "
          f"{pooled['victims']} victims, {pooled['occurrences']} occurrences")
    print('SHAPE   ' + '  '.join(f"{lb}: {pooled[f'share_{lb}']:.4f}"
                                 for _, _, lb in BANDS))
    print(f"BAND    year means {pooled['year_min']:.4f} to {pooled['year_max']:.4f}")
    print(f'\n-> {out_path}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=SOURCE)
    p.add_argument('-o', '--out', default=OUT)
    run(p.parse_args().source, p.parse_args().out)
