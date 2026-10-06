# SobreVIDA

An agent-based model of intimate-partner violence and of the Brazilian protective-order
chain (*medidas protetivas de urgência*), for the metropolitan regions of Porto Alegre
and Belo Horizonte. Python 3.12, built on mesa 2.2.4.

## Install

```bash
pip install -r requirements.txt
```

## Run

All commands are run from the repository root.

One model run, with default outputs written to `output/`:

```bash
python sobrevida.py -m "PORTO ALEGRE" -i 5 -s 144
```

The policy scenarios (arms against the calibrated baseline):

```bash
PYTHONPATH=violence python violence/scenarios.py -m "PORTO ALEGRE" -i 10 -s 144 -j 8
```

The whole paper pipeline, unattended (about 7 hours on 8 workers). It skips stages
whose outputs already exist, so an interrupted run can be restarted:

```bash
python run_all.py --dry-run     # print the plan
python run_all.py               # run it
```

Tables from finished runs:

```bash
PYTHONPATH=violence python violence/table1.py --suffix _final
PYTHONPATH=violence python violence/table_quartile.py --suffix _final
```

Every script takes `--help`.

## Layout

| Path | Contents |
|---|---|
| `violence/model.py`, `agents.py`, `schedule.py` | the model |
| `violence/settings.py` | every parameter, with its source and how it was set |
| `violence/scenarios.py` | the policy arms |
| `violence/*_sweep.py`, `persistence_diagnostic.py`, `femicide_scale.py` | calibration harnesses |
| `violence/separation_stock.py`, `separation_window.py`, `denounce_repeat.py` | held-out validation |
| `violence/table1.py`, `table_quartile.py` | paper tables |
| `violence/input/` | census inputs (IBGE 2010, by weighting area) and the population generator |
| `validation/pcmg_target.py` | the Belo Horizonte validation target |

The Belo Horizonte target is computed from police microdata (Polícia Civil de Minas
Gerais) that are not redistributed here. The derived target is included as
`output/pcmg_target_BELO HORIZONTE.csv`, and `violence/denounce_repeat.py` carries its
values.

## License

MIT. See `LICENSE`.
