#!/usr/bin/env python3
"""Run the whole paper pipeline in sequence, unattended.

    python run_all.py                  # everything, 8 workers, ~7 h
    python run_all.py --dry-run        # print the plan and the commands, run nothing
    python run_all.py --only blockA    # one stage
    python run_all.py --only sens      # the three opt-in sensitivity axes, not in the default
    python run_all.py --only sens-pviol   # just one of them
    python run_all.py --only smoke --only sepwin   # the separation-window diagnostic
    python run_all.py --workers 32     # override the core budget

Runs the paper's commands in order. It streams a timestamped log to disk,
SKIPS stages whose outputs already exist so an interrupted run can simply be started again,
and on a failure skips the dependent stages while independent ones still run. The summary
at the end says which was which.

Stage A runs at the calibrated scale. Everything here is a scenario over the calibrated parameters: the arms override
parameters at run time and `settings.py` is never edited.

Nothing here needs the repository as the working directory — the script locates the root
from its own path and sets PYTHONPATH itself, so it can be launched from anywhere:

    nohup python /path/to/run_all.py > /dev/null 2>&1 &
"""

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT = ROOT / 'output'
PY = sys.executable

# Workers, not cores. Arms run sequentially and the replicates inside an arm run in
# parallel, with the pool capped at the replicate count -- so wall-clock goes as
# n_arms * ceil(iterates / workers) * seconds_per_run, and extra workers buy nothing until
# they cross a ceil() step. Measured on this project: 83 s per run at 1000 families and
# 144 steps.
DEFAULT_WORKERS = 8

CITIES = ['PORTO ALEGRE', 'BELO HORIZONTE']

# Stage A: the calibrated scale, -f 1000. At 5000 families lifetime prevalence shifts by
# z = -2.6.
BLOCK_A = dict(iterates=150, steps=144, families=1000, suffix='_final')

# Stage S: the sensitivity axes of the Robustness section, at the replicate count of stage
# A. These stages are OPT-IN: they run only when --only names them, since each costs hours.
# Suffixes end in 150, so the 40-replicate `_sens` files are not overwritten.
SENS_BLOCK = dict(iterates=150, steps=144, families=1000)

# One entry per axis. `arms` is the smallest set that makes the axis interpretable, because
# wall-clock is linear in it: an arm the axis cannot move is hours spent to confirm a zero.
SENS_AXES = [
    dict(name='sens-escalation', param='ESCALATION_STEP', suffix='_sens150',
         values=['0.010', '0.013', '0.020', '0.025', '0.030'],
         arms=['baseline', 'no_patrol', 'full_patrol', 'no_mpu'],
         # Wider than the calibrated window of ~2 thousandths, out to values where the
         # calibration itself fails, to test whether the policy conclusions survive there.
         # Same five values as the 40-replicate axis, so the two compare cell by cell.
         about='the published axis at stage A resolution -- resolves the no_mpu '
               'magnitude, which at 40 replicates carries an SE of 1.7 pp'),
    dict(name='sens-pviol', param='P_VIOL', suffix='_sensviol150',
         values=['0.0', '0.2', '0.4', '0.6', '0.8'],
         arms=['baseline', 'inert_mpu', 'no_mpu'],
         # P_VIOL is an assumption, and it is the channel `inert_mpu` leaves live: that arm
         # zeroes the boost, the monitoring and the arrest but not the removal from the
         # home, which is why it is not causally inert and why it carries Belo Horizonte.
         # 0.0 is the value at which the arm would BE inert, so the endpoint is the test.
         # Its reference arm no_mpu grants no orders, so the axis cannot move it -- it is
         # here because ARM_CONTRASTS pairs inert_mpu against it.
         about='the removal-from-the-home probability, assumed at 0.40 and never swept, '
               'on which the inert_mpu contrast rests'),
    dict(name='sens-postsep', param='POST_SEP_CONTACT_BASE', suffix='_senssep150',
         values=['0.0', '0.05', '0.15', '0.30', '0.50'],
         arms=['baseline', 'no_mpu'],
         # Not identified by the calibration moments. 0.0 switches the post-separation
         # channel off entirely.
         about='the post-separation contact hazard, declared unanchored but never '
               'swept -- 0.0 switches the channel off'),
]


def scenarios_cmd(metro, iterates, steps, families, suffix, workers,
                  arms=None, no_panel=False, sens=None):
    cmd = [PY, '-u', 'violence/scenarios.py', '-m', metro,
           '-i', str(iterates), '-s', str(steps), '-f', str(families),
           '-j', str(workers), '-o', suffix]
    if arms:
        cmd += ['-a', *arms]
    if no_panel:
        cmd.append('--no-panel')
    if sens:
        cmd += ['--sens', sens[0], *sens[1]]
    return cmd


def build_plan(workers):
    """The ordered stages. `produces` is what makes a stage resumable; `needs` is what
    makes a failure contained rather than cascading into meaningless output."""
    plan = [
        dict(name='smoke',
             about='2-replicate preflight, ~1 min -- fails fast on a broken install',
             cmd=scenarios_cmd('PORTO ALEGRE', 2, 36, 300, '_smoke', workers,
                               arms=['baseline', 'more_reporting'], no_panel=True),
             produces=[OUTPUT / 'scenarios_PORTO ALEGRE_smoke.csv'],
             needs=[], hours=0.02),
    ]
    for city in CITIES:
        plan.append(dict(
            name=f'blockA:{city}',
            about=f'{BLOCK_A["iterates"]} replicates x 8 arms at '
                  f'{BLOCK_A["families"]} families -- Table 1, the funnel, '
                  f'more_reporting x full_patrol, inert_mpu vs no_mpu',
            cmd=scenarios_cmd(city, workers=workers, **BLOCK_A),
            produces=[OUTPUT / f'scenarios_{city}{BLOCK_A["suffix"]}.csv'],
            needs=['smoke'], hours=8 * -(-BLOCK_A['iterates'] // workers) * 83 / 3600))
    plan.append(dict(
        name='table1',
        about='regenerate Table 1 from the stage A run, markdown and LaTeX',
        cmd=[PY, '-u', 'violence/table1.py', '--suffix', BLOCK_A['suffix']],
        produces=[OUTPUT / 'table1_final.md'],
        stdout_to=OUTPUT / 'table1_final.md',
        needs=[f'blockA:{c}' for c in CITIES], hours=0.01))
    plan.append(dict(
        name='table1-latex',
        about='the same table as a LaTeX tabular, for the manuscript',
        cmd=[PY, '-u', 'violence/table1.py', '--suffix', BLOCK_A['suffix'], '--latex'],
        produces=[OUTPUT / 'table1_final.tex'],
        stdout_to=OUTPUT / 'table1_final.tex',
        needs=[f'blockA:{c}' for c in CITIES], hours=0.01))
    for axis in SENS_AXES:
        plan.append(dict(
            name=axis['name'],
            about=f'OPT-IN: {axis["param"]} axis, {len(axis["values"])} values x '
                  f'{len(axis["arms"])} arms x {SENS_BLOCK["iterates"]} replicates -- '
                  f'{axis["about"]}',
            cmd=scenarios_cmd('PORTO ALEGRE', workers=workers, no_panel=True,
                              suffix=axis['suffix'], arms=axis['arms'],
                              sens=(axis['param'], axis['values']), **SENS_BLOCK),
            produces=[OUTPUT / f'scenarios_PORTO ALEGRE{axis["suffix"]}.csv'],
            needs=['smoke'], optional=True,
            hours=(len(axis['values']) * len(axis['arms'])
                   * -(-SENS_BLOCK['iterates'] // workers) * 83 / 3600)))
    for city in CITIES:
        plan.append(dict(
            name=f'sepwin:{city}',
            about='OPT-IN: held-out separation stock by assaulted years, with the life '
                  'table that tests whether the calibrated flow is a constant hazard',
            cmd=[PY, '-u', 'violence/separation_window.py', '-m', city,
                 '-i', str(BLOCK_A['iterates']), '-s', str(BLOCK_A['steps']),
                 '-j', str(workers), '-o', '_150rep'],
            produces=[OUTPUT / f'separation_window_{city}_150rep_hazard.csv'],
            needs=['smoke'], optional=True,
            hours=-(-BLOCK_A['iterates'] // workers) * 90 / 3600))
    return plan


# The versions the frozen calibration was produced on. Checked before anything long runs.
REQUIRED = {'mesa': '2.2.4'}


def preflight(log):
    """Verify the pinned dependencies before committing the night to a run.

    `mesa>=2.2,<3.0` resolves to 2.4.0, which warns on every place_agent call for an agent
    that already has a position. The model does that ~3,060 times per run and the message
    embeds the agent id, so Python cannot deduplicate it: block A would emit ~7.3 million
    warnings into the log while still producing numbers. The check fails before the run.

    Returns a list of problems; empty means the environment matches the calibration.
    """
    problems = []
    for package, wanted in REQUIRED.items():
        probe = subprocess.run(
            [PY, '-c', f'import {package}; print({package}.__version__)'],
            cwd=ROOT, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH='violence'))
        if probe.returncode != 0:
            problems.append(f'{package} is not importable: '
                            f'{probe.stderr.strip().splitlines()[-1:] or ["?"]}')
            continue
        found = probe.stdout.strip()
        if found != wanted:
            problems.append(
                f'{package} {found} installed, calibration was frozen on {wanted} '
                f'-- pip install "{package}=={wanted}"')
    for problem in problems:
        log(f'PREFLIGHT  {problem}')
    return problems


class Log:
    def __init__(self, path):
        self.path = path
        self.handle = path.open('a', encoding='utf-8')

    def __call__(self, message=''):
        stamp = dt.datetime.now().strftime('%H:%M:%S')
        line = f'[{stamp}] {message}' if message else ''
        print(line, flush=True)
        self.handle.write(line + '\n')
        self.handle.flush()


def run_stage(stage, log, force):
    """Run one stage, streaming its output to the console and the log."""
    done = [p for p in stage['produces'] if p.exists()]
    if done and not force:
        log(f"SKIP  {stage['name']} -- already produced {done[0].name} "
            f"(delete it, or pass --force, to redo)")
        return 'skipped', 0.0

    log(f"START {stage['name']} -- {stage['about']}")
    log(f"      {' '.join(stage['cmd'])}")
    env = dict(os.environ, PYTHONPATH='violence', PYTHONUNBUFFERED='1')
    started = time.time()
    sink = stage.get('stdout_to')
    handle = sink.open('w', encoding='utf-8') if sink else None
    captured = []
    try:
        process = subprocess.Popen(
            stage['cmd'], cwd=ROOT, env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            line = line.rstrip('\n')
            if handle:
                handle.write(line + '\n')
                captured.append(line)
            else:
                # The child's own progress lines go to the log verbatim, unstamped, so a
                # long run stays readable when read back in the morning.
                print(line, flush=True)
                log.handle.write(line + '\n')
        code = process.wait()
    finally:
        if handle:
            handle.close()
        log.handle.flush()
    elapsed = (time.time() - started) / 3600

    if code != 0:
        log(f"FAIL  {stage['name']} -- exit {code} after {elapsed:.2f} h")
        # A redirected stage swallows its own error message into the output file, which is
        # lost in the file -- surface it. And DELETE that file: a half-written one whose
        # contents are an error message would otherwise satisfy the `produces` check and
        # make the next run skip this stage as though it had succeeded.
        for line in captured[-12:]:
            log(f'      | {line}')
        if sink and sink.exists():
            sink.unlink()
            log(f'      (removed partial {sink.name})')
        return 'failed', elapsed
    log(f"OK    {stage['name']} -- {elapsed:.2f} h")
    return 'ok', elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-j', '--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'parallel replicates per arm (default {DEFAULT_WORKERS}). '
                             f'Make it a divisor of the replicate count to avoid idle '
                             f'slots in the last round.')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan and exit without running anything')
    parser.add_argument('--only', action='append',
                        help='run only stages whose name contains this; repeatable')
    parser.add_argument('--skip', action='append',
                        help='skip stages whose name contains this; repeatable')
    parser.add_argument('--force', action='store_true',
                        help='re-run stages whose outputs already exist')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='run even if the installed versions do not match the ones the '
                             'calibration was frozen on (see REQUIRED)')
    args = parser.parse_args()

    plan = build_plan(args.workers)
    if args.only:
        plan = [s for s in plan if any(k in s['name'] for k in args.only)]
    else:
        # Opt-in stages are not part of "the whole pipeline" -- naming one in --only is the
        # only way to run it, so a bare `python run_all.py` never grows an extra 9 hours.
        plan = [s for s in plan if not s.get('optional')]
    if args.skip:
        plan = [s for s in plan if not any(k in s['name'] for k in args.skip)]

    OUTPUT.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log = Log(OUTPUT / f'run_all_{stamp}.log')

    total = sum(s['hours'] for s in plan)
    log(f'SobreVIDA pipeline -- {len(plan)} stages, {args.workers} workers, '
        f'~{total:.1f} h estimated')
    log(f'root {ROOT}')
    log(f'log  {log.path}')
    log()
    for stage in plan:
        log(f"  {stage['name']:<24} ~{stage['hours']:>5.2f} h   {stage['about']}")
    log()

    problems = preflight(log)
    if problems and not args.skip_preflight:
        log('\nRefusing to start: the environment does not match the frozen calibration.')
        log('Fix the versions above, or pass --skip-preflight to run anyway.')
        return 2
    if problems:
        log('\n--skip-preflight given; continuing on a mismatched environment.')
    log()

    if args.dry_run:
        log('dry run -- nothing executed')
        return 0

    results, elapsed_total = {}, 0.0
    for stage in plan:
        blocked = [n for n in stage['needs'] if results.get(n) == 'failed']
        if blocked:
            log(f"BLOCK {stage['name']} -- needs {', '.join(blocked)}, which failed")
            results[stage['name']] = 'blocked'
            continue
        status, elapsed = run_stage(stage, log, args.force)
        results[stage['name']] = status
        elapsed_total += elapsed
        log()

    log('=' * 68)
    log(f'DONE in {elapsed_total:.2f} h')
    for name, status in results.items():
        log(f'  {status.upper():<8} {name}')
    failed = [n for n, s in results.items() if s in ('failed', 'blocked')]
    if failed:
        log(f'\n{len(failed)} stage(s) did not complete: {", ".join(failed)}')
        log('Re-running this script will resume: completed stages are skipped.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
