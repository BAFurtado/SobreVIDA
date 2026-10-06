from data import *
from agents import Woman

_12_MONTHS = 12


def merge_df(*dfs) -> pd.DataFrame:
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, left_index=True, right_index=True)
    return merged


def _get_agents(home):
    return list(home.schedule.agents_t) + list(home.schedule.dead_agents)


def _get_municipio(agent):
    return mun_codes[
        mun_codes['cod_mun'].astype(str) == str(agent.address)[:7]
    ]['nome_mun']


def year_register(home: model.Home, year):
    muns, areaps = {}, {}

    for agent in _get_agents(home):
        mun = _get_municipio(agent)

        # AREAP aggregation
        if agent.address not in areaps:
            areaps[agent.address] = edataframe(agent)
        else:
            fill_data(agent, areaps)

        # Municipality aggregation
        if len(mun.values):
            mun_name = mun.values[0]
            if mun_name not in muns:
                muns[mun_name] = edataframe(agent)
            else:
                fill_data(agent, muns, True, mun)

    reset_annual_counters(home)

    return fdataframe(muns, year), fdataframe(areaps, year, label='AREAP')


def reset_annual_counters(home):
    """Clear the counters that are read as within-year quantities.

    `denounce`, `break_times` and `remain_times` are also read by
    Woman.peer_action_norm, so the reset is behavioural, not only bookkeeping: it keeps
    the peer norm a norm of the current year. Any harness that steps the model a year at
    a time must call this, or it calibrates a different model from the one main() runs.
    """
    for agent in _get_agents(home):
        agent.got_attacked = 0
        agent.denounce = False
        agent.break_times = 0
        agent.remain_times = 0
        if isinstance(agent, Woman):
            agent.action_times = 0
            agent.denounce_times = 0

    home.schedule.dead_agents = set()


def action_year_counts(home):
    """Victim-side annual moments: among women assaulted this year, how many acted.

    The two calibration targets of the action channel are annual rates with the
    ASSAULTED women as denominator (denunciation in [0.142, 0.257], separation 0.15),
    while the AREAP/município tables carry rates over ALL adult women (see
    _compute_rates), so neither target is recoverable from those tables.

    Reads the annual counters, so it must be called at a year boundary and BEFORE
    reset_annual_counters. Dead agents are included: a woman killed during the year was
    assaulted during it.
    """
    # Two levels, deliberately kept apart. The per-woman flags (`acted`, `denounced`,
    # `broke`, `remained`) are the calibration targets, which are shares of victims.
    # The `*_events` totals are per-act and are what ACTION_DENOUNCE_SHARE governs; a
    # woman can act more than once in a year, so the two levels do not agree.
    women = attacked = acted = denounced = broke = remained = 0
    assaults = action_events = denounce_events = break_events = 0
    props, thresholds = [], []
    # Denunciations of each woman who denounced at least once this year. This is the
    # unit the PCMG holds — occurrences per victim-year, conditional on having a record
    # — and neither `denounced` (a share of victims) nor `denounce_events` (a total)
    # recovers its distribution. Validation only.
    denounce_counts = []

    for agent in _get_agents(home):
        if not isinstance(agent, Woman):
            continue
        women += 1
        if not agent.got_attacked:
            continue
        attacked += 1
        assaults += agent.got_attacked
        props.append(agent.disposition_action)
        thresholds.append(agent.threshold_action)

        action_events += agent.action_times
        denounce_events += agent.denounce_times
        break_events += agent.break_times

        if agent.action_times:
            acted += 1
        if agent.denounce:
            denounced += 1
            denounce_counts.append(agent.denounce_times)
        if agent.break_times:
            broke += 1
        if agent.remain_times:
            remained += 1

    return {'women': women, 'attacked': attacked, 'assaults': assaults,
            'acted': acted, 'denounced': denounced, 'broke': broke,
            'remained': remained,
            'action_events': action_events, 'denounce_events': denounce_events,
            'break_events': break_events,
            'props': props, 'thresholds': thresholds,
            'denounce_counts': denounce_counts}


def separation_stock_moment(homes, metro, write=True):
    """Held-out moment: among women ever assaulted, the share who ever left.

    DataSenado 2025 reports a STOCK — 79% of the women who reported assault by a
    husband/partner were ex-partners at interview — while ACTION_THRESHOLD is calibrated
    against a FLOW (0.15 separations per assaulted woman-year). The stock is never used to
    tune anything, so it is a test of the specification rather than a target: the model
    emits the accumulated share over the run and it is compared to 0.79 directly, with no
    hazard-to-stock conversion assumed on either side.

    Two definitions, because the survey question and the model's separation flow are not
    the same object:
      - `share_ever_broke` — she ended a relationship THROUGH the action stage, which is
        the accumulated counterpart of the 0.15 flow. Strict, and the primary number.
      - `share_ex_partner` — a relationship in which she was assaulted has ended, by any
        route (she left, he left, partner rotation). Reads the lifetime
        `ever_separated_from_aggressor` flag. Broader, and the closer counterpart to what
        the interviewer records.

    Denominator is LIVING adult women, matching a survey of living respondents; the
    all-women variant includes the dead and is reported alongside, since excluding
    femicide victims drops women who were assaulted and never left.

    The run length caps what is reachable: 144 steps is 12 years of exposure, and women
    assaulted late in the run have had less than that, so the emitted share is a floor on
    what the same hazard would produce over a full adult lifetime.
    """
    ever_attacked = ever_broke = ex_partner = 0
    ever_attacked_all = ever_broke_all = 0
    assaults = 0

    for home in homes:
        for agent in _get_agents(home):
            if not isinstance(agent, Woman) or agent.age <= 18:
                continue
            if not agent.ever_attacked:
                continue
            ever_attacked_all += 1
            if agent.ever_broke:
                ever_broke_all += 1
            if agent.is_dead:
                continue
            ever_attacked += 1
            assaults += agent.assaults_lifetime
            if agent.ever_broke:
                ever_broke += 1
            if agent.ever_separated_from_aggressor:
                ex_partner += 1

    row = {
        'metro': metro,
        'iterations': len(homes),
        'ever_attacked_living': ever_attacked,
        'ever_broke_living': ever_broke,
        'share_ever_broke': (ever_broke / ever_attacked) if ever_attacked else float('nan'),
        'ex_partner_living': ex_partner,
        'share_ex_partner': (ex_partner / ever_attacked) if ever_attacked else float('nan'),
        # How many times the average victim reached the action stage at all. The stock is
        # a function of this, not of elapsed years: the decision is drawn per assault.
        'assaults_per_victim': (assaults / ever_attacked) if ever_attacked else float('nan'),
        'ever_attacked_all': ever_attacked_all,
        'share_ever_broke_all': (ever_broke_all / ever_attacked_all) if ever_attacked_all
        else float('nan'),
    }
    if write:
        pd.DataFrame([row]).to_csv(f'output/separation_stock_{metro}.csv', index=False)
    return row


def prevalence_of_violence(homes):
    prevalence, total = {}, {}

    for home in homes:
        for agent in _get_agents(home):
            addr = agent.address

            total[addr] = total.get(addr, 0) + 1

            if agent.violence_prevalence:
                if addr not in prevalence:
                    prevalence[addr] = dict(
                        zip(range(_12_MONTHS), agent.violence_prevalence.values())
                    )
                else:
                    for month in range(_12_MONTHS):
                        prevalence[addr][month] = (
                            prevalence[addr].get(month, 0)
                            + agent.violence_prevalence[month]
                        )

    df = pd.DataFrame(
        [
            (key, *[prevalence[key][i] for i in prevalence[key]])
            for key in prevalence
        ],
        columns=[
            'AREAP', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'
        ]
    )

    df['occr'] = [total[row[0]] for row in df.values]
    df['mean'] = [
        sum(row[1:13]) / total[row[0]]
        for row in df.values
    ]

    return df


def mpu_summary(homes, metro, write=True):
    """Aggregate the MPU instrumentation counters across iterations.

    Measurement only: reads counters accumulated by the existing mechanism.
    Returns a one-row dict; the three emergent moments are held out for validation
    (non-compliance, and femicide-with-prior-MPU under two definitions).
    """
    import settings
    requested = sum(h.mpu_requested_count for h in homes)
    granted = sum(h.mpu_granted_count for h in homes)
    violated = sum(h.mpu_grant_violated_count for h in homes)
    fem = sum(h.femicide_count for h in homes)
    fem_ever = sum(h.femicide_with_ever_mpu for h in homes)
    fem_rel = sum(h.femicide_with_relationship_mpu for h in homes)

    row = {
        'metro': metro,
        'iterations': len(homes),
        # Parameters in force (self-describing rows for the sweep)
        'MPU_THRESHOLD_BOOST': settings.MPU_THRESHOLD_BOOST,
        'P_VIOL': settings.P_VIOL,
        'P_MPU_REQUESTED': settings.P_MPU_REQUESTED,
        'P_MPU_GRANTED': settings.P_MPU_GRANTED,
        'MPU_MONITORING_LEVEL': settings.MPU_MONITORING_LEVEL,
        'MPU_DURATION': settings.MPU_DURATION,
        # Order-level non-compliance moment (target 0.21 +/- 0.04, see settings)
        'mpu_requested': requested,
        'mpu_granted': granted,
        'mpu_violated': violated,
        'non_compliance_rate': (violated / granted) if granted else float('nan'),
        # Femicide-composition moment (CNJ target 0.324)
        'femicides': fem,
        'femicide_with_ever_mpu': fem_ever,
        'share_femicide_ever_mpu': (fem_ever / fem) if fem else float('nan'),
        'femicide_with_relationship_mpu': fem_rel,
        'share_femicide_relationship_mpu': (fem_rel / fem) if fem else float('nan'),
        # Post-separation channel. `post_sep_contacts` is the conduct art. 24-A actually
        # defines, so it is the numerator the FBSP-style non-compliance rate needs.
        'POST_SEP_CONTACT_BASE': settings.POST_SEP_CONTACT_BASE,
        'POST_SEP_WINDOW': settings.POST_SEP_WINDOW,
        'post_sep_contacts': sum(h.post_sep_contact_count for h in homes),
        'post_sep_assaults': sum(h.post_sep_assault_count for h in homes),
        'femicide_post_separation': sum(h.femicide_post_separation for h in homes),
    }
    if write:
        pd.DataFrame([row]).to_csv(f'output/mpu_summary_{metro}.csv', index=False)
    return row


def _compute_rates(df):
    """Rates over ALL adult women in the cell, the population at risk.

    Not the same denominator as the action-channel targets, which are shares of the women
    ASSAULTED in the year — those come from action_year_counts. An AREAP with no adult
    women yields NaN rather than a division by zero.
    """
    females = df['Females'].replace(0, pd.NA)
    df['Attacks per female'] = df['Number of attacks'] / females
    df['Denounce rate'] = df['Denounce'] / females
    df['Break rate'] = df['Break'] / females      # separation events per adult woman-year
    df['Remain rate'] = df['Remain'] / females    # stay decisions per adult woman-year
    return df


def _compute_real_numbers(df, metro, number_of_agents_mean, population):
    df['Number of real attacks'] = (
        population[metro] * df['Number of attacks'] // number_of_agents_mean
    )
    return df


def main(metro='CAMPO GRANDE', iterates=2, steps=100):

    metropolis_number_of_womans = {
        'BELO HORIZONTE': 1192740,
        'BRASILIA': 1450951,
        'CAMPO GRANDE': 462521,
        'FORTALEZA': 1250784,
        'GOIANIA': 740243,
        'PORTO ALEGRE': 686415,
        'RECIFE': 766793,
        'RIO DE JANEIRO': 3198779,
        'SALVADOR': 1245104,
        'SAO PAULO': 5897779,
        'VITORIA': 166277
    }

    result_mun = pd.DataFrame(columns=[
        'Year', 'Município', 'Attacked', 'Denounce', 'Females',
        'Femicide', 'Number of attacks', 'Break', 'Remain'
    ])

    result_areap = pd.DataFrame(columns=[
        'Year', 'AREAP', 'Attacked', 'Denounce', 'Females',
        'Femicide', 'Number of attacks', 'Break', 'Remain'
    ])

    homes, number_of_agents = [], []

    for _ in range(iterates):
        home = model.Home(metro=metro)
        homes.append(home)

        start_year = 2010

        for step in range(1, steps + 1):
            home.step()

            if step % _12_MONTHS == 0:
                home.update_incarcerated()

                df_mun, df_areap = year_register(home, start_year)

                result_mun = pd.concat([result_mun, df_mun], ignore_index=True)
                result_areap = pd.concat([result_areap, df_areap], ignore_index=True)

                start_year += 1

        number_of_agents.append(len(_get_agents(home)))
        print('n. agents:', number_of_agents)

    number_of_agents_mean = sum(number_of_agents) // iterates * 50

    prevalence_data = prevalence_of_violence(homes)

    prevalence_data['Number of real cases'] = (
        metropolis_number_of_womans[metro]
        * prevalence_data.iloc[:, 1:13].sum(axis=1)
        // number_of_agents_mean
    )

    prevalence_data.to_csv(f'output/prevalence_last_12_months_{metro}.csv', index=False)

    result_areap = result_areap.groupby('AREAP').sum().reset_index()
    result_mun = result_mun.groupby(['Município', 'Year']).sum().reset_index()

    mun_total = result_mun.groupby('Município').sum().reset_index()
    areap_total = result_areap.groupby('AREAP').sum().reset_index()

    # Rates
    result_areap = _compute_rates(result_areap)
    result_mun = _compute_rates(result_mun)
    areap_total = _compute_rates(areap_total)
    mun_total = _compute_rates(mun_total)

    # Real numbers
    result_areap = _compute_real_numbers(result_areap, metro, number_of_agents_mean, metropolis_number_of_womans)
    result_mun = _compute_real_numbers(result_mun, metro, number_of_agents_mean, metropolis_number_of_womans)
    areap_total = _compute_real_numbers(areap_total, metro, number_of_agents_mean, metropolis_number_of_womans)
    mun_total = _compute_real_numbers(mun_total, metro, number_of_agents_mean, metropolis_number_of_womans)

    # Outputs
    result_mun.to_csv('output/output_MUN.csv')
    result_areap.to_csv('output/output_AREAP.csv')

    mun_total.to_csv(f'output/output_total_MUN_{metro}.csv')
    areap_total.to_csv(f'output/output_total_AREAP_{metro}.csv')

    # MPU emergent moments (measurement only; held out for validation)
    m = mpu_summary(homes, metro)
    print('--- MPU moments ---')
    print(f"  boost={m['MPU_THRESHOLD_BOOST']}  P_VIOL={m['P_VIOL']}  "
          f"grants={m['mpu_granted']}  femicides={m['femicides']}")
    print(f"  non-compliance rate        = {m['non_compliance_rate']:.3f}  (FBSP 0.183 nat / 0.232 RS)")
    print(f"  femicide w/ prior MPU (ever)= {m['share_femicide_ever_mpu']:.3f}  (CNJ 0.324)")
    print(f"  femicide w/ prior MPU (rel) = {m['share_femicide_relationship_mpu']:.3f}")
    print(f"  post-sep contacts/assaults  = {m['post_sep_contacts']} / {m['post_sep_assaults']}"
          f"   femicides post-sep = {m['femicide_post_separation']}")

    # Accumulated separation (held out; DataSenado 2025 stock, never calibrated against)
    s = separation_stock_moment(homes, metro)
    print('--- separation stock (held out) ---')
    print(f"  ever-assaulted living women = {s['ever_attacked_living']}")
    print(f"  share who ever left (action)= {s['share_ever_broke']:.3f}  (DataSenado 0.79)")
    print(f"  share now ex-partner (any)  = {s['share_ex_partner']:.3f}")
    print(f"  assaults per victim (life)  = {s['assaults_per_victim']:.2f}")
    m.update({k: v for k, v in s.items() if k not in ('metro', 'iterations')})
    return m


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='SobreVIDA',
        description='A simulation model of domestic violence'
    )

    parser.add_argument('-i', '--iterations', default=5)
    parser.add_argument('-s', '--steps', default=144)

    args = vars(parser.parse_args())

    metropolis = [
        'BELO HORIZONTE', 'BRASILIA', 'CAMPO GRANDE',
        'FORTALEZA', 'GOIANIA', 'PORTO ALEGRE', 'RECIFE',
        'RIO DE JANEIRO', 'SALVADOR', 'SAO PAULO', 'VITORIA'
    ]

    for m in metropolis:
        main(metro=m, iterates=int(args['iterations']), steps=int(args['steps']))