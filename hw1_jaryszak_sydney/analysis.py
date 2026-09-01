import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

## Part 1

## Provided Data
temps = [86, 88, 91, 90, 93, 95, 94, 92, 89, 87,
88, 90, 92, 94, 96, 95, 93, 91, 90, 88]
train = temps[:14]
test = temps[14:]

def part1a_plot(temps, train, out_path="fig_1a_train_test_split.png"):
    days = np.arange(1, len(temps) + 1)
    train_days = days[: len(train)]
    test_days = days[len(train):]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_days, train, marker="o", linewidth=2, color="tab:blue", label="Training (days 1-14)")
    ax.plot(test_days, temps[len(train):], marker="o", linewidth=2, color="tab:red", label="Held-out (days 15-20)")

    ax.axvline(x=14.5, color="black", linestyle="--", linewidth=1.2, label="Forecast cutoff")

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Maximum Temperature (°F)")
    ax.set_title("Daily Maximum Temperature: Training vs. Held-Out Period")
    ax.set_xticks(days)
    ax.set_xlim(0.5, len(temps) + 0.5)
    ax.legend(frameon=True)

    fig.text(
        0.5,
        0.01,
        "Only information up to day 14 is legal at the forecast cutoff.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def part1b_summary(temps, cutoff=14):
    """Return the forecasting setup for the daily temperature series."""
    if not 0 < cutoff < len(temps):
        raise ValueError("cutoff must be strictly between 0 and len(temps)")

    return {
        "unit_of_analysis": "one day in the 20-day temperature series",
        "response": "daily maximum temperature (°F)",
        "information_cutoff": f"After day {cutoff}, only observations through day {cutoff} are legal to use for forecasting.",
        "forecast_target": f"The next {len(temps) - cutoff} future daily maximum temperatures: days {cutoff + 1} through {len(temps)}.",
    }


def part1c_descriptive_stats(train):
    """Return training-set descriptive summaries using the sample variance divisor n - 1."""
    x = np.asarray(train, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("train must contain at least two observations")

    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": n,
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "range": float(x.max() - x.min()),
        "variance": float(x.var(ddof=1)),
        "std_dev": float(x.std(ddof=1)),
        "iqr": float(q3 - q1),
        "variance_denominator": n - 1,
    }


def part1d_mean_ci(train, confidence=0.95):
    """Return a two-sided t confidence interval for the population mean."""
    x = np.asarray(train, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("train must contain at least two observations")

    mean = x.mean()
    std = x.std(ddof=1)
    se = std / np.sqrt(n)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = t_crit * se
    return {
        "confidence": confidence,
        "mean": float(mean),
        "std_dev": float(std),
        "n": n,
        "se": float(se),
        "t_crit": float(t_crit),
        "df": n - 1,
        "interval": (float(mean - margin), float(mean + margin)),
        "formula": "mean ± t_{1-α/2, n-1} · (s / √n)",
    }


def part1e_prediction_interval(train, confidence=0.95):
    """Return a two-sided t prediction interval for one new observation."""
    x = np.asarray(train, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("train must contain at least two observations")

    mean = x.mean()
    std = x.std(ddof=1)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = t_crit * std * np.sqrt(1.0 + 1.0 / n)
    return {
        "confidence": confidence,
        "mean": float(mean),
        "std_dev": float(std),
        "n": n,
        "t_crit": float(t_crit),
        "df": n - 1,
        "interval": (float(mean - margin), float(mean + margin)),
        "formula": "mean ± t_{1-α/2, n-1} · s · √(1 + 1/n)",
    }


## Part 2


def part2_forecasts(train, test):
    """Return fixed six-step mean and last-value baseline forecasts."""
    train_arr = np.asarray(train, dtype=float)
    test_arr = np.asarray(test, dtype=float)
    return {
        "mean_baseline": np.repeat(train_arr.mean(), len(test_arr)),
        "last_value_baseline": np.repeat(train_arr[-1], len(test_arr)),
    }


def part2a_plot(
    temps,
    train,
    test,
    mean_forecast,
    last_forecast,
    out_path="fig_2a_baseline_forecasts.png",
):
    """Overlay both baseline forecast paths on the ordered temperature series."""
    days = np.arange(1, len(temps) + 1)
    train_days = days[: len(train)]
    test_days = days[len(train) :]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_days, train, marker="o", linewidth=2, color="tab:blue", label="Training (days 1-14)")
    ax.plot(test_days, test, marker="o", linewidth=2, color="tab:red", label="Held-out actuals (days 15-20)")
    ax.plot(
        test_days,
        mean_forecast,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="tab:green",
        label="Mean baseline forecast",
    )
    ax.plot(
        test_days,
        last_forecast,
        marker="^",
        linestyle="--",
        linewidth=2,
        color="tab:orange",
        label="Last-value baseline forecast",
    )

    ax.axvline(x=14.5, color="black", linestyle="--", linewidth=1.2, label="Forecast cutoff")
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily Maximum Temperature (°F)")
    ax.set_title("Ordered Temperatures with Fixed Baseline Forecasts")
    ax.set_xticks(days)
    ax.set_xlim(0.5, len(temps) + 0.5)
    ax.legend(frameon=True)

    fig.text(
        0.5,
        0.01,
        "Both baselines are fixed at the cutoff and cannot update once the held-out period begins.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def part2b_metrics(test, mean_forecast, last_forecast):
    """Return MAE and RMSE for both baselines on the held-out period."""
    actual = np.asarray(test, dtype=float)
    mean_pred = np.asarray(mean_forecast, dtype=float)
    last_pred = np.asarray(last_forecast, dtype=float)

    def _errors(pred):
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        return mae, rmse

    mean_mae, mean_rmse = _errors(mean_pred)
    last_mae, last_rmse = _errors(last_pred)

    return {
        "mean_baseline": {"mae": mean_mae, "rmse": mean_rmse},
        "last_value_baseline": {"mae": last_mae, "rmse": last_rmse},
        "better_mae": "mean baseline" if mean_mae < last_mae else "last-value baseline",
        "better_rmse": "mean baseline" if mean_rmse < last_rmse else "last-value baseline",
    }


## Part 3

MELTING_POINT = {
    "n": 10,
    "sample_mean": 154.2,
    "sigma": 1.5,
    "mu0": 155.0,
    "alpha": 0.01,
    "mu_alt": 150.0,
}


def part3a_hypothesis_test(n=10, sample_mean=154.2, sigma=1.5, mu0=155.0, alpha=0.01):
    """Two-sided z test with known population standard deviation."""
    se = sigma / np.sqrt(n)
    z_stat = (sample_mean - mu0) / se
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    reject = abs(z_stat) > z_crit
    return {
        "hypotheses": {"H0": f"mu = {mu0}", "Ha": f"mu != {mu0}"},
        "alpha": alpha,
        "z_statistic": float(z_stat),
        "rejection_rule": f"|Z| > {z_crit:.4f}",
        "reject_H0": bool(reject),
        "decision": "reject H0" if reject else "fail to reject H0",
    }


def part3b_p_value(n=10, sample_mean=154.2, sigma=1.5, mu0=155.0):
    """Two-sided p-value for the known-sigma z test."""
    z_stat = (sample_mean - mu0) / (sigma / np.sqrt(n))
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    return {
        "z_statistic": float(z_stat),
        "p_value": p_value,
    }


def part3c_type_ii_error(n=10, sigma=1.5, mu0=155.0, alpha=0.01, mu_alt=150.0):
    """Probability of failing to reject H0 when the true mean is mu_alt."""
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    se = sigma / np.sqrt(n)
    accept_low = mu0 - z_crit * se
    accept_high = mu0 + z_crit * se
    beta = float(
        stats.norm.cdf(accept_high, loc=mu_alt, scale=se)
        - stats.norm.cdf(accept_low, loc=mu_alt, scale=se)
    )
    return {
        "acceptance_region": (float(accept_low), float(accept_high)),
        "mu_alt": mu_alt,
        "beta": beta,
    }


def part3d_verify(n=10, sample_mean=154.2, sigma=1.5, mu0=155.0, alpha=0.01, mu_alt=150.0):
    """Bundle Part 3 results for comparison with hand calculations."""
    test = part3a_hypothesis_test(
        n=n, sample_mean=sample_mean, sigma=sigma, mu0=mu0, alpha=alpha
    )
    pval = part3b_p_value(n=n, sample_mean=sample_mean, sigma=sigma, mu0=mu0)
    beta = part3c_type_ii_error(
        n=n, sigma=sigma, mu0=mu0, alpha=alpha, mu_alt=mu_alt
    )
    return {"hypothesis_test": test, "p_value": pval, "type_ii_error": beta}


def _print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


if __name__ == "__main__":
    part1a_plot(temps, train)
    _print_section("Part 1b: Forecasting setup")
    for key, value in part1b_summary(temps).items():
        print(f"  {key}: {value}")

    _print_section("Part 1c: Training descriptive summaries")
    stats_1c = part1c_descriptive_stats(train)
    print(f"  n = {stats_1c['n']}")
    print(f"  mean = {stats_1c['mean']:.4f}")
    print(f"  median = {stats_1c['median']:.4f}")
    print(f"  range = {stats_1c['range']:.4f}")
    print(
        f"  sample variance = {stats_1c['variance']:.4f} "
        f"(denominator n - 1 = {stats_1c['variance_denominator']})"
    )
    print(f"  sample std dev = {stats_1c['std_dev']:.4f}")
    print(f"  IQR = {stats_1c['iqr']:.4f}")

    _print_section("Part 1d: 95% t confidence interval for the mean")
    ci_1d = part1d_mean_ci(train)
    print(f"  formula: {ci_1d['formula']}")
    print(f"  t_crit(df={ci_1d['df']}) = {ci_1d['t_crit']:.4f}")
    print(f"  interval = ({ci_1d['interval'][0]:.4f}, {ci_1d['interval'][1]:.4f})")

    _print_section("Part 1e: 95% prediction interval for one new day")
    pi_1e = part1e_prediction_interval(train)
    print(f"  formula: {pi_1e['formula']}")
    print(f"  interval = ({pi_1e['interval'][0]:.4f}, {pi_1e['interval'][1]:.4f})")
    print(f"  width = {pi_1e['interval'][1] - pi_1e['interval'][0]:.4f}")
    print(f"  mean CI width = {ci_1d['interval'][1] - ci_1d['interval'][0]:.4f}")

    forecasts = part2_forecasts(train, test)
    part2a_plot(temps, train, test, forecasts["mean_baseline"], forecasts["last_value_baseline"])

    _print_section("Part 2a: Six-step baseline forecasts")
    for day, actual, mean_fc, last_fc in zip(
        range(15, 21), test, forecasts["mean_baseline"], forecasts["last_value_baseline"]
    ):
        print(
            f"  day {day}: actual={actual}, "
            f"mean baseline={mean_fc:.4f}, last-value baseline={last_fc:.4f}"
        )

    _print_section("Part 2b: Held-out forecast errors")
    metrics = part2b_metrics(test, forecasts["mean_baseline"], forecasts["last_value_baseline"])
    print(f"  mean baseline: MAE={metrics['mean_baseline']['mae']:.4f}, RMSE={metrics['mean_baseline']['rmse']:.4f}")
    print(
        f"  last-value baseline: MAE={metrics['last_value_baseline']['mae']:.4f}, "
        f"RMSE={metrics['last_value_baseline']['rmse']:.4f}"
    )
    print(f"  better MAE: {metrics['better_mae']}")
    print(f"  better RMSE: {metrics['better_rmse']}")

    _print_section("Part 3: One-sample inference (melting-point test)")
    part3 = part3d_verify(**MELTING_POINT)
    print(f"  z statistic = {part3['hypothesis_test']['z_statistic']:.4f}")
    print(f"  rejection rule (alpha=0.01): {part3['hypothesis_test']['rejection_rule']}")
    print(f"  decision: {part3['hypothesis_test']['decision']}")
    print(f"  two-sided p-value = {part3['p_value']['p_value']:.6f}")
    print(
        "  acceptance region for x-bar: "
        f"({part3['type_ii_error']['acceptance_region'][0]:.4f}, "
        f"{part3['type_ii_error']['acceptance_region'][1]:.4f})"
    )
    print(f"  beta at mu=150: {part3['type_ii_error']['beta']:.6f}")