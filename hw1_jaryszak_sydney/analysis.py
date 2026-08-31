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

if __name__ == "__main__":
    part1a_plot(temps, train)
    print(part1b_summary(temps))