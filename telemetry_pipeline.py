import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fastf1
from fastf1 import plotting

# Enable FastF1 default plotting style and setup cache
plotting.setup_mpl()

# Setup cache directory to store downloaded telemetry data locally
CACHE_DIR = "fastf1_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)


class F1TelemetryAnalyzer:
    """A pipeline for ingesting, processing, and plotting F1 session telemetry data."""

    def __init__(self, year: int, grand_prix: str, session_type: str):
        """Initialize and load the session data.

        Args:
            year (int): Season year (e.g., 2023).
            grand_prix (str): Grand Prix location or name (e.g., 'Monza').
            session_type (str): Session code ('Q' for Qualifying, 'R' for Race,
              etc.).
        """
        self.year = year
        self.grand_prix = grand_prix
        self.session_type = session_type
        self.session = fastf1.get_session(year, grand_prix, session_type)
        self.session.load()

    def get_fastest_lap_telemetry(self, driver_code: str) -> pd.DataFrame:
        """Extract telemetry for a driver's fastest lap in the session.

        Args:
            driver_code (str): Driver 3-letter abbreviation (e.g., 'VER',
              'HAM').

        Returns:
            pd.DataFrame: Telemetry data frame containing distance, speed,
            throttle, etc.
        """
        driver_laps = self.session.laps.pick_driver(driver_code)
        fastest_lap = driver_laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry().add_distance()
        return telemetry

    def calculate_delta(
        self, ref_telemetry: pd.DataFrame, comp_telemetry: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the delta time between two drivers over distance.

        Args:
            ref_telemetry (pd.DataFrame): Telemetry of the reference driver
              (baseline).
            comp_telemetry (pd.DataFrame): Telemetry of the comparison driver.

        Returns:
            tuple[np.ndarray, np.ndarray]: Distance axis and calculated time
            delta in seconds.
        """
        # Interpolate comparison driver time onto the reference driver's distance grid
        comp_time_interpolated = np.interp(
            ref_telemetry["Distance"],
            comp_telemetry["Distance"],
            comp_telemetry["Time"].dt.total_seconds(),
        )

        ref_time = ref_telemetry["Time"].dt.total_seconds().to_numpy()
        delta_time = comp_time_interpolated - ref_time

        return ref_telemetry["Distance"].to_numpy(), delta_time

    def plot_telemetry_comparison(
        self,
        driver1: str,
        driver2: str,
        save_path: str = "telemetry_comparison.png",
    ):
        """Generate and save telemetry comparison graphs (Speed Trace & Time Delta).

        Args:
            driver1 (str): Reference driver code.
            driver2 (str): Comparison driver code.
            save_path (str): File path where the figure will be saved.
        """
        t1 = self.get_fastest_lap_telemetry(driver1)
        t2 = self.get_fastest_lap_telemetry(driver2)

        distance, delta_time = self.calculate_delta(t1, t2)

        # Plot setup
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        fig.suptitle(
            f"{self.year} {self.grand_prix} GP ({self.session_type}) - Fastest Lap Comparison\n{driver1} vs {driver2}",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Speed Trace Overlay
        ax1.plot(
            t1["Distance"],
            t1["Speed"],
            label=f"{driver1}",
            color=fastf1.plotting.get_driver_color(driver1, session=self.session),
        )
        ax1.plot(
            t2["Distance"],
            t2["Speed"],
            label=f"{driver2}",
            color=fastf1.plotting.get_driver_color(driver2, session=self.session),
            linestyle="--",
        )
        ax1.set_ylabel("Speed (km/h)")
        ax1.legend(loc="lower right")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 2. Delta Time Chart
        ax2.plot(distance, delta_time, color="white", linewidth=1.5)
        ax2.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax2.set_ylabel(f"Delta (s)\n(Positive = {driver2} slower)")
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. Throttle Trace
        ax3.plot(
            t1["Distance"],
            t1["Throttle"],
            color=fastf1.plotting.get_driver_color(driver1, session=self.session),
        )
        ax3.plot(
            t2["Distance"],
            t2["Throttle"],
            color=fastf1.plotting.get_driver_color(driver2, session=self.session),
            linestyle="--",
        )
        ax3.set_ylabel("Throttle (%)")
        ax3.set_xlabel("Lap Distance (m)")
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"[+] Plot saved successfully as '{save_path}'")
        plt.show()


if __name__ == "__main__":
    # Example execution: 2023 Monza Qualifying session between VER and LEC
    analyzer = F1TelemetryAnalyzer(
        year=2023, grand_prix="Monza", session_type="Q"
    )
    analyzer.plot_telemetry_comparison(driver1="VER", driver2="LEC")