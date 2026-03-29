import pandas as pd


def find_best_router_locations(results_file, top_n=10):
    """
    Parse router placement results and find the top N best locations.

    Args:
        results_file: Path to the results.txt file containing router placement data
        top_n: Number of top locations to return (default: 10)

    Returns:
        DataFrame with top N router locations sorted by quality metric

    Raises:
        FileNotFoundError: If the results file does not exist
    """
    # Read the CSV file
    df = pd.read_csv(results_file)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Extract signal columns
    signal_columns = [
        "living_room_signal",
        "bedroom_signal",
        "kitchen_signal",
        "bathroom_signal",
    ]

    # Calculate metrics for each location
    # We want to maximize signal strength (minimize loss)
    # Use average signal strength as the metric
    df["avg_signal"] = df[signal_columns].mean(axis=1)

    # Also calculate minimum signal strength (to ensure no dead zones)
    df["min_signal"] = df[signal_columns].min(axis=1)

    # Calculate standard deviation of signal strength (prefer balanced coverage)
    df["signal_std"] = df[signal_columns].std(axis=1)

    # Sort by average signal (descending) to get best locations
    # Secondary sort by minimum signal to avoid dead zones
    df_sorted = df.sort_values(["avg_signal", "min_signal"], ascending=[False, False])

    # Get top N results
    top_locations = df_sorted.head(top_n)[
        ["x_r", "y_r", "avg_signal", "min_signal", "signal_std"] + signal_columns
    ]

    return top_locations


def print_results(top_locations):
    """
    Print the top router locations in a formatted table.

    Args:
        top_locations: DataFrame containing top router locations
    """
    print("\n" + "=" * 120)
    print("TOP 10 BEST ROUTER PLACEMENT LOCATIONS")
    print("=" * 120)
    print("\nMetrics:")
    print("  - avg_signal: Average signal strength across all rooms (higher is better)")
    print(
        "  - min_signal: Minimum signal strength (lower is better, should be > -70 dB)"
    )
    print(
        "  - signal_std: Standard deviation of signal (lower is better for balanced coverage)"
    )
    print("\n")

    for idx, (_, row) in enumerate(top_locations.iterrows(), 1):
        print(f"Rank {idx}: Position ({row['x_r']:.2f}, {row['y_r']:.2f})")
        print(f"  Average Signal: {row['avg_signal']:7.2f} dB")
        print(f"  Min Signal:     {row['min_signal']:7.2f} dB")
        print(f"  Std Dev:        {row['signal_std']:7.2f} dB")
        print(f"  Living Room:    {row['living_room_signal']:7.2f} dB")
        print(f"  Bedroom:        {row['bedroom_signal']:7.2f} dB")
        print(f"  Kitchen:        {row['kitchen_signal']:7.2f} dB")
        print(f"  Bathroom:       {row['bathroom_signal']:7.2f} dB")
        print()


def main():
    """
    Main function to find and display the top 10 best router locations.
    """
    results_file = (
        "/home/wessel/Documents/Scientific Computing/Assignment 3/src/results.txt"
    )

    try:
        top_locations = find_best_router_locations(results_file, top_n=10)
        print_results(top_locations)

        # Also save results to a CSV file
        output_file = "/home/wessel/Documents/Scientific Computing/Assignment 3/src/top_10_routers.csv"
        top_locations.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Could not find file {results_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
