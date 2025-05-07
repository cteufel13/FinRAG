import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ESG_metric(metric_list: list = None):

    if not metric_list:
        print("No data provided.")
        return

    # Create DataFrame
    df = pd.DataFrame(metric_list)

    # Ensure Company column exists
    if "Company" not in df.columns:
        df["Company"] = [f"Company {i+1}" for i in range(len(df))]

    # Reorder columns: Company last
    company_col = df.pop("Company")
    df.insert(0, "Company", company_col)

    # Define ESG metrics (all columns except 'Company')
    metric_columns = [col for col in df.columns if col != "Company"]

    # Compute ESG total score (mean)
    df["Total_ESG"] = df[metric_columns].mean(axis=1)

    # Set index to Company for plotting
    df_plot = df.set_index("Company")[metric_columns]

    # Plot stacked bar chart
    ax = df_plot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")

    # Annotate total ESG score on top of each bar
    for idx, total in enumerate(df["Total_ESG"]):
        ax.text(
            idx,
            df_plot.iloc[idx].sum() + 1,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("ESG Component Breakdown and Total ESG Score per Company", fontsize=14)
    plt.xlabel("Company", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(title="ESG Component", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, df_plot.sum(axis=1).max() * 1.15)
    plt.tight_layout()
    plt.show()
