# Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set team colors if requested
    if use_team_colors and color is None:
        colors_list = [get_team_color(team) for team in plot_df[x_col]]
    else:
        colors_list = color
    
    # Create the bar plot
    if horizontal:
        bars = ax.barh(plot_df[x_col], plot_df[y_col], color=colors_list)
    else:
        bars = ax.bar(plot_df[x_col], plot_df[y_col], color=colors_list)
    
    # Add data labels on top/end of bars
    if show_values:
        for bar in bars:
            if horizontal:
                width = bar.get_width()
                x_pos = width + (width * 0.01)
                y_pos = bar.get_y() + bar.get_height() / 2
                ha = 'left'
                va = 'center'
            else:
                height = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = height + (height * 0.01)
                ha = 'center'
                va = 'bottom'
            
            ax.text(x_pos, y_pos, f'{plot_df[y_col].iloc[bars.index(bar)]:{value_format}}', 
                   ha=ha, va=va, fontweight='bold')
    
    # Customize appearance
    ax.set_title(title, fontsize=16, pad=20)
    
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    
    # Set grid
    ax.grid(grid, axis='y' if horizontal else 'x', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if not horizontal
    if not horizontal:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved bar plot to {out}")
    
    return fig


def line_plot(
    df: pd.DataFrame, 
    x_col: str, 
    y_cols: List[str], 
    title: str, 
    out: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend_title: Optional[str] = None,
    grid: bool = True,
    dpi: int = 300,
    date_format: Optional[str] = None,
    use_team_colors: bool = False
) -> plt.Figure:
    """Create a multi-line plot visualization.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis lines
        title: Chart title
        out: Optional output file path
        figsize: Figure size as (width, height) tuple
        colors: Optional list of line colors
        markers: Optional list of markers for lines
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        legend_title: Optional title for legend
        grid: Whether to show grid
        dpi: Resolution for saved image
        date_format: Optional format for date x-axis
        use_team_colors: Whether to use team colors (if y_cols contains team names)
        
    Returns:
        Matplotlib figure
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for line plot")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set team colors if requested
    if use_team_colors and colors is None:
        colors = [get_team_color(col) for col in y_cols]
    
    # Set default markers if not provided
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
        
    # If too few markers, cycle through the list    
    if len(markers) < len(y_cols):
        markers = markers * (len(y_cols) // len(markers) + 1)
    
    # Plot each line
    for i, col in enumerate(y_cols):
        color = colors[i] if colors and i < len(colors) else None
        marker = markers[i % len(markers)]
        ax.plot(df[x_col], df[col], marker=marker, linewidth=2, label=col, color=color)
    
    # Handle date x-axis
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        # Format x-axis as dates
        if date_format:
            date_formatter = mdates.DateFormatter(date_format)
            ax.xaxis.set_major_formatter(date_formatter)
        else:
            # Auto-select date format based on range
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        
        # Rotate date labels for readability
        plt.xticks(rotation=45, ha='right')
    
    # Customize appearance
    ax.set_title(title, fontsize=16, pad=20)
    
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    
    # Set grid
    ax.grid(grid, linestyle='--', alpha=0.7)
    
    # Add legend
    if legend_title:
        ax.legend(title=legend_title)
    else:
        ax.legend()
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved line plot to {out}")
    
    return fig


def form_heatmap(
    teams: List[str], 
    season: int, 
    last_n: int = 5, 
    out: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    dpi: int = 300,
    show_match_info: bool = True
) -> plt.Figure:
    """Create a heatmap showing form of multiple teams.
    
    Args:
        teams: List of team names
        season: Season year
        last_n: Number of recent matches to include
        out: Optional output file path
        figsize: Optional figure size
        title: Optional custom title
        dpi: Resolution for saved image
        show_match_info: Whether to show opponent and score in cells
        
    Returns:
        Matplotlib figure
    """
    from database_utils import execute_query
    
    # Calculate appropriate figure size based on number of teams
    if figsize is None:
        figsize = (12, len(teams) * 0.8 + 2)
    
    # Get match data for all teams
    results_data = []
    
    for team in teams:
        query = """
        SELECT date, team, opponent, result, gf, ga, venue 
        FROM matches 
        WHERE team = :team AND season = :season
        ORDER BY date DESC 
        LIMIT :limit
        """
        
        df = execute_query(query, {"team": team, "season": season, "limit": last_n})
        
        if not df.empty:
            # Reverse to show oldest to newest
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Create result array with color codes
            result_array = []
            for _, row in df.iterrows():
                if row["result"] == "W":
                    result_array.append(3)  # Win
                elif row["result"] == "D":
                    result_array.append(1)  # Draw
                else:
                    result_array.append(0)  # Loss
            
            results_data.append({
                "team": team,
                "results": result_array,
                "opponents": df["opponent"].tolist(),
                "scores": [f"{r['gf']}-{r['ga']}" for _, r in df.iterrows()],
                "venues": df["venue"].tolist()
            })
    
    if not results_data:
        logger.warning("No data found for form heatmap")
        return plt.figure()
        
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom color map: red for loss, yellow for draw, green for win
    cmap = colors.ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
    bounds = [-0.5, 0.5, 1.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Prepare data for heatmap
    data = np.zeros((len(teams), last_n))
    yticks = []
    
    for i, team_data in enumerate(results_data):
        # Fill in available results, leaving zeros for missing matches
        for j, result in enumerate(team_data["results"]):
            if j < last_n:
                data[i, j] = result
        
        yticks.append(team_data["team"])
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    
    # Add match labels
    if show_match_info:
        for i, team_data in enumerate(results_data):
            for j, (result, opponent, score, venue) in enumerate(zip(
                team_data.get("results", []), 
                team_data.get("opponents", []), 
                team_data.get("scores", []),
                team_data.get("venues", [])
            )):
                if j < last_n:
                    venue_marker = "H" if venue == "Home" else "A"
                    text = f"{opponent} ({venue_marker})\n{score}"
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    
    match_numbers = [f"Match {i+1}" for i in range(last_n)]
    ax.set_xticks(np.arange(last_n))
    ax.set_xticklabels(match_numbers)
    
    # Use custom title or default
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f"Team Form - Last {last_n} Matches", fontsize=16)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#99ff99', label='Win'),
        Patch(facecolor='#ffff99', label='Draw'),
        Patch(facecolor='#ff9999', label='Loss')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved form heatmap to {out}")
    
    return fig

###############################################################################
# Advanced Football Charts                                                   #
###############################################################################

def radar_chart(
    stats: Dict[str, float],
    team_name: str,
    categories: Optional[List[str]] = None,
    comparison_stats: Optional[Dict[str, float]] = None,
    comparison_name: Optional[str] = "League Average",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    out: Optional[Path] = None,
    dpi: int = 300,
    color: Optional[str] = None,
    comparison_color: str = "#999999"
) -> plt.Figure:
    """Create a radar chart for team statistics.
    
    Args:
        stats: Dictionary of statistics {metric: value}
        team_name: Name of the team
        categories: Optional list of categories to include (defaults to all in stats)
        comparison_stats: Optional stats for comparison (e.g., league average)
        comparison_name: Name for the comparison entity
        title: Optional chart title
        figsize: Figure size
        out: Optional output file path
        dpi: Resolution for saved image
        color: Optional team color
        comparison_color: Color for comparison stats
        
    Returns:
        Matplotlib figure
    """
    # If categories not specified, use all keys from stats
    if categories is None:
        categories = list(stats.keys())
    
    # Extract values for each category
    values = [stats.get(cat, 0) for cat in categories]
    
    # Set team color
    if color is None:
        color = get_team_color(team_name)
    
    # Handle comparison data if provided
    comp_values = None
    if comparison_stats:
        comp_values = [comparison_stats.get(cat, 0) for cat in categories]
    
    # Repeat first value to close the polygon
    categories = categories + [categories[0]]
    values = values + [values[0]]
    if comp_values:
        comp_values = comp_values + [comp_values[0]]
    
    # Calculate angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot team values
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=team_name)
    ax.fill(angles, values, color=color, alpha=0.25)
    
    # Plot comparison values if provided
    if comp_values:
        ax.plot(angles, comp_values, 'o-', linewidth=2, color=comparison_color, label=comparison_name)
        ax.fill(angles, comp_values, color=comparison_color, alpha=0.1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=12)
    
    # Remove radial labels and set grid style
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title if provided
    if title:
        plt.title(title, size=20, pad=20)
    else:
        plt.title(f"{team_name} Performance Radar", size=20, pad=20)
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved radar chart to {out}")
    
    return fig


def position_chart(
    team: str,
    season: int,
    out: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    color: Optional[str] = None,
    include_points: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """Create a chart showing a team's league position over time.
    
    Args:
        team: Team name
        season: Season year
        out: Optional output file path
        figsize: Figure size
        title: Optional chart title
        color: Optional team color
        include_points: Whether to include points on secondary axis
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib figure
    """
    from analytics import get_league_position_history
    
    # Get position history data
    df = get_league_position_history(team, season)
    
    if df.empty:
        logger.warning(f"No position history data found for {team} in season {season}")
        return plt.figure()
    
    # Set team color
    if color is None:
        color = get_team_color(team)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot position (inverted y-axis so that 1st is at the top)
    ax1.plot(df["matchday"], df["position"], marker='o', linestyle='-', color=color, linewidth=2)
    ax1.set_ylabel("League Position", fontsize=12)
    ax1.invert_yaxis()  # Invert so that 1st position is at the top
    
    # Set y-axis ticks to integers only
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Add secondary axis for points if requested
    if include_points:
        ax2 = ax1.twinx()
        ax2.plot(df["matchday"], df["points"], marker='s', linestyle='--', color='gray', linewidth=1.5)
        ax2.set_ylabel("Points", fontsize=12)
        ax2.grid(False)
    
    # Set x-axis label
    ax1.set_xlabel("Matchday", fontsize=12)
    
    # Set x-axis ticks to integers only
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Set title
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title(f"{team} League Position - {season}/{season+1} Season", fontsize=16, pad=20)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if points are included
    if include_points:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, ["Position", "Points"], loc='best')
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved position chart to {out}")
    
    return fig


def head_to_head_comparison(
    team1: str,
    team2: str,
    metrics: Dict[str, Dict[str, float]],
    out: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a bar chart comparing two teams across multiple metrics.
    
    Args:
        team1: First team name
        team2: Second team name
        metrics: Dictionary of metrics to compare {metric_name: {team1: value1, team2: value2}}
        out: Optional output file path
        figsize: Figure size
        title: Optional chart title
        colors: Optional list of two colors for the teams
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib figure
    """
    if not metrics:
        logger.warning("No metrics provided for head-to-head comparison")
        return plt.figure()
    
    # Set team colors
    if colors is None:
        colors = [get_team_color(team1), get_team_color(team2)]
    
    # Extract metric names and values
    metric_names = list(metrics.keys())
    team1_values = [metrics[m].get(team1, 0) for m in metric_names]
    team2_values = [metrics[m].get(team2, 0) for m in metric_names]
    
    # Create positions for bars
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    ax.bar(x - width/2, team1_values, width, label=team1, color=colors[0])
    ax.bar(x + width/2, team2_values, width, label=team2, color=colors[1])
    
    # Customize appearance
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    else:
        ax.set_title(f"{team1} vs {team2} Comparison", fontsize=16, pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(team1_values):
        ax.text(i - width/2, v + (v * 0.02), f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(team2_values):
        ax.text(i + width/2, v + (v * 0.02), f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved head-to-head comparison chart to {out}")
    
    return fig


def shot_map(
    team: str,
    season: int,
    out: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    color: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a shot map visualization (simplified version without actual shot locations).
    
    This is a placeholder - in a real implementation, you would use actual shot location data.
    
    Args:
        team: Team name
        season: Season year
        out: Optional output file path
        figsize: Figure size
        title: Optional chart title
        color: Optional team color
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib figure
    """
    from database_utils import execute_query
    
    # Get shot data (this would normally include x,y coordinates)
    query = """
    SELECT gf, ga, date, opponent
    FROM matches
    WHERE team = :team AND season = :season
    ORDER BY date
    """
    
    df = execute_query(query, {"team": team, "season": season})
    
    if df.empty:
        logger.warning(f"No match data found for {team} in season {season}")
        return plt.figure()
    
    # Set team color
    if color is None:
        color = get_team_color(team)
    
    # Create figure with football pitch
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw a simplified football pitch
    # Pitch outline
    pitch_length = 120
    pitch_width = 80
    pitch = plt.Rectangle((0, 0), pitch_length, pitch_width, fill=False, color='black')
    ax.add_patch(pitch)
    
    # Center line
    plt.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color='black')
    
    # Center circle
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15, fill=False, color='black')
    ax.add_patch(center_circle)
    
    # Penalty areas
    penalty_area_left = plt.Rectangle((0, pitch_width/2 - 20.16), 16.5, 40.32, fill=False, color='black')
    penalty_area_right = plt.Rectangle((pitch_length - 16.5, pitch_width/2 - 20.16), 16.5, 40.32, fill=False, color='black')
    ax.add_patch(penalty_area_left)
    ax.add_patch(penalty_area_right)
    
    # Goal boxes
    goal_box_left = plt.Rectangle((0, pitch_width/2 - 9.16), 5.5, 18.32, fill=False, color='black')
    goal_box_right = plt.Rectangle((pitch_length - 5.5, pitch_width/2 - 9.16), 5.5, 18.32, fill=False, color='black')
    ax.add_patch(goal_box_left)
    ax.add_patch(goal_box_right)
    
    # Goals
    plt.plot([0, 0], [pitch_width/2 - 3.66, pitch_width/2 + 3.66], color='black', linewidth=3)
    plt.plot([pitch_length, pitch_length], [pitch_width/2 - 3.66, pitch_width/2 + 3.66], color='black', linewidth=3)
    
    # Since we don't have actual shot location data, we'll show a placeholder message
    ax.text(pitch_length/2, pitch_width/2, 
            "Shot map placeholder - actual implementation\nwould use x,y coordinates of shots",
            ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add team goals as summary text
    total_goals = df["gf"].sum()
    goals_per_game = total_goals / len(df)
    
    summary_text = (
        f"{team} - {season}/{season+1} Season\n"
        f"Total Goals: {total_goals}\n"
        f"Goals per Game: {goals_per_game:.2f}"
    )
    
    ax.text(5, 5, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Set title
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title(f"{team} Shot Map - {season}/{season+1} Season", fontsize=16, pad=20)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved shot map to {out}")
    
    return fig


def dashboard(
    team: str,
    season: int,
    out: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
    title: Optional[str] = None,
    color: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a dashboard with multiple visualizations for a team.
    
    Args:
        team: Team name
        season: Season year
        out: Optional output file path
        figsize: Figure size
        title: Optional dashboard title
        color: Optional team color
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib figure
    """
    from analytics import calculate_team_advanced_metrics, team_form_analysis
    from database_utils import execute_query
    
    # Set team color
    if color is None:
        color = get_team_color(team)
    
    # Get team data
    metrics = calculate_team_advanced_metrics(team, season)
    form = team_form_analysis(team, matches=5)
    
    if not metrics:
        logger.warning(f"No metrics found for {team} in season {season}")
        return plt.figure()
    
    # Create figure with grid
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(f"{team} Performance Dashboard - {season}/{season+1} Season", fontsize=20)
    
    # Form chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if not form.empty:
        form_string = form["form_string"].iloc[0] if "form_string" in form.columns else ""
        form_data = []
        for result in form_string:
            if result == "W":
                form_data.append(3)
            elif result == "D":
                form_data.append(1)
            else:
                form_data.append(0)
        
        cmap = colors.ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
        bounds = [-0.5, 0.5, 1.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        form_array = np.array(form_data).reshape(1, -1)
        ax1.imshow(form_array, cmap=cmap, norm=norm, aspect='auto')
        ax1.set_yticks([])
        ax1.set_xticks(range(len(form_data)))
        ax1.set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
        ax1.set_title("Recent Form")
    else:
        ax1.text(0.5, 0.5, "No form data available", ha='center', va='center')
        ax1.axis('off')
    
    # Goals chart (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    goals_data = {
        "For": metrics.get("goals_for", 0),
        "Against": metrics.get("goals_against", 0)
    }
    ax2.bar(goals_data.keys(), goals_data.values(), color=[color, 'gray'])
    for i, v in enumerate(goals_data.values()):
        ax2.text(i, v + 0.5, str(v), ha='center')
    ax2.set_title("Goals")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Points chart (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if "home_points" in metrics and "away_points" in metrics:
        points_data = {
            "Home": metrics["home_points"],
            "Away": metrics["away_points"],
            "Total": metrics["points"]
        """Enhanced visualization module with more chart types and better customization."""
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import cm, colors
import squarify

# Configure logging
logger = logging.getLogger("fbref_toolkit.visualization")

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palettes
TEAM_COLORS = {
    "manchester united": "#DA291C",
    "man city": "#6CABDD",
    "liverpool": "#C8102E",
    "chelsea": "#034694",
    "arsenal": "#EF0107",
    "tottenham hotspur": "#132257",
    "west ham united": "#7A263A",
    "leicester city": "#003090",
    "aston villa": "#95BFE5",
    "everton": "#003399",
    "newcastle united": "#241F20",
    "wolverhampton": "#FDB913",
    "brighton & hove albion": "#0057B8",
    "crystal palace": "#1B458F",
    "southampton": "#D71920",
    "real madrid": "#FFFFFF",
    "barcelona": "#A50044",
    "atletico madrid": "#CB3524",
    "bayern munich": "#DC052D",
    "dortmund": "#FDE100",
    "psg": "#004170",
    "juventus": "#000000",
    "inter": "#0057B8",
    "milan": "#FB090B",
}

def get_team_color(team: str, default_color: str = "#1f77b4") -> str:
    """Get the official color for a team or return a default color.
    
    Args:
        team: Team name
        default_color: Default color to use if team color not found
        
    Returns:
        HEX color code
    """
    return TEAM_COLORS.get(team.lower(), default_color)

def setup_figure(
    figsize: Tuple[int, int] = (10, 6),
    style: str = 'seaborn-v0_8-whitegrid',
    dpi: int = 100,
    tight_layout: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a matplotlib figure with custom styling.
    
    Args:
        figsize: Figure size as (width, height)
        style: Matplotlib style to use
        dpi: Dots per inch
        tight_layout: Whether to use tight layout
        
    Returns:
        Tuple of (figure, axes)
    """
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if tight_layout:
        plt.tight_layout()
        
    return fig, ax

###############################################################################
# Basic Charts                                                               #
###############################################################################

def treemap(
    df: pd.DataFrame, 
    value_col: str, 
    label_col: str, 
    title: str, 
    out: Optional[Path] = None,
    cmap: str = "viridis", 
    figsize: Tuple[int, int] = (12, 8),
    custom_colors: Optional[Dict[str, str]] = None,
    show_values: bool = True,
    value_format: str = ".1f",
    dpi: int = 300
) -> plt.Figure:
    """Create a treemap visualization of data.
    
    Args:
        df: DataFrame with data
        value_col: Column name for values (box sizes)
        label_col: Column name for labels
        title: Chart title
        out: Optional output file path
        cmap: Colormap name
        figsize: Figure size as (width, height) tuple
        custom_colors: Optional dict mapping labels to colors
        show_values: Whether to show values in boxes
        value_format: Format string for values
        dpi: Resolution for saved image
        
    Returns:
        Matplotlib figure
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for treemap")
        return plt.figure()
        
    sizes = df[value_col].astype(float).tolist()
    
    if show_values:
        labels = [f"{l}\n{v:{value_format}}" for l, v in zip(df[label_col], df[value_col])]
    else:
        labels = df[label_col].tolist()

    # Handle custom colors or use colormap
    if custom_colors:
        # Try to use custom colors for each label
        clrs = [custom_colors.get(label, 'gray') for label in df[label_col]]
    else:
        norm = colors.Normalize(vmin=min(sizes), vmax=max(sizes))
        color_map = cm.get_cmap(cmap)
        clrs = [color_map(norm(s)) for s in sizes]

    fig, ax = plt.subplots(figsize=figsize)
    squarify.plot(sizes=sizes, label=labels, color=clrs, ax=ax,
                 text_kwargs=dict(fontsize=12, weight="bold", color="white"))
    
    plt.suptitle(title, fontsize=16, y=0.95)
    ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if out:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved treemap to {out}")
    
    return fig


def bar_plot(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    title: str, 
    out: Optional[Path] = None,
    color: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[int, int] = (12, 8),
    sort: bool = True, 
    limit: Optional[int] = None,
    horizontal: bool = False,
    show_values: bool = True,
    value_format: str = ".1f",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    grid: bool = True,
    dpi: int = 300,
    use_team_colors: bool = False
) -> plt.Figure:
    """Create a bar plot visualization.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        out: Optional output file path
        color: Bar color or list of colors
        figsize: Figure size as (width, height) tuple
        sort: Whether to sort by y values
        limit: Optional limit on number of bars to show
        horizontal: If True, create a horizontal bar chart
        show_values: Whether to show values on bars
        value_format: Format string for values
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        grid: Whether to show grid
        dpi: Resolution for saved image
        use_team_colors: Whether to use team colors (if x_col contains team names)
        
    Returns:
        Matplotlib figure
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for bar plot")
        return plt.figure()
    
    # Sort data if requested
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(y_col, ascending=False)
    
    # Limit number of bars if requested
    if limit and len(plot_df) > limit:
        plot_df = plot_df.head(limit)
    
    # Create figure
    fig, ax = plt.subplots(