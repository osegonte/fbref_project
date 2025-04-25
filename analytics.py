"""Advanced football analytics module with improved statistical calculations."""
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from database_utils import get_session, execute_query

# Configure logging
logger = logging.getLogger("fbref_toolkit.analytics")

###############################################################################
# Form Analysis                                                               #
###############################################################################

def team_form_analysis(
    team: str, 
    matches: int = 5, 
    all_comps: bool = False,
    season: Optional[int] = None
) -> pd.DataFrame:
    """Calculate recent form statistics for a team with enhanced metrics.
    
    Args:
        team: Team name
        matches: Number of recent matches to analyze
        all_comps: If True, include all competitions (not just league matches)
        season: Optional season filter
        
    Returns:
        DataFrame with form statistics
    """
    # Build query with filters
    params = {"team": team, "limit": matches}
    
    query = """
    SELECT * FROM matches 
    WHERE team = :team 
    """
    
    if not all_comps:
        # Only include league matches (filter out cups)
        query += " AND comp NOT LIKE '%Cup%' AND comp NOT LIKE '%Champions%' AND comp NOT LIKE '%Europa%'"
    
    if season:
        query += " AND season = :season"
        params["season"] = season
    
    query += " ORDER BY date DESC LIMIT :limit"
    
    df = execute_query(query, params)
    
    if df.empty:
        logger.warning(f"No match data found for {team}")
        return pd.DataFrame()
    
    # Calculate form metrics
    form_metrics = {
        "team": team,
        "period": f"Last {matches} matches",
        "wins": len(df[df["result"] == "W"]),
        "draws": len(df[df["result"] == "D"]),
        "losses": len(df[df["result"] == "L"]),
        "points": df["points"].sum(),
        "points_per_game": round(df["points"].sum() / len(df), 2),
        "goals_for": df["gf"].sum(),
        "goals_against": df["ga"].sum(),
        "goal_diff": df["gf"].sum() - df["ga"].sum(),
        "clean_sheets": len(df[df["ga"] == 0]),
        "failed_to_score": len(df[df["gf"] == 0]),
        "avg_corners_for": round(df["corner_for"].mean(), 1) if "corner_for" in df.columns else None,
        "avg_shots": round(df["sh"].mean(), 1) if "sh" in df.columns else None,
        "avg_shots_on_target": round(df["sot"].mean(), 1) if "sot" in df.columns else None,
        "avg_possession": round(df["poss"].mean(), 1) if "poss" in df.columns else None,
        "shot_conversion": round((df["gf"].sum() / df["sh"].sum()) * 100, 1) if "sh" in df.columns and df["sh"].sum() > 0 else None,
        "shot_accuracy": round((df["sot"].sum() / df["sh"].sum()) * 100, 1) if "sh" in df.columns and "sot" in df.columns and df["sh"].sum() > 0 else None,
        "defensive_solidity": round(df["ga"].sum() / len(df), 2),
        "attack_rating": round(df["gf"].sum() / len(df), 2),
    }
    
    # Calculate expected goals metrics if available
    if "xg" in df.columns and not df["xg"].isna().all():
        form_metrics["total_xg"] = round(df["xg"].sum(), 2)
        form_metrics["avg_xg"] = round(df["xg"].mean(), 2)
        form_metrics["xg_outperformance"] = round(df["gf"].sum() - df["xg"].sum(), 2)
    
    if "xga" in df.columns and not df["xga"].isna().all():
        form_metrics["total_xga"] = round(df["xga"].sum(), 2)
        form_metrics["avg_xga"] = round(df["xga"].mean(), 2)
        form_metrics["xga_outperformance"] = round(df["xga"].sum() - df["ga"].sum(), 2)
    
    # Add home/away split
    home_matches = df[df["is_home"] == True]
    away_matches = df[df["is_home"] == False]
    
    if not home_matches.empty:
        form_metrics["home_points"] = home_matches["points"].sum()
        form_metrics["home_goals_for"] = home_matches["gf"].sum()
        form_metrics["home_goals_against"] = home_matches["ga"].sum()
    
    if not away_matches.empty:
        form_metrics["away_points"] = away_matches["points"].sum()
        form_metrics["away_goals_for"] = away_matches["gf"].sum()
        form_metrics["away_goals_against"] = away_matches["ga"].sum()
        
    # Add form string (W, D, L for last matches from oldest to newest)
    form_string = "".join(df["result"].iloc[::-1])
    form_metrics["form_string"] = form_string
    
    # Add form momentum (trending up, stable, or down)
    if len(df) >= 3:
        recent_points = df.head(len(df) // 2)["points"].sum()
        earlier_points = df.tail(len(df) // 2)["points"].sum()
        momentum = recent_points - earlier_points
        
        if momentum > 0:
            form_metrics["momentum"] = "Improving"
        elif momentum < 0:
            form_metrics["momentum"] = "Declining"
        else:
            form_metrics["momentum"] = "Stable"
            
        form_metrics["momentum_value"] = momentum
    
    # Calculate statistical significance of form
    if "xg" in df.columns and not df["xg"].isna().all() and "xga" in df.columns and not df["xga"].isna().all():
        # Calculate if the team is significantly overperforming xG
        t_stat, p_value = stats.ttest_rel(df["gf"], df["xg"])
        form_metrics["xg_p_value"] = round(p_value, 3)
        form_metrics["significantly_overperforming_xg"] = p_value < 0.05 and df["gf"].mean() > df["xg"].mean()
        form_metrics["significantly_underperforming_xg"] = p_value < 0.05 and df["gf"].mean() < df["xg"].mean()
    
    return pd.DataFrame([form_metrics])


def get_league_position_history(team: str, season: int) -> pd.DataFrame:
    """Get a team's league position history over the course of a season.
    
    Args:
        team: Team name
        season: Season year
        
    Returns:
        DataFrame with position history by matchday
    """
    # First, get the league the team is in
    league_query = """
    SELECT DISTINCT comp
    FROM matches
    WHERE team = :team AND season = :season AND comp NOT LIKE '%Cup%'
    LIMIT 1
    """
    
    league_df = execute_query(league_query, {"team": team, "season": season})
    
    if league_df.empty:
        logger.warning(f"No league data found for {team} in season {season}")
        return pd.DataFrame()
        
    league = league_df.iloc[0]["comp"]
    
    # Now get all teams in this league
    teams_query = """
    SELECT DISTINCT team
    FROM matches
    WHERE season = :season AND comp = :league
    ORDER BY team
    """
    
    teams_df = execute_query(teams_query, {"season": season, "league": league})
    teams = teams_df["team"].tolist()
    
    if not teams:
        logger.warning(f"No teams found for {league} in season {season}")
        return pd.DataFrame()
    
    # Get all matches in the league for this season
    matches_query = """
    SELECT date, team, opponent, result, gf, ga, points
    FROM matches
    WHERE season = :season AND comp = :league
    ORDER BY date
    """
    
    matches_df = execute_query(matches_query, {"season": season, "league": league})
    
    if matches_df.empty:
        logger.warning(f"No matches found for {league} in season {season}")
        return pd.DataFrame()
    
    # Group matches by date to identify matchdays
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    
    # Initialize a dictionary to track points and position for each team
    team_stats = {t: {"points": 0, "position": 0, "history": []} for t in teams}
    
    # Process matches chronologically to build position history
    matchdays = []
    current_matchday = 1
    
    # Group matches into matchdays based on date proximity
    matches_df = matches_df.sort_values("date")
    dates = matches_df["date"].unique()
    
    for date in dates:
        # Get matches on this date
        day_matches = matches_df[matches_df["date"] == date]
        
        # Update team points
        for _, match in day_matches.iterrows():
            team = match["team"]
            if team in team_stats:
                team_stats[team]["points"] += match["points"]
        
        # Check if this is a new matchday (enough teams have played)
        matches_played = sum(1 for t in teams if any(matches_df[(matches_df["team"] == t) & (matches_df["date"] <= date)].index))
        
        if matches_played >= len(teams) * 0.75:  # At least 75% of teams have played
            matchdays.append(date)
            
            # Calculate positions for this matchday
            positions = sorted([(t, team_stats[t]["points"]) for t in teams], key=lambda x: (-x[1], x[0]))
            
            for pos, (t, _) in enumerate(positions, 1):
                team_stats[t]["position"] = pos
                
            # Record position history for each team
            for t in teams:
                team_stats[t]["history"].append({
                    "matchday": current_matchday,
                    "date": date,
                    "position": team_stats[t]["position"],
                    "points": team_stats[t]["points"]
                })
                
            current_matchday += 1
    
    # Extract position history for the requested team
    if team in team_stats:
        history = team_stats[team]["history"]
        return pd.DataFrame(history)
    else:
        logger.warning(f"Team {team} not found in position history")
        return pd.DataFrame()


def calculate_head_to_head_stats(team1: str, team2: str, limit: int = 10) -> Dict[str, Any]:
    """Calculate head-to-head statistics between two teams with enhanced metrics.
    
    Args:
        team1: First team name
        team2: Second team name
        limit: Maximum number of matches to consider
    
    Returns:
        Dictionary with head-to-head statistics
    """
    # Get matches where these teams played each other
    query = """
    SELECT * FROM matches 
    WHERE (team = :team1 AND opponent = :team2) OR (team = :team2 AND opponent = :team1)
    ORDER BY date DESC
    LIMIT :limit
    """
    
    df = execute_query(query, {"team1": team1, "team2": team2, "limit": limit})
    
    if df.empty:
        return {"error": f"No head-to-head data found for {team1} vs {team2}"}
    
    # Initialize stats
    stats = {
        "team1": team1,
        "team2": team2,
        "total_matches": len(df),
        "team1_wins": 0,
        "team2_wins": 0,
        "draws": 0,
        "team1_goals": 0,
        "team2_goals": 0,
        "team1_xg": 0,
        "team2_xg": 0,
        "matches": []
    }
    
    # Calculate stats
    for _, match in df.iterrows():
        # Record match details
        match_info = {
            "date": match["date"],
            "score": f"{match['gf']}-{match['ga']}",
            "venue": match["venue"],
            "competition": match["comp"] if "comp" in match else "Unknown"
        }
        
        # Add xG if available
        if "xg" in match and not pd.isna(match["xg"]):
            match_info["xg"] = match["xg"]
        if "xga" in match and not pd.isna(match["xga"]):
            match_info["xga"] = match["xga"]
        
        # Determine who was home/away and who won
        if match["team"] == team1:
            match_info["home_team"] = team1 if match["venue"] == "Home" else team2
            match_info["away_team"] = team2 if match["venue"] == "Home" else team1
            
            if match["result"] == "W":
                stats["team1_wins"] += 1
            elif match["result"] == "L":
                stats["team2_wins"] += 1
            else:
                stats["draws"] += 1
                
            stats["team1_goals"] += match["gf"]
            stats["team2_goals"] += match["ga"]
            
            # Add xG if available
            if "xg" in match and not pd.isna(match["xg"]):
                stats["team1_xg"] += match["xg"]
            if "xga" in match and not pd.isna(match["xga"]):
                stats["team2_xg"] += match["xga"]
            
        else:  # match["team"] == team2
            match_info["home_team"] = team2 if match["venue"] == "Home" else team1
            match_info["away_team"] = team1 if match["venue"] == "Home" else team2
            
            if match["result"] == "W":
                stats["team2_wins"] += 1
            elif match["result"] == "L":
                stats["team1_wins"] += 1
            else:
                stats["draws"] += 1
                
            stats["team2_goals"] += match["gf"]
            stats["team1_goals"] += match["ga"]
            
            # Add xG if available
            if "xg" in match and not pd.isna(match["xg"]):
                stats["team2_xg"] += match["xg"]
            if "xga" in match and not pd.isna(match["xga"]):
                stats["team1_xg"] += match["xga"]
        
        stats["matches"].append(match_info)
    
    # Calculate additional stats
    stats["avg_goals_per_match"] = round((stats["team1_goals"] + stats["team2_goals"]) / stats["total_matches"], 2)
    stats["team1_win_percentage"] = round((stats["team1_wins"] / stats["total_matches"]) * 100, 1)
    stats["team2_win_percentage"] = round((stats["team2_wins"] / stats["total_matches"]) * 100, 1)
    stats["draw_percentage"] = round((stats["draws"] / stats["total_matches"]) * 100, 1)
    
    # Home vs Away analysis
    home_matches_team1 = df[(df["team"] == team1) & (df["venue"] == "Home")]
    away_matches_team1 = df[(df["team"] == team1) & (df["venue"] == "Away")]
    home_matches_team2 = df[(df["team"] == team2) & (df["venue"] == "Home")]
    away_matches_team2 = df[(df["team"] == team2) & (df["venue"] == "Away")]
    
    # Team 1 home record
    stats["team1_home_matches"] = len(home_matches_team1)
    if stats["team1_home_matches"] > 0:
        stats["team1_home_wins"] = len(home_matches_team1[home_matches_team1["result"] == "W"])
        stats["team1_home_draws"] = len(home_matches_team1[home_matches_team1["result"] == "D"])
        stats["team1_home_losses"] = len(home_matches_team1[home_matches_team1["result"] == "L"])
        stats["team1_home_goals_for"] = home_matches_team1["gf"].sum()
        stats["team1_home_goals_against"] = home_matches_team1["ga"].sum()
    
    # Team 2 home record
    stats["team2_home_matches"] = len(home_matches_team2)
    if stats["team2_home_matches"] > 0:
        stats["team2_home_wins"] = len(home_matches_team2[home_matches_team2["result"] == "W"])
        stats["team2_home_draws"] = len(home_matches_team2[home_matches_team2["result"] == "D"])
        stats["team2_home_losses"] = len(home_matches_team2[home_matches_team2["result"] == "L"])
        stats["team2_home_goals_for"] = home_matches_team2["gf"].sum()
        stats["team2_home_goals_against"] = home_matches_team2["ga"].sum()
    
    # Recent form comparison (last 3 meetings)
    if len(df) >= 3:
        recent = df.head(3)
        
        team1_recent_points = sum(3 if row["team"] == team1 and row["result"] == "W" or
                                  row["team"] == team2 and row["result"] == "L" else
                                  1 if row["result"] == "D" else 0
                                  for _, row in recent.iterrows())
                                  
        team2_recent_points = sum(3 if row["team"] == team2 and row["result"] == "W" or
                                  row["team"] == team1 and row["result"] == "L" else
                                  1 if row["result"] == "D" else 0
                                  for _, row in recent.iterrows())
                                  
        stats["team1_recent_points"] = team1_recent_points
        stats["team2_recent_points"] = team2_recent_points
        
        if team1_recent_points > team2_recent_points:
            stats["recent_form_advantage"] = team1
        elif team2_recent_points > team1_recent_points:
            stats["recent_form_advantage"] = team2
        else:
            stats["recent_form_advantage"] = "Equal"
    
    return stats

###############################################################################
# Advanced Team Metrics                                                      #
###############################################################################

def calculate_team_advanced_metrics(team: str, season: int) -> Dict[str, Any]:
    """Calculate advanced metrics for a team in a specific season.
    
    Args:
        team: Team name
        season: Season year
    
    Returns:
        Dictionary with advanced metrics
    """
    # Get all matches for this team in the season
    query = """
    SELECT * FROM matches 
    WHERE team = :team AND season = :season
    ORDER BY date
    """
    
    df = execute_query(query, {"team": team, "season": season})
    
    if df.empty:
        logger.warning(f"No matches found for {team} in season {season}")
        return {}
    
    # Calculate basic stats
    total_matches = len(df)
    wins = len(df[df["result"] == "W"])
    draws = len(df[df["result"] == "D"])
    losses = len(df[df["result"] == "L"])
    points = df["points"].sum()
    goals_for = df["gf"].sum()
    goals_against = df["ga"].sum()
    
    # Calculate advanced metrics
    metrics = {
        "team": team,
        "season": season,
        "total_matches": total_matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "points_per_game": round(points / total_matches, 2),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "goals_per_game": round(goals_for / total_matches, 2),
        "goals_against_per_game": round(goals_against / total_matches, 2),
        "clean_sheets": len(df[df["ga"] == 0]),
        "clean_sheet_percentage": round(len(df[df["ga"] == 0]) / total_matches * 100, 1),
        "failed_to_score": len(df[df["gf"] == 0]),
        "win_percentage": round(wins / total_matches * 100, 1),
        "loss_percentage": round(losses / total_matches * 100, 1),
        "draw_percentage": round(draws / total_matches * 100, 1),
    }
    
    # Home vs Away split
    home_df = df[df["venue"] == "Home"]
    away_df = df[df["venue"] == "Away"]
    
    if not home_df.empty:
        home_matches = len(home_df)
        metrics["home_matches"] = home_matches
        metrics["home_wins"] = len(home_df[home_df["result"] == "W"])
        metrics["home_draws"] = len(home_df[home_df["result"] == "D"])
        metrics["home_losses"] = len(home_df[home_df["result"] == "L"])
        metrics["home_points"] = home_df["points"].sum()
        metrics["home_points_per_game"] = round(home_df["points"].sum() / home_matches, 2)
        metrics["home_goals_for"] = home_df["gf"].sum()
        metrics["home_goals_against"] = home_df["ga"].sum()
        metrics["home_clean_sheets"] = len(home_df[home_df["ga"] == 0])
    
    if not away_df.empty:
        away_matches = len(away_df)
        metrics["away_matches"] = away_matches
        metrics["away_wins"] = len(away_df[away_df["result"] == "W"])
        metrics["away_draws"] = len(away_df[away_df["result"] == "D"])
        metrics["away_losses"] = len(away_df[away_df["result"] == "L"])
        metrics["away_points"] = away_df["points"].sum()
        metrics["away_points_per_game"] = round(away_df["points"].sum() / away_matches, 2)
        metrics["away_goals_for"] = away_df["gf"].sum()
        metrics["away_goals_against"] = away_df["ga"].sum()
        metrics["away_clean_sheets"] = len(away_df[away_df["ga"] == 0])
    
    # Add expected goals metrics if available
    if "xg" in df.columns and not df["xg"].isna().all():
        total_xg = df["xg"].sum()
        metrics["total_xg"] = round(total_xg, 2)
        metrics["xg_per_game"] = round(total_xg / total_matches, 2)
        metrics["xg_difference"] = round(goals_for - total_xg, 2)
        metrics["xg_efficiency"] = round(goals_for / total_xg * 100, 1) if total_xg > 0 else 0
    
    if "xga" in df.columns and not df["xga"].isna().all():
        total_xga = df["xga"].sum()
        metrics["total_xga"] = round(total_xga, 2)
        metrics["xga_per_game"] = round(total_xga / total_matches, 2)
        metrics["xga_difference"] = round(goals_against - total_xga, 2)
        metrics["defensive_efficiency"] = round(goals_against / total_xga * 100, 1) if total_xga > 0 else 0
    
    # Add shot metrics if available
    if "sh" in df.columns and not df["sh"].isna().all():
        total_shots = df["sh"].sum()
        metrics["total_shots"] = total_shots
        metrics["shots_per_game"] = round(total_shots / total_matches, 1)
        metrics["shot_conversion"] = round(goals_for / total_shots * 100, 1) if total_shots > 0 else 0
    
    if "sot" in df.columns and not df["sot"].isna().all() and "sh" in df.columns:
        total_shots_on_target = df["sot"].sum()
        metrics["total_shots_on_target"] = total_shots_on_target
        metrics["shots_on_target_per_game"] = round(total_shots_on_target / total_matches, 1)
        metrics["shooting_accuracy"] = round(total_shots_on_target / total_shots * 100, 1) if total_shots > 0 else 0
        metrics["shot_on_target_conversion"] = round(goals_for / total_shots_on_target * 100, 1) if total_shots_on_target > 0 else 0
    
    # Add possession metrics if available
    if "poss" in df.columns and not df["poss"].isna().all():
        metrics["avg_possession"] = round(df["poss"].mean(), 1)
        
        # Possession efficiency (goals per % possession)
        if metrics["avg_possession"] > 0:
            metrics["possession_efficiency"] = round(goals_for / (metrics["avg_possession"] * total_matches) * 100, 2)
    
    # First/second half analysis
    if "1st_half_goals_for" in df.columns and "2nd_half_goals_for" in df.columns:
        metrics["1st_half_goals_for"] = df["1st_half_goals_for"].sum()
        metrics["2nd_half_goals_for"] = df["2nd_half_goals_for"].sum()
        metrics["1st_half_goals_against"] = df["1st_half_goals_against"].sum()
        metrics["2nd_half_goals_against"] = df["2nd_half_goals_against"].sum()
        
        # Percentage of goals scored in each half
        if goals_for > 0:
            metrics["1st_half_goals_percentage"] = round(metrics["1st_half_goals_for"] / goals_for * 100, 1)
            metrics["2nd_half_goals_percentage"] = round(metrics["2nd_half_goals_for"] / goals_for * 100, 1)
    
    # Scoring and conceding patterns
    scored_first = df[(df["result"] == "W") & (df["gf"] > 0)].count()["result"]
    conceded_first = df[(df["result"] == "L") & (df["ga"] > 0)].count()["result"]
    
    # Win/loss after scoring/conceding first (if we can extract this data)
    metrics["scored_first_count"] = scored_first
    metrics["conceded_first_count"] = conceded_first
    
    # Form consistency (standard deviation of points in 5-game windows)
    if total_matches >= 5:
        window_size = 5
        rolling_points = []
        
        for i in range(total_matches - window_size + 1):
            window = df.iloc[i:i+window_size]
            rolling_points.append(window["points"].sum())
        
        metrics["form_consistency"] = round(np.std(rolling_points), 2)
        metrics["form_consistency_rating"] = "High" if metrics["form_consistency"] < 2 else "Medium" if metrics["form_consistency"] < 4 else "Low"
    
    return metrics

###############################################################################
# Player Analysis                                                            #
###############################################################################

def analyze_player(player: str, team: str, season: int) -> Dict[str, Any]:
    """Analyze player performance for a specific season.
    
    Args:
        player: Player name
        team: Team name
        season: Season year
    
    Returns:
        Dictionary with player analysis
    """
    # Get player data
    query = """
    SELECT * FROM players 
    WHERE player = :player AND team = :team AND season = :season
    """
    
    df = execute_query(query, {"player": player, "team": team, "season": season})
    
    if df.empty:
        logger.warning(f"No data found for player {player}, team {team}, season {season}")
        return {}
    
    # Initialize analysis dictionary
    analysis = {
        "player": player,
        "team": team,
        "season": season,
        "position": df["pos"].iloc[0] if "pos" in df.columns else "Unknown",
        "age": df["age"].iloc[0] if "age" in df.columns else None,
        "nationality": df["nation"].iloc[0] if "nation" in df.columns else "Unknown",
    }
    
    # Extract standard stats if available
    standard_stats = df[df["stats_type"] == "Standard Stats"]
    if not standard_stats.empty:
        analysis["appearances"] = standard_stats["games"].iloc[0] if "games" in standard_stats.columns else None
        analysis["starts"] = standard_stats["games_starts"].iloc[0] if "games_starts" in standard_stats.columns else None
        analysis["minutes"] = standard_stats["minutes"].iloc[0] if "minutes" in standard_stats.columns else None
        analysis["goals"] = standard_stats["goals"].iloc[0] if "goals" in standard_stats.columns else None
        analysis["assists"] = standard_stats["assists"].iloc[0] if "assists" in standard_stats.columns else None
        analysis["yellow_cards"] = standard_stats["cards_yellow"].iloc[0] if "cards_yellow" in standard_stats.columns else None
        analysis["red_cards"] = standard_stats["cards_red"].iloc[0] if "cards_red" in standard_stats.columns else None
    
    # Extract shooting stats if available
    shooting_stats = df[df["stats_type"] == "Shooting"]
    if not shooting_stats.empty:
        analysis["shots"] = shooting_stats["sh"].iloc[0] if "sh" in shooting_stats.columns else None
        analysis["shots_on_target"] = shooting_stats["sot"].iloc[0] if "sot" in shooting_stats.columns else None
        
        # Calculate shooting accuracy
        if analysis.get("shots") and analysis.get("shots_on_target"):
            analysis["shooting_accuracy"] = round(analysis["shots_on_target"] / analysis["shots"] * 100, 1)
        
        # Calculate shot conversion rate
        if analysis.get("shots") and analysis.get("goals"):
            analysis["shot_conversion"] = round(analysis["goals"] / analysis["shots"] * 100, 1)
    
    # Extract passing stats if available
    passing_stats = df[df["stats_type"] == "Passing"]
    if not passing_stats.empty:
        analysis["passes_completed"] = passing_stats["cmp"].iloc[0] if "cmp" in passing_stats.columns else None
        analysis["passes_attempted"] = passing_stats["att"].iloc[0] if "att" in passing_stats.columns else None
        analysis["pass_completion"] = passing_stats["cmp_pct"].iloc[0] if "cmp_pct" in passing_stats.columns else None
        analysis["key_passes"] = passing_stats["kp"].iloc[0] if "kp" in passing_stats.columns else None
        
        # Calculate passes per 90 minutes
        if analysis.get("passes_attempted") and analysis.get("minutes"):
            analysis["passes_per_90"] = round(analysis["passes_attempted"] / (analysis["minutes"] / 90), 1)
    
    # Extract defensive stats if available
    defense_stats = df[df["stats_type"] == "Defensive Actions"]
    if not defense_stats.empty:
        analysis["tackles"] = defense_stats["tackles"].iloc[0] if "tackles" in defense_stats.columns else None
        analysis["interceptions"] = defense_stats["int"].iloc[0] if "int" in defense_stats.columns else None
        analysis["blocks"] = defense_stats["blocks"].iloc[0] if "blocks" in defense_stats.columns else None
        
        # Calculate defensive actions per 90 minutes
        if analysis.get("minutes"):
            def_actions = sum(analysis.get(x, 0) for x in ["tackles", "interceptions", "blocks"])
            analysis["defensive_actions_per_90"] = round(def_actions / (analysis["minutes"] / 90), 1)
    
    # Calculate per 90 stats for key metrics
    if analysis.get("minutes"):
        minutes_per_90 = analysis["minutes"] / 90
        
        if analysis.get("goals"):
            analysis["goals_per_90"] = round(analysis["goals"] / minutes_per_90, 2)
        
        if analysis.get("assists"):
            analysis["assists_per_90"] = round(analysis["assists"] / minutes_per_90, 2)
        
        if analysis.get("goals") and analysis.get("assists"):
            analysis["goal_contributions_per_90"] = round((analysis["goals"] + analysis["assists"]) / minutes_per_90, 2)
    
    # Get team's total matches to calculate participation percentage
    team_query = """
    SELECT COUNT(*) as total_matches FROM matches 
    WHERE team = :team AND season = :season
    """
    
    team_df = execute_query(team_query, {"team": team, "season": season})
    
    if not team_df.empty and analysis.get("appearances"):
        total_matches = team_df.iloc[0]["total_matches"]
        analysis["participation_percentage"] = round(analysis["appearances"] / total_matches * 100, 1)
    
    return analysis

###############################################################################
# Team Comparison                                                            #
###############################################################################

def compare_teams(teams: List[str], season: int) -> Dict[str, Any]:
    """Compare multiple teams across different metrics.
    
    Args:
        teams: List of team names to compare
        season: Season year
    
    Returns:
        Dictionary with team comparison data
    """
    if not teams or len(teams) < 2:
        return {"error": "At least two teams are required for comparison"}
    
    comparison = {
        "teams": teams,
        "season": season,
        "metrics": {},
        "rankings": {}
    }
    
    # Get data for all teams
    team_data = {}
    for team in teams:
        metrics = calculate_team_advanced_metrics(team, season)
        if metrics:
            team_data[team] = metrics
    
    if not team_data:
        return {"error": "No data found for the specified teams"}
    
    # Define key metrics to compare
    key_metrics = [
        "points_per_game",
        "goals_per_game",
        "goals_against_per_game",
        "clean_sheet_percentage",
        "win_percentage",
        "shot_conversion",
        "shooting_accuracy",
        "avg_possession",
    ]
    
    # Add expected goals metrics if available
    if all("total_xg" in team_data[team] for team in team_data):
        key_metrics.extend(["xg_per_game", "xg_efficiency"])
    
    if all("total_xga" in team_data[team] for team in team_data):
        key_metrics.extend(["xga_per_game", "defensive_efficiency"])
    
    # Compare metrics
    for metric in key_metrics:
        comparison["metrics"][metric] = {team: team_data[team].get(metric) for team in team_data if metric in team_data[team]}
    
    # Calculate rankings for each metric
    for metric in key_metrics:
        if metric in comparison["metrics"] and len(comparison["metrics"][metric]) > 1:
            # Determine if higher is better for this metric
            higher_is_better = metric not in ["goals_against_per_game", "xga_per_game"]
            
            # Extract values and sort
            metric_values = [(team, val) for team, val in comparison["metrics"][metric].items() if val is not None]
            
            if higher_is_better:
                sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=True)
            else:
                sorted_values = sorted(metric_values, key=lambda x: x[1])
            
            # Assign rankings
            comparison["rankings"][metric] = {team: i+1 for i, (team, _) in enumerate(sorted_values)}
    
    # Calculate overall ranking
    if comparison["rankings"]:
        overall_scores = {team: 0 for team in team_data}
        
        for metric, rankings in comparison["rankings"].items():
            for team, rank in rankings.items():
                # Lower rank is better
                overall_scores[team] += rank
        
        # Sort by overall score (lower is better)
        sorted_teams = sorted(overall_scores.items(), key=lambda x: x[1])
        comparison["overall_ranking"] = {team: i+1 for i, (team, _) in enumerate(sorted_teams)}
    
    # Add head-to-head records
    comparison["head_to_head"] = {}
    
    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            h2h = calculate_head_to_head_stats(team1, team2)
            if "error" not in h2h:
                comparison["head_to_head"][f"{team1}_vs_{team2}"] = h2h
    
    return comparison


def identify_team_strengths_weaknesses(team: str, season: int) -> Dict[str, Any]:
    """Identify a team's strengths and weaknesses based on their performance metrics.
    
    Args:
        team: Team name
        season: Season year
    
    Returns:
        Dictionary with strengths and weaknesses analysis
    """
    # Get team's metrics
    metrics = calculate_team_advanced_metrics(team, season)
    
    if not metrics:
        return {"error": f"No data found for {team} in season {season}"}
    
    # Get league average metrics for comparison
    query = """
    SELECT 
        AVG(points) / COUNT(*) AS avg_points_per_game,
        AVG(gf) / COUNT(*) AS avg_goals_per_game,
        AVG(ga) / COUNT(*) AS avg_goals_against_per_game,
        SUM(CASE WHEN ga = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS avg_clean_sheet_percentage,
        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS avg_win_percentage,
        AVG(sh) AS avg_shots_per_game,
        AVG(sot) AS avg_shots_on_target_per_game,
        AVG(poss) AS avg_possession,
        AVG(xg) AS avg_xg_per_game,
        AVG(xga) AS avg_xga_per_game
    FROM matches
    WHERE season = :season
    """
    
    league_df = execute_query(query, {"season": season})
    
    if league_df.empty:
        return {"error": "Could not calculate league averages"}
    
    league_avg = league_df.iloc[0].to_dict()
    
    # Define metrics to compare
    compare_metrics = {
        "points_per_game": {"league_avg": league_avg["avg_points_per_game"], "team": metrics.get("points_per_game")},
        "goals_per_game": {"league_avg": league_avg["avg_goals_per_game"], "team": metrics.get("goals_per_game")},
        "goals_against_per_game": {"league_avg": league_avg["avg_goals_against_per_game"], "team": metrics.get("goals_against_per_game")},
        "clean_sheet_percentage": {"league_avg": league_avg["avg_clean_sheet_percentage"], "team": metrics.get("clean_sheet_percentage")},
        "win_percentage": {"league_avg": league_avg["avg_win_percentage"], "team": metrics.get("win_percentage")},
    }
    
    # Add shot metrics if available
    if "avg_shots_per_game" in league_avg and not pd.isna(league_avg["avg_shots_per_game"]):
        compare_metrics["shots_per_game"] = {"league_avg": league_avg["avg_shots_per_game"], "team": metrics.get("shots_per_game")}
    
    if "avg_shots_on_target_per_game" in league_avg and not pd.isna(league_avg["avg_shots_on_target_per_game"]):
        compare_metrics["shots_on_target_per_game"] = {"league_avg": league_avg["avg_shots_on_target_per_game"], "team": metrics.get("shots_on_target_per_game")}
    
    # Add possession metrics if available
    if "avg_possession" in league_avg and not pd.isna(league_avg["avg_possession"]):
        compare_metrics["avg_possession"] = {"league_avg": league_avg["avg_possession"], "team": metrics.get("avg_possession")}
    
    # Add xG metrics if available
    if "avg_xg_per_game" in league_avg and not pd.isna(league_avg["avg_xg_per_game"]):
        compare_metrics["xg_per_game"] = {"league_avg": league_avg["avg_xg_per_game"], "team": metrics.get("xg_per_game")}
    
    if "avg_xga_per_game" in league_avg and not pd.isna(league_avg["avg_xga_per_game"]):
        compare_metrics["xga_per_game"] = {"league_avg": league_avg["avg_xga_per_game"], "team": metrics.get("xga_per_game")}
    
    # Analyze strengths and weaknesses
    strengths = []
    weaknesses = []
    
    for metric, values in compare_metrics.items():
        if values["team"] is None or values["league_avg"] is None:
            continue
        
        # Define if higher is better for this metric
        higher_is_better = metric not in ["goals_against_per_game", "xga_per_game"]
        
        # Calculate percentage difference from league average
        pct_diff = (values["team"] - values["league_avg"]) / values["league_avg"] * 100
        
        # Determine if this is a strength or weakness
        if (higher_is_better and pct_diff > 15) or (not higher_is_better and pct_diff < -15):
            strengths.append({
                "metric": metric,
                "team_value": round(values["team"], 2),
                "league_avg": round(values["league_avg"], 2),
                "pct_diff": round(pct_diff, 1)
            })
        elif (higher_is_better and pct_diff < -15) or (not higher_is_better and pct_diff > 15):
            weaknesses.append({
                "metric": metric,
                "team_value": round(values["team"], 2),
                "league_avg": round(values["league_avg"], 2),
                "pct_diff": round(pct_diff, 1)
            })
    
    # Add home/away analysis
    if "home_points_per_game" in metrics and "away_points_per_game" in metrics:
        home_away_diff = metrics["home_points_per_game"] - metrics["away_points_per_game"]
        
        if home_away_diff > 0.75:
            strengths.append({
                "metric": "home_performance",
                "description": f"Strong home form with {metrics['home_points_per_game']} PPG vs {metrics['away_points_per_game']} PPG away"
            })
        elif home_away_diff < -0.25:
            strengths.append({
                "metric": "away_performance",
                "description": f"Better away form with {metrics['away_points_per_game']} PPG vs {metrics['home_points_per_game']} PPG at home"
            })
    
    # First half vs second half performance
    if "1st_half_goals_for" in metrics and "2nd_half_goals_for" in metrics:
        if metrics["1st_half_goals_for"] > metrics["2nd_half_goals_for"] * 1.5:
            strengths.append({
                "metric": "1st_half_scoring",
                "description": f"Strong first-half scoring ({metrics['1st_half_goals_for']} goals vs {metrics['2nd_half_goals_for']} in 2nd half)"
            })
        elif metrics["2nd_half_goals_for"] > metrics["1st_half_goals_for"] * 1.5:
            strengths.append({
                "metric": "2nd_half_scoring",
                "description": f"Strong second-half scoring ({metrics['2nd_half_goals_for']} goals vs {metrics['1st_half_goals_for']} in 1st half)"
            })
    
    # Form consistency analysis
    if "form_consistency" in metrics:
        if metrics["form_consistency"] < 2.0:
            strengths.append({
                "metric": "form_consistency",
                "description": "Highly consistent performance throughout the season"
            })
        elif metrics["form_consistency"] > 4.0:
            weaknesses.append({
                "metric": "form_consistency",
                "description": "Inconsistent performance with significant form swings"
            })
    
    # Sort strengths and weaknesses by significance
    strengths = sorted(strengths, key=lambda x: abs(x.get("pct_diff", 0)), reverse=True)
    weaknesses = sorted(weaknesses, key=lambda x: abs(x.get("pct_diff", 0)), reverse=True)
    
    return {
        "team": team,
        "season": season,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "metrics": metrics,
        "league_averages": {k.replace("avg_", ""): v for k, v in league_avg.items()}
    }

###############################################################################
# Match Prediction                                                           #
###############################################################################

def predict_match(home_team: str, away_team: str, season: int) -> Dict[str, Any]:
    """Predict the outcome of a match based on team statistics.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        season: Season year
        
    Returns:
        Dictionary with match prediction
    """
    # Get team metrics
    home_metrics = calculate_team_advanced_metrics(home_team, season)
    away_metrics = calculate_team_advanced_metrics(away_team, season)
    
    if not home_metrics or not away_metrics:
        missing = "home team" if not home_metrics else "away team"
        return {"error": f"Missing data for {missing}"}
    
    # Get head-to-head record
    h2h = calculate_head_to_head_stats(home_team, away_team)
    
    # Initialize prediction
    prediction = {
        "home_team": home_team,
        "away_team": away_team,
        "season": season,
        "home_team_metrics": home_metrics,
        "away_team_metrics": away_metrics,
        "head_to_head": h2h if "error" not in h2h else {},
        "prediction": {},
        "factors": []
    }
    
    # Calculate expected goals
    home_xg = 0
    away_xg = 0
    
    # Home team's offense vs away team's defense
    if "goals_per_game" in home_metrics and "goals_against_per_game" in away_metrics:
        home_xg += home_metrics["goals_per_game"] * 1.1  # Home advantage boost
    
    # Away team's offense vs home team's defense
    if "goals_per_game" in away_metrics and "goals_against_per_game" in home_metrics:
        away_xg += away_metrics["goals_per_game"] * 0.9  # Away disadvantage
    
    # Adjust with xG data if available
    if "xg_per_game" in home_metrics and "xga_per_game" in away_metrics:
        home_xg = (home_xg + home_metrics["xg_per_game"] * 1.1) / 2
    
    if "xg_per_game" in away_metrics and "xga_per_game" in home_metrics:
        away_xg = (away_xg + away_metrics["xg_per_game"] * 0.9) / 2
    
    # Adjust based on home/away performance
    if "home_goals_per_game" in home_metrics:
        home_xg = (home_xg + home_metrics["home_goals_per_game"]) / 2
    
    if "away_goals_per_game" in away_metrics:
        away_xg = (away_xg + away_metrics["away_goals_per_game"]) / 2
    
    # Adjust based on recent head-to-head
    if "matches" in h2h and len(h2h["matches"]) > 0:
        recent_matches = h2h["matches"][:3]  # Last 3 matches
        
        h2h_home_goals = sum(int(m["score"].split("-")[0]) for m in recent_matches 
                            if m["home_team"] == home_team)
        h2h_away_goals = sum(int(m["score"].split("-")[1]) for m in recent_matches 
                            if m["home_team"] == home_team)
        
        h2h_matches_count = len([m for m in recent_matches if m["home_team"] == home_team])
        
        if h2h_matches_count > 0:
            h2h_home_avg = h2h_home_goals / h2h_matches_count
            h2h_away_avg = h2h_away_goals / h2h_matches_count
            
            # Blend with current prediction (30% weight to h2h)
            home_xg = home_xg * 0.7 + h2h_home_avg * 0.3
            away_xg = away_xg * 0.7 + h2h_away_avg * 0.3
    
    # Round to 2 decimals
    home_xg = round(home_xg, 2)
    away_xg = round(away_xg, 2)
    
    # Store expected goals
    prediction["prediction"]["home_expected_goals"] = home_xg
    prediction["prediction"]["away_expected_goals"] = away_xg
    
    # Calculate win probabilities using Poisson distribution
    import scipy.stats as stats
    from math import factorial
    
    # Probability of each scoreline (up to 5 goals each)
    score_probs = {}
    max_goals = 5
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            home_prob = stats.poisson.pmf(home_goals, home_xg)
            away_prob = stats.poisson.pmf(away_goals, away_xg)
            score_probs[f"{home_goals}-{away_goals}"] = home_prob * away_prob
    
    # Normalize probabilities
    total_prob = sum(score_probs.values())
    for score, prob in score_probs.items():
        score_probs[score] = prob / total_prob
    
    # Get most likely scorelines
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    prediction["prediction"]["most_likely_scores"] = [
        {"score": score, "probability": round(prob * 100, 1)} 
        for score, prob in top_scores
    ]
    
    # Calculate overall match outcome probabilities
    home_win_prob = sum(prob for score, prob in score_probs.items() 
                        if int(score.split("-")[0]) > int(score.split("-")[1]))
    draw_prob = sum(prob for score, prob in score_probs.items() 
                    if int(score.split("-")[0]) == int(score.split("-")[1]))
    away_win_prob = sum(prob for score, prob in score_probs.items() 
                        if int(score.split("-")[0]) < int(score.split("-")[1]))
    
    prediction["prediction"]["home_win_probability"] = round(home_win_prob * 100, 1)
    prediction["prediction"]["draw_probability"] = round(draw_prob * 100, 1)
    prediction["prediction"]["away_win_probability"] = round(away_win_prob * 100, 1)
    
    # Determine most likely outcome
    max_prob = max(home_win_prob, draw_prob, away_win_prob)
    if max_prob == home_win_prob:
        prediction["prediction"]["most_likely_outcome"] = f"{home_team} win"
    elif max_prob == draw_prob:
        prediction["prediction"]["most_likely_outcome"] = "Draw"
    else:
        prediction["prediction"]["most_likely_outcome"] = f"{away_team} win"
    
    # Add key factors influencing the prediction
    factors = []
    
    # Home advantage
    home_away_diff = (home_metrics.get("home_points_per_game", 0) or 0) - (home_metrics.get("away_points_per_game", 0) or 0)
    if home_away_diff > 0.5:
        factors.append(f"{home_team} strong home form: {home_metrics.get('home_points_per_game')} PPG at home vs {home_metrics.get('away_points_per_game')} away")
    
    # Recent form
    if "momentum" in home_metrics and "momentum" in away_metrics:
        if home_metrics["momentum"] == "Improving" and away_metrics["momentum"] != "Improving":
            factors.append(f"{home_team} has improving form while {away_team} does not")
        elif away_metrics["momentum"] == "Improving" and home_metrics["momentum"] != "Improving":
            factors.append(f"{away_team} has improving form while {home_team} does not")
    
    # Head-to-head dominance
    if "team1_win_percentage" in h2h and "team2_win_percentage" in h2h:
        if home_team == h2h["team1"] and h2h["team1_win_percentage"] > 60:
            factors.append(f"{home_team} has won {h2h['team1_win_percentage']}% of recent head-to-head matches")
        elif home_team == h2h["team2"] and h2h["team2_win_percentage"] > 60:
            factors.append(f"{home_team} has won {h2h['team2_win_percentage']}% of recent head-to-head matches")
        elif away_team == h2h["team1"] and h2h["team1_win_percentage"] > 60:
            factors.append(f"{away_team} has won {h2h['team1_win_percentage']}% of recent head-to-head matches")
        elif away_team == h2h["team2"] and h2h["team2_win_percentage"] > 60:
            factors.append(f"{away_team} has won {h2h['team2_win_percentage']}% of recent head-to-head matches")
    
    # Attacking/defensive mismatches
    if "goals_per_game" in home_metrics and "goals_per_game" in away_metrics:
        if home_metrics["goals_per_game"] > away_metrics["goals_per_game"] * 1.5:
            factors.append(f"{home_team} has significantly stronger attack ({home_metrics['goals_per_game']} vs {away_metrics['goals_per_game']} goals per game)")
        elif away_metrics["goals_per_game"] > home_metrics["goals_per_game"] * 1.5:
            factors.append(f"{away_team} has significantly stronger attack ({away_metrics['goals_per_game']} vs {home_metrics['goals_per_game']} goals per game)")
    
    if "goals_against_per_game" in home_metrics and "goals_against_per_game" in away_metrics:
        if home_metrics["goals_against_per_game"] < away_metrics["goals_against_per_game"] * 0.6:
            factors.append(f"{home_team} has significantly stronger defense ({home_metrics['goals_against_per_game']} vs {away_metrics['goals_against_per_game']} goals conceded per game)")
        elif away_metrics["goals_against_per_game"] < home_metrics["goals_against_per_game"] * 0.6:
            factors.append(f"{away_team} has significantly stronger defense ({away_metrics['goals_against_per_game']} vs {home_metrics['goals_against_per_game']} goals conceded per game)")
    
    prediction["factors"] = factors
    
    return prediction