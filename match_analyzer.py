#!/usr/bin/env python3
"""
Match Analyzer

Processes football match data by team and match time, generating specialized reports
and visualizations. Analyzes team performance, head-to-head statistics, and
upcoming match predictions.

Usage:
  python match_analyzer.py --team "Liverpool"  # Analyze specific team
  python match_analyzer.py --upcoming          # Analyze upcoming matches
  python match_analyzer.py --league "Premier League"  # Analyze by league
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/match_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("match_analyzer")

def get_db_engine():
    """Get database connection"""
    pg_uri = os.getenv('PG_URI')
    if not pg_uri:
        logger.error("PG_URI environment variable not found")
        raise ValueError("Database connection string not configured")
    
    return create_engine(pg_uri)

def analyze_team(team_name, output_dir="data/analysis"):
    """Analyze performance data for a specific team"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info(f"Analyzing team: {team_name}")
        
        # Get team's recent matches
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    date, opponent, venue, result, gf, ga, points, 
                    sh, sot, possession, xg, xga, comp
                FROM recent_matches
                WHERE team = :team
                ORDER BY date DESC
                LIMIT 10
            """)
            
            result = conn.execute(query, {"team": team_name})
            
            # Convert to DataFrame
            columns = result.keys()
            matches = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if matches.empty:
            logger.warning(f"No match data found for {team_name}")
            return {'success': False, 'error': 'No match data found'}
        
        logger.info(f"Found {len(matches)} recent matches for {team_name}")
        
        # Basic statistics
        stats = {
            'team': team_name,
            'matches_analyzed': len(matches),
            'wins': len(matches[matches['result'] == 'W']),
            'draws': len(matches[matches['result'] == 'D']),
            'losses': len(matches[matches['result'] == 'L']),
            'goals_for': matches['gf'].sum(),
            'goals_against': matches['ga'].sum(),
            'points': matches['points'].sum(),
            'points_per_game': matches['points'].mean(),
            'average_possession': matches['possession'].mean(),
            'shots_per_game': matches['sh'].mean(),
            'shots_on_target_per_game': matches['sot'].mean(),
            'xg_per_game': matches['xg'].mean(),
            'xga_per_game': matches['xga'].mean()
        }
        
        # Save stats to CSV
        stats_df = pd.DataFrame([stats])
        stats_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_stats.csv")
        stats_df.to_csv(stats_file, index=False)
        
        # Get home/away split
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    venue,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                    SUM(gf) as goals_for,
                    SUM(ga) as goals_against,
                    SUM(gf) - SUM(ga) as goal_diff,
                    SUM(points) as total_points,
                    ROUND(AVG(points)::numeric, 2) as avg_points_per_game,
                    ROUND(AVG(sh)::numeric, 2) as avg_shots,
                    ROUND(AVG(sot)::numeric, 2) as avg_shots_on_target
                FROM recent_matches
                WHERE team = :team
                GROUP BY venue
                ORDER BY venue
            """)
            
            result = conn.execute(query, {"team": team_name})
            
            # Convert to DataFrame
            columns = result.keys()
            venue_stats = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if not venue_stats.empty:
            # Save venue stats to CSV
            venue_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_venue_stats.csv")
            venue_stats.to_csv(venue_file, index=False)
        
        # Get form over time (last 10 games)
        matches['match_number'] = range(1, len(matches) + 1)
        matches['cumulative_points'] = matches['points'].cumsum()
        matches['rolling_ppg'] = matches['cumulative_points'] / matches['match_number']
        
        # Save form data
        form_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_form.csv")
        matches.to_csv(form_file, index=False)
        
        # Create visualization folder
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate form chart
        plt.figure(figsize=(12, 6))
        plt.plot(matches['match_number'], matches['rolling_ppg'], marker='o', linewidth=2)
        plt.title(f"{team_name} - Points Per Game (Rolling Average)")
        plt.xlabel("Match Number")
        plt.ylabel("Points Per Game")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(viz_dir, f"{team_name.replace(' ', '_')}_form_chart.png")
        plt.savefig(chart_file)
        plt.close()
        
        # Generate goals chart
        plt.figure(figsize=(12, 6))
        plt.bar(matches['match_number'], matches['gf'], color='blue', alpha=0.7, label='Goals For')
        plt.bar(matches['match_number'], matches['ga'], color='red', alpha=0.7, label='Goals Against')
        plt.title(f"{team_name} - Goals For/Against by Match")
        plt.xlabel("Match Number")
        plt.ylabel("Goals")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        goals_chart_file = os.path.join(viz_dir, f"{team_name.replace(' ', '_')}_goals_chart.png")
        plt.savefig(goals_chart_file)
        plt.close()
        
        # Get upcoming fixtures
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    date, home_team, away_team, league, venue, start_time
                FROM fixtures
                WHERE (home_team = :team OR away_team = :team)
                AND date >= CURRENT_DATE
                ORDER BY date ASC
                LIMIT 5
            """)
            
            result = conn.execute(query, {"team": team_name})
            
            # Convert to DataFrame
            columns = result.keys()
            upcoming = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if not upcoming.empty:
            # Save upcoming fixtures
            upcoming_file = os.path.join(output_dir, f"{team_name.replace(' ', '_')}_upcoming.csv")
            upcoming.to_csv(upcoming_file, index=False)
        
        logger.info(f"Team analysis for {team_name} completed successfully")
        return {
            'success': True, 
            'team': team_name,
            'matches_analyzed': len(matches),
            'stats_file': stats_file,
            'form_file': form_file,
            'charts': [chart_file, goals_chart_file] if 'chart_file' in locals() else []
        }
        
    except Exception as e:
        logger.exception(f"Error analyzing team {team_name}: {e}")
        return {'success': False, 'error': str(e)}

def analyze_upcoming_matches(days=7, output_dir="data/analysis/upcoming"):
    """Analyze upcoming matches and generate predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info(f"Analyzing upcoming matches for the next {days} days")
        
        # Get upcoming fixtures
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    date, home_team, away_team, league, country, 
                    start_time, venue
                FROM fixtures
                WHERE date BETWEEN CURRENT_DATE AND (CURRENT_DATE + :days)
                ORDER BY date ASC, start_time ASC
            """)
            
            result = conn.execute(query, {"days": days})
            
            # Convert to DataFrame
            columns = result.keys()
            fixtures = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if fixtures.empty:
            logger.warning(f"No upcoming fixtures found for the next {days} days")
            return {'success': False, 'error': 'No upcoming fixtures found'}
        
        logger.info(f"Found {len(fixtures)} upcoming fixtures")
        
        # Save all fixtures
        fixtures_file = os.path.join(output_dir, f"upcoming_fixtures_{days}_days.csv")
        fixtures.to_csv(fixtures_file, index=False)
        
        # Group fixtures by date
        fixtures['date'] = pd.to_datetime(fixtures['date'])
        by_date = fixtures.groupby(fixtures['date'].dt.date)
        
        # For each date, create a separate file
        for date, group in by_date:
            date_str = date.strftime("%Y-%m-%d")
            date_file = os.path.join(output_dir, f"fixtures_{date_str}.csv")
            group.to_csv(date_file, index=False)
        
        # Analyze each fixture and create match predictions
        predictions = []
        
        for _, fixture in fixtures.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            match_date = fixture['date']
            
            # Get home team stats
            with engine.connect() as conn:
                home_query = text("""
                    SELECT 
                        COUNT(*) as matches,
                        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                        AVG(points) as avg_points,
                        AVG(gf) as avg_gf,
                        AVG(ga) as avg_ga,
                        AVG(xg) as avg_xg,
                        AVG(xga) as avg_xga
                    FROM recent_matches
                    WHERE team = :team AND venue = 'Home'
                """)
                
                home_result = conn.execute(home_query, {"team": home_team})
                home_stats = dict(zip(home_result.keys(), home_result.fetchone() or [None] * 9))
            
            # Get away team stats
            with engine.connect() as conn:
                away_query = text("""
                    SELECT 
                        COUNT(*) as matches,
                        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                        AVG(points) as avg_points,
                        AVG(gf) as avg_gf,
                        AVG(ga) as avg_ga,
                        AVG(xg) as avg_xg,
                        AVG(xga) as avg_xga
                    FROM recent_matches
                    WHERE team = :team AND venue = 'Away'
                """)
                
                away_result = conn.execute(away_query, {"team": away_team})
                away_stats = dict(zip(away_result.keys(), away_result.fetchone() or [None] * 9))
            
            # Get head-to-head stats
            with engine.connect() as conn:
                h2h_query = text("""
                    SELECT 
                        team, opponent, result,
                        COUNT(*) as matches,
                        SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                        SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses
                    FROM recent_matches
                    WHERE (team = :team1 AND opponent = :team2)
                       OR (team = :team2 AND opponent = :team1)
                    GROUP BY team, opponent, result
                """)
                
                h2h_result = conn.execute(h2h_query, {"team1": home_team, "team2": away_team})
                
                # Process h2h results
                h2h_data = [dict(zip(h2h_result.keys(), row)) for row in h2h_result]
                
                # Calculate h2h summary
                home_wins = sum(1 for row in h2h_data if row['team'] == home_team and row['result'] == 'W')
                away_wins = sum(1 for row in h2h_data if row['team'] == away_team and row['result'] == 'W')
                draws = sum(1 for row in h2h_data if row['result'] == 'D')
                
                h2h_stats = {
                    'matches': home_wins + away_wins + draws,
                    'home_wins': home_wins,
                    'away_wins': away_wins,
                    'draws': draws
                }
            
            # Simple prediction logic (can be enhanced)
            home_strength = home_stats.get('avg_points', 0) or 0
            away_strength = away_stats.get('avg_points', 0) or 0
            
            # Adjust for home advantage
            home_advantage = 0.2
            
            # Calculate win probabilities
            home_win_prob = (home_strength + home_advantage) / (home_strength + away_strength + home_advantage)
            away_win_prob = away_strength / (home_strength + away_strength + home_advantage)
            draw_prob = 1 - home_win_prob - away_win_prob
            
            # Ensure probabilities are valid
            if home_win_prob < 0: home_win_prob = 0
            if away_win_prob < 0: away_win_prob = 0
            if draw_prob < 0: draw_prob = 0
            
            total = home_win_prob + away_win_prob + draw_prob
            if total > 0:
                home_win_prob /= total
                away_win_prob /= total
                draw_prob /= total
            
            # Create prediction
            prediction = {
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'league': fixture['league'],
                'home_win_prob': round(home_win_prob * 100, 1),
                'draw_prob': round(draw_prob * 100, 1),
                'away_win_prob': round(away_win_prob * 100, 1),
                'home_expected_goals': round(home_stats.get('avg_gf', 0) or 0, 1),
                'away_expected_goals': round(away_stats.get('avg_gf', 0) or 0, 1),
                'h2h_home_wins': h2h_stats['home_wins'],
                'h2h_away_wins': h2h_stats['away_wins'],
                'h2h_draws': h2h_stats['draws']
            }
            
            predictions.append(prediction)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_file = os.path.join(output_dir, "match_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)
        
        # Group by date for daily predictions
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        by_date = predictions_df.groupby(predictions_df['date'].dt.date)
        
        for date, group in by_date:
            date_str = date.strftime("%Y-%m-%d")
            date_pred_file = os.path.join(output_dir, f"predictions_{date_str}.csv")
            group.to_csv(date_pred_file, index=False)
        
        logger.info("Upcoming match analysis completed successfully")
        return {
            'success': True,
            'fixtures_analyzed': len(fixtures),
            'predictions_made': len(predictions),
            'fixtures_file': fixtures_file,
            'predictions_file': predictions_file
        }
        
    except Exception as e:
        logger.exception(f"Error analyzing upcoming matches: {e}")
        return {'success': False, 'error': str(e)}

def analyze_by_league(league_name, output_dir="data/analysis/leagues"):
    """Analyze all teams within a specific league"""
    league_dir = os.path.join(output_dir, league_name.replace(' ', '_'))
    os.makedirs(league_dir, exist_ok=True)
    
    try:
        engine = get_db_engine()
        
        logger.info(f"Analyzing league: {league_name}")
        
        # Get teams in the league
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT team
                FROM recent_matches
                WHERE comp = :league
                ORDER BY team
            """)
            
            result = conn.execute(query, {"league": league_name})
            teams = [row[0] for row in result]
        
        if not teams:
            logger.warning(f"No teams found in league: {league_name}")
            return {'success': False, 'error': 'No teams found in league'}
        
        logger.info(f"Found {len(teams)} teams in {league_name}")
        
        # Analyze each team
        team_results = []
        for team in teams:
            logger.info(f"Analyzing {team} in {league_name}")
            team_dir = os.path.join(league_dir, "teams")
            team_result = analyze_team(team, output_dir=team_dir)
            team_results.append(team_result)
        
        # Create league table
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    team,
                    COUNT(*) as matches_played,
                    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
                    SUM(gf) as goals_for,
                    SUM(ga) as goals_against,
                    SUM(gf) - SUM(ga) as goal_diff,
                    SUM(points) as points
                FROM recent_matches
                WHERE comp = :league
                GROUP BY team
                ORDER BY points DESC, goal_diff DESC
            """)
            
            result = conn.execute(query, {"league": league_name})
            
            # Convert to DataFrame
            columns = result.keys()
            table = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if not table.empty:
            # Add position column
            table['position'] = range(1, len(table) + 1)
            
            # Save league table
            table_file = os.path.join(league_dir, "league_table.csv")
            table.to_csv(table_file, index=False)
            
            # Generate visualization
            viz_dir = os.path.join(league_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Points chart
            plt.figure(figsize=(12, 8))
            bars = plt.barh(table['team'], table['points'], color='skyblue')
            plt.xlabel('Points')
            plt.ylabel('Team')
            plt.title(f"{league_name} - Points Table")
            plt.tight_layout()
            
            # Add point values at the end of each bar
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.0f}", 
                        ha='left', va='center')
            
            # Save chart
            points_chart = os.path.join(viz_dir, "league_points.png")
            plt.savefig(points_chart)
            plt.close()
            
            # Goal difference chart
            plt.figure(figsize=(12, 8))
            plt.barh(table['team'], table['goal_diff'], color=['green' if x >= 0 else 'red' for x in table['goal_diff']])
            plt.xlabel('Goal Difference')
            plt.ylabel('Team')
            plt.title(f"{league_name} - Goal Difference")
            plt.axvline(x=0, color='black', linestyle='--')
            plt.tight_layout()
            
            # Save chart
            gd_chart = os.path.join(viz_dir, "league_goal_diff.png")
            plt.savefig(gd_chart)
            plt.close()
        
        # Get upcoming fixtures in this league
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    date, home_team, away_team, venue, start_time
                FROM fixtures
                WHERE league = :league
                AND date >= CURRENT_DATE
                ORDER BY date ASC, start_time ASC
                LIMIT 20
            """)
            
            result = conn.execute(query, {"league": league_name})
            
            # Convert to DataFrame
            columns = result.keys()
            fixtures = pd.DataFrame([dict(zip(columns, row)) for row in result])
        
        if not fixtures.empty:
            # Save upcoming fixtures
            fixtures_file = os.path.join(league_dir, "upcoming_fixtures.csv")
            fixtures.to_csv(fixtures_file, index=False)
        
        logger.info(f"League analysis for {league_name} completed successfully")
        return {
            'success': True,
            'league': league_name,
            'teams_analyzed': len(teams),
            'league_table_file': table_file if 'table_file' in locals() else None,
            'fixtures_file': fixtures_file if 'fixtures_file' in locals() else None
        }
        
    except Exception as e:
        logger.exception(f"Error analyzing league {league_name}: {e}")
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Football Match Analyzer")
    parser.add_argument('--team', help='Analyze specific team')
    parser.add_argument('--upcoming', action='store_true', help='Analyze upcoming matches')
    parser.add_argument('--league', help='Analyze specific league')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze for upcoming matches')
    
    args = parser.parse_args()
    
    if args.team:
        result = analyze_team(args.team)
        if result['success']:
            print(f"Successfully analyzed {args.team}")
            print(f"Stats saved to {result.get('stats_file')}")
            return 0
        else:
            print(f"Failed to analyze {args.team}: {result.get('error')}")
            return 1
    
    elif args.upcoming:
        result = analyze_upcoming_matches(days=args.days)
        if result['success']:
            print(f"Successfully analyzed upcoming matches for the next {args.days} days")
            print(f"Predictions saved to {result.get('predictions_file')}")
            return 0
        else:
            print(f"Failed to analyze upcoming matches: {result.get('error')}")
            return 1
    
    elif args.league:
        result = analyze_by_league(args.league)
        if result['success']:
            print(f"Successfully analyzed {args.league}")
            print(f"League table saved to {result.get('league_table_file')}")
            return 0
        else:
            print(f"Failed to analyze {args.league}: {result.get('error')}")
            return 1
    
    else:
        print("No analysis option selected. Use --team, --upcoming, or --league")
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())