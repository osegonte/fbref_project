prediction = predict_match(home_team, away_team, season)
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])
        return prediction
    
    @app.get("/player/{player}")
    def get_player_analysis(player: str, team: str, season: int):
        analysis = analyze_player(player, team, season)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"No data found for player {player}")
        return analysis
    
    @app.get("/compare-teams")
    def compare_multiple_teams(teams: str, season: int):
        team_list = [t.strip() for t in teams.split(",")]
        if len(team_list) < 2:
            raise HTTPException(status_code=400, detail="At least two teams are required for comparison")
            
        comparison = compare_teams(team_list, season)
        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])
        return comparison
    
    @app.get("/leagues")
    def get_leagues():
        return {"leagues": list(LEAGUES.keys())}
    
    @app.get("/database-info")
    def get_database_info():
        # Get info about all tables
        tables = ["matches", "players", "league_tables", "player_valuations"]
        info = {}
        
        for table in tables:
            info[table] = get_table_info(table)
            
        return info
    
    return app


def run_api(host: str = API_HOST, port: int = API_PORT):
    """Run the FastAPI app.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    if not HAS_FASTAPI:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return
        
    import uvicorn
    app = create_api_app()
    if app:
        logger.info(f"Starting API server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        logger.error("Failed to create API app")

###############################################################################
# CLI                                                                       #
###############################################################################

def cmd_scrape(args):
    """Command to scrape data."""
    run_one_time_update(args.league_name, args.season, args.recent, args.async_)


def cmd_plot(args):
    """Command to create visualization."""
    import pandas as pd
    
    if args.type == "treemap":
        # Treemap of goals per club
        sql = """
        SELECT team, SUM(gf) AS goals
        FROM   matches
        WHERE  season = :season AND comp LIKE :league
        GROUP  BY team ORDER BY goals DESC;
        """
        df = execute_query(
            sql, 
            {"season": args.season, "league": f"%{args.league_name}%"}
        )
        
        out_path = Path(args.out) if args.out else None
        treemap(
            df, "goals", "team", 
            f"Goals – {args.league_name} {args.season}/{args.season+1}", 
            out_path,
            use_team_colors=True
        )
               
    elif args.type == "form":
        # Form heatmap for top teams
        sql = """
        SELECT squad FROM league_tables
        WHERE season = :season AND league = :league
        ORDER BY rank
        LIMIT 6;
        """
        df = execute_query(
            sql, 
            {"season": args.season, "league": args.league_name}
        )
        
        if not df.empty:
            teams = df["squad"].tolist()
            out_path = Path(args.out) if args.out else None
            form_heatmap(
                teams, args.season, args.limit or 5, 
                out_path,
                title=f"Form Comparison - {args.league_name} {args.season}/{args.season+1}"
            )
        else:
            # Fallback to getting teams from matches
            sql = """
            SELECT DISTINCT team FROM matches 
            WHERE season = :season AND comp LIKE :league
            LIMIT 6;
            """
            df = execute_query(
                sql, 
                {"season": args.season, "league": f"%{args.league_name}%"}
            )
            
            teams = df["team"].tolist()
            out_path = Path(args.out) if args.out else None
            form_heatmap(
                teams, args.season, args.limit or 5, 
                out_path,
                title=f"Form Comparison - {args.league_name} {args.season}/{args.season+1}"
            )
    
    elif args.type == "bar":
        # Bar chart of goals, points, or other metrics
        metric = args.metric or "gf"
        metric_name = {
            "gf": "Goals For",
            "ga": "Goals Against",
            "points": "Points",
            "sh": "Shots",
            "sot": "Shots on Target",
            "corner_for": "Corners",
            "win_percentage": "Win Percentage",
            "xg": "Expected Goals",
            "xga": "Expected Goals Against"
        }.get(metric, metric)
        
        sql = f"""
        SELECT team, SUM({metric}) AS value
        FROM   matches
        WHERE  season = :season AND comp LIKE :league
        GROUP  BY team ORDER BY value DESC;
        """
        df = execute_query(
            sql, 
            {"season": args.season, "league": f"%{args.league_name}%"}
        )
        
        out_path = Path(args.out) if args.out else None
        bar_plot(
            df, "team", "value", 
            f"{metric_name} – {args.league_name} {args.season}/{args.season+1}", 
            out_path,
            limit=args.limit,
            use_team_colors=True,
            y_label=metric_name
        )
        
    elif args.type == "radar":
        # Radar chart for team performance
        if not args.team:
            logger.error("Team name is required for radar chart")
            return
            
        # Get team metrics
        metrics = calculate_team_advanced_metrics(args.team, args.season)
        if not metrics:
            logger.error(f"No metrics found for {args.team} in season {args.season}")
            return
            
        # Get league averages for comparison
        sql = """
        SELECT 
            AVG(points) / COUNT(*) AS avg_points_per_game,
            AVG(gf) / COUNT(*) AS avg_goals_per_game,
            AVG(ga) / COUNT(*) AS avg_goals_against_per_game,
            SUM(CASE WHEN ga = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS avg_clean_sheet_percentage,
            SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS avg_win_percentage
        FROM matches
        WHERE season = :season
        """
        
        league_df = execute_query(sql, {"season": args.season})
        
        if league_df.empty:
            logger.error("Could not calculate league averages")
            return
            
        league_avg = league_df.iloc[0].to_dict()
        
        # Extract metrics for radar chart
        stat_categories = [
            "points_per_game",
            "goals_per_game",
            "goals_against_per_game",
            "clean_sheet_percentage",
            "win_percentage"
        ]
        
        team_stats = {cat: metrics.get(cat, 0) for cat in stat_categories}
        league_stats = {
            "points_per_game": league_avg["avg_points_per_game"],
            "goals_per_game": league_avg["avg_goals_per_game"],
            "goals_against_per_game": league_avg["avg_goals_against_per_game"],
            "clean_sheet_percentage": league_avg["avg_clean_sheet_percentage"],
            "win_percentage": league_avg["avg_win_percentage"]
        }
        
        # Better labels for display
        display_categories = [
            "Points per Game",
            "Goals per Game",
            "Goals Against per Game",
            "Clean Sheet %",
            "Win %"
        ]
        
        category_mapping = dict(zip(stat_categories, display_categories))
        
        # Create radar chart
        out_path = Path(args.out) if args.out else None
        radar_chart(
            team_stats,
            args.team,
            categories=display_categories,
            comparison_stats=league_stats,
            comparison_name="League Average",
            title=f"{args.team} Performance - {args.season}/{args.season+1}",
            out=out_path
        )
    
    elif args.type == "position":
        # Position chart
        if not args.team:
            logger.error("Team name is required for position chart")
            return
            
        out_path = Path(args.out) if args.out else None
        position_chart(
            args.team,
            args.season,
            out=out_path,
            title=f"{args.team} League Position - {args.season}/{args.season+1}"
        )
    
    elif args.type == "xg":
        # Expected goals timeline
        if not args.team:
            logger.error("Team name is required for xG timeline")
            return
            
        out_path = Path(args.out) if args.out else None
        xg_timeline(
            args.team,
            args.season,
            last_n=args.limit or 10,
            out=out_path,
            title=f"{args.team} Goals vs xG - {args.season}/{args.season+1}"
        )
    
    elif args.type == "dashboard":
        # Team dashboard
        if not args.team:
            logger.error("Team name is required for dashboard")
            return
            
        out_path = Path(args.out) if args.out else None
        dashboard(
            args.team,
            args.season,
            out=out_path,
            title=f"{args.team} Performance Dashboard - {args.season}/{args.season+1}"
        )
    
    elif args.type == "h2h":
        # Head-to-head comparison
        if not args.team or not args.opponent:
            logger.error("Both team and opponent are required for head-to-head comparison")
            return
            
        # Get metrics for both teams
        team1_metrics = calculate_team_advanced_metrics(args.team, args.season)
        team2_metrics = calculate_team_advanced_metrics(args.opponent, args.season)
        
        if not team1_metrics or not team2_metrics:
            logger.error(f"Missing metrics for {args.team} or {args.opponent}")
            return
            
        # Extract comparison metrics
        compare_metrics = {
            "Points per Game": {
                args.team: team1_metrics.get("points_per_game"),
                args.opponent: team2_metrics.get("points_per_game")
            },
            "Goals per Game": {
                args.team: team1_metrics.get("goals_per_game"),
                args.opponent: team2_metrics.get("goals_per_game")
            },
            "Goals Against per Game": {
                args.team: team1_metrics.get("goals_against_per_game"),
                args.opponent: team2_metrics.get("goals_against_per_game")
            },
            "Win Percentage": {
                args.team: team1_metrics.get("win_percentage"),
                args.opponent: team2_metrics.get("win_percentage")
            },
            "Clean Sheet Percentage": {
                args.team: team1_metrics.get("clean_sheet_percentage"),
                args.opponent: team2_metrics.get("clean_sheet_percentage")
            }
        }
        
        # Add possession if available
        if "avg_possession" in team1_metrics and "avg_possession" in team2_metrics:
            compare_metrics["Possession"] = {
                args.team: team1_metrics["avg_possession"],
                args.opponent: team2_metrics["avg_possession"]
            }
        
        # Add xG if available
        if "xg_per_game" in team1_metrics and "xg_per_game" in team2_metrics:
            compare_metrics["xG per Game"] = {
                args.team: team1_metrics["xg_per_game"],
                args.opponent: team2_metrics["xg_per_game"]
            }
        
        out_path = Path(args.out) if args.out else None
        head_to_head_comparison(
            args.team,
            args.opponent,
            compare_metrics,
            out=out_path,
            title=f"{args.team} vs {args.opponent} - {args.season}/{args.season+1}"
        )


def cmd_export(args):
    """Command to export data to file."""
    if args.query:
        # Custom SQL query
        export_data(args.table, args.query, None, args.format, args.out)
    else:
        # Direct table export
        export_data(args.table, None, None, args.format, args.out)


def cmd_api(args):
    """Command to start API server."""
    run_api(args.host, args.port)


def cmd_schedule(args):
    """Command to schedule daily updates."""
    leagues = args.leagues.split(",") if args.leagues else None
    schedule_daily_update(args.hour, args.minute, leagues)


def cmd_analyze(args):
    """Command to run analysis on teams or matches."""
    import json
    
    if args.type == "form":
        # Team form analysis
        form_df = team_form_analysis(args.team, args.matches, args.all_comps)
        if not form_df.empty:
            if args.json:
                print(form_df.to_json(orient="records"))
            else:
                print(form_df.to_string())
        else:
            print(f"No form data available for {args.team}")
            
    elif args.type == "h2h":
        # Head-to-head analysis
        h2h_stats = calculate_head_to_head_stats(args.team1, args.team2, args.matches)
        if "error" not in h2h_stats:
            if args.json:
                print(json.dumps(h2h_stats, default=str))
            else:
                print(f"\nHead-to-head: {args.team1} vs {args.team2}")
                print(f"Total matches: {h2h_stats['total_matches']}")
                print(f"{args.team1} wins: {h2h_stats['team1_wins']} ({h2h_stats['team1_win_percentage']:.1f}%)")
                print(f"{args.team2} wins: {h2h_stats['team2_wins']} ({h2h_stats['team2_win_percentage']:.1f}%)")
                print(f"Draws: {h2h_stats['draws']} ({h2h_stats['draw_percentage']:.1f}%)")
                print(f"Goals: {args.team1} {h2h_stats['team1_goals']} - {h2h_stats['team2_goals']} {args.team2}")
                print(f"Avg. goals per match: {h2h_stats['avg_goals_per_match']:.2f}")
                
                print("\nRecent matches:")
                for i, match in enumerate(h2h_stats['matches'][:5]):
                    print(f"{match['date']}: {match['home_team']} {match['score']} {match['away_team']}")
        else:
            print(h2h_stats["error"])
    
    elif args.type == "advanced":
        # Advanced team metrics
        metrics = calculate_team_advanced_metrics(args.team, args.season)
        if metrics:
            if args.json:
                print(json.dumps(metrics, default=str))
            else:
                print(f"\nAdvanced metrics for {args.team} ({args.season}/{args.season+1}):")
                # Print key metrics in categories
                categories = {
                    "Basic": ["total_matches", "wins", "draws", "losses", "points", "points_per_game"],
                    "Offense": ["goals_for", "goals_per_game", "failed_to_score"],
                    "Defense": ["goals_against", "goals_against_per_game", "clean_sheets", "clean_sheet_percentage"],
                    "Home/Away": ["home_points", "away_points", "home_goals_for", "away_goals_for"]
                }
                
                for category, metric_keys in categories.items():
                    print(f"\n{category}:")
                    for key in metric_keys:
                        if key in metrics:
                            value = metrics[key]
                            if isinstance(value, float):
                                print(f"  {key}: {value:.2f}")
                            else:
                                print(f"  {key}: {value}")
        else:
            print(f"No advanced metrics available for {args.team}")
    
    elif args.type == "strengths":
        # Team strengths and weaknesses
        analysis = identify_team_strengths_weaknesses(args.team, args.season)
        if "error" not in analysis:
            if args.json:
                print(json.dumps(analysis, default=str))
            else:
                print(f"\nStrengths and Weaknesses Analysis for {args.team} ({args.season}/{args.season+1}):")
                
                print("\nStrengths:")
                for strength in analysis["strengths"]:
                    if "description" in strength:
                        print(f"- {strength['description']}")
                    else:
                        metric = strength["metric"].replace("_", " ").title()
                        print(f"- {metric}: {strength['team_value']} (League avg: {strength['league_avg']}, {strength['pct_diff']:+.1f}%)")
                
                print("\nWeaknesses:")
                for weakness in analysis["weaknesses"]:
                    if "description" in weakness:
                        print(f"- {weakness['description']}")
                    else:
                        metric = weakness["metric"].replace("_", " ").title()
                        print(f"- {metric}: {weakness['team_value']} (League avg: {weakness['league_avg']}, {weakness['pct_diff']:+.1f}%)")
        else:
            print(analysis["error"])
    
    elif args.type == "predict":
        # Match prediction
        if not args.team or not args.opponent:
            print("Both team and opponent are required for match prediction")
            return
            
        prediction = predict_match(args.team, args.opponent, args.season)
        if "error" not in prediction:
            if args.json:
                print(json.dumps(prediction, default=str))
            else:
                print(f"\nMatch Prediction: {args.team} vs {args.opponent}")
                print(f"Most likely outcome: {prediction['prediction']['most_likely_outcome']}")
                print(f"Probabilities: {args.team} win {prediction['prediction']['home_win_probability']}%, Draw {prediction['prediction']['draw_probability']}%, {args.opponent} win {prediction['prediction']['away_win_probability']}%")
                print(f"Expected goals: {args.team} {prediction['prediction']['home_expected_goals']:.2f} - {prediction['prediction']['away_expected_goals']:.2f} {args.opponent}")
                
                print("\nMost likely scores:")
                for score in prediction["prediction"]["most_likely_scores"][:3]:
                    print(f"- {score['score']}: {score['probability']}%")
                
                print("\nKey factors:")
                for factor in prediction["factors"]:
                    print(f"- {factor}")
        else:
            print(prediction["error"])
    
    elif args.type == "player":
        # Player analysis
        if not args.player or not args.team:
            print("Both player name and team are required for player analysis")
            return
            
        analysis = analyze_player(args.player, args.team, args.season)
        if analysis:
            if args.json:
                print(json.dumps(analysis, default=str))
            else:
                print(f"\nPlayer Analysis: {args.player} ({args.team}, {args.season}/{args.season+1})")
                print(f"Position: {analysis.get('position', 'Unknown')}")
                print(f"Age: {analysis.get('age', 'Unknown')}")
                print(f"Nationality: {analysis.get('nationality', 'Unknown')}")
                
                print("\nAppearances:")
                print(f"- Games: {analysis.get('appearances', 'N/A')}")
                print(f"- Starts: {analysis.get('starts', 'N/A')}")
                print(f"- Minutes: {analysis.get('minutes', 'N/A')}")
                print(f"- Participation: {analysis.get('participation_percentage', 'N/A')}%")
                
                print("\nOffensive Contribution:")
                print(f"- Goals: {analysis.get('goals', 'N/A')}")
                print(f"- Assists: {analysis.get('assists', 'N/A')}")
                print(f"- Goals per 90: {analysis.get('goals_per_90', 'N/A')}")
                print(f"- Assists per 90: {analysis.get('assists_per_90', 'N/A')}")
                print(f"- Goal Contributions per 90: {analysis.get('goal_contributions_per_90', 'N/A')}")
                
                if "shots" in analysis or "shots_on_target" in analysis:
                    print("\nShooting:")
                    print(f"- Shots: {analysis.get('shots', 'N/A')}")
                    print(f"- Shots on Target: {analysis.get('shots_on_target', 'N/A')}")
                    print(f"- Shooting Accuracy: {analysis.get('shooting_accuracy', 'N/A')}%")
                    print(f"- Shot Conversion: {analysis.get('shot_conversion', 'N/A')}%")
                
                if "passes_completed" in analysis or "passes_attempted" in analysis:
                    print("\nPassing:")
                    print(f"- Passes Completed: {analysis.get('passes_completed', 'N/A')}")
                    print(f"- Passes Attempted: {analysis.get('passes_attempted', 'N/A')}")
                    print(f"- Pass Completion: {analysis.get('pass_completion', 'N/A')}%")
                    print(f"- Key Passes: {analysis.get('key_passes', 'N/A')}")
                    print(f"- Passes per 90: {analysis.get('passes_per_90', 'N/A')}")
                
                if "tackles" in analysis or "interceptions" in analysis:
                    print("\nDefensive Contribution:")
                    print(f"- Tackles: {analysis.get('tackles', 'N/A')}")
                    print(f"- Interceptions: {analysis.get('interceptions', 'N/A')}")
                    print(f"- Blocks: {analysis.get('blocks', 'N/A')}")
                    print(f"- Defensive Actions per 90: {analysis.get('defensive_actions_per_90', 'N/A')}")
        else:
            print(f"No data found for player {args.player} at {args.team}")
    
    elif args.type == "compare":
        # Compare teams
        if not args.teams or "," not in args.teams:
            print("At least two teams are required for comparison (comma-separated)")
            return
            
        team_list = [t.strip() for t in args.teams.split(",")]
        comparison = compare_teams(team_list, args.season)
        
        if "error" not in comparison:
            if args.json:
                print(json.dumps(comparison, default=str))
            else:
                print(f"\nTeam Comparison ({args.season}/{args.season+1}):")
                
                print("\nOverall Ranking:")
                for team, rank in comparison.get("overall_ranking", {}).items():
                    print(f"#{rank}: {team}")
                
                print("\nKey Metrics:")
                for metric, values in comparison.get("metrics", {}).items():
                    print(f"\n{metric.replace('_', ' ').title()}:")
                    for team, value in sorted(values.items(), key=lambda x: x[1], reverse=True):
                        print(f"- {team}: {value:.2f}")
                
                if comparison.get("head_to_head"):
                    print("\nHead-to-Head Records:")
                    for matchup, h2h in comparison["head_to_head"].items():
                        team1, team2 = h2h["team1"], h2h["team2"]
                        print(f"\n{team1} vs {team2}:")
                        print(f"- {team1} wins: {h2h['team1_wins']} ({h2h['team1_win_percentage']:.1f}%)")
                        print(f"- {team2} wins: {h2h['team2_wins']} ({h2h['team2_win_percentage']:.1f}%)")
                        print(f"- Draws: {h2h['draws']} ({h2h['draw_percentage']:.1f}%)")
        else:
            print(comparison["error"])
        

def cmd_cache(args):
    """Command to manage cache."""
    if args.clear:
        deleted = clear_cache(args.older_than)
        print(f"Cleared {deleted} cache files")
    else:
        # Print cache statistics
        cache_files = list(CACHE_DIR_PATH.glob("*.html"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"Cache status:")
        print(f"- Location: {CACHE_DIR_PATH}")
        print(f"- Files: {len(cache_files)}")
        print(f"- Total size: {total_size / (1024*1024):.2f} MB")
        
        # Show cache age distribution
        if cache_files:
            now = time.time()
            age_bins = {
                "< 1 day": 0,
                "1-7 days": 0,
                "7-30 days": 0,
                "> 30 days": 0
            }
            
            for f in cache_files:
                age_days = (now - f.stat().st_mtime) / (60 * 60 * 24)
                if age_days < 1:
                    age_bins["< 1 day"] += 1
                elif age_days < 7:
                    age_bins["1-7 days"] += 1
                elif age_days < 30:
                    age_bins["7-30 days"] += 1
                else:
                    age_bins["> 30 days"] += 1
                    
            print("\nAge distribution:")
            for age, count in age_bins.items():
                print(f"- {age}: {count} files")


def cmd_dashboard(args):
    """Command to run Streamlit dashboard."""
    if not HAS_STREAMLIT:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return
        
    import subprocess
    
    # Get dashboard script path
    dashboard_script = Path(__file__).parent / "fbref_dashboard.py"
    if not dashboard_script.exists():
        logger.error(f"Dashboard script not found at {dashboard_script}")
        return
        
    logger.info(f"Starting Streamlit dashboard at {dashboard_script}")
    subprocess.run(["streamlit", "run", str(dashboard_script)])


def build_cli():
    """Build command-line interface."""
    ap = argparse.ArgumentParser(
        description="FBref Toolkit – Football Data Scraper and Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = ap.add_subparsers(dest="cmd", required=True, help="Command to run")

    # scrape command
    sc = subparsers.add_parser("scrape", help="Pull match data from FBref and store to database")
    sc.add_argument("--league-name", default="Premier League", help="League name (e.g. 'Premier League')")
    sc.add_argument("--season", type=int, default=datetime.now().year, help="Season starting year")
    sc.add_argument("--recent", action="store_true", help="Only last 7 matches (faster update)")
    sc.add_argument("--async", dest="async_", action="store_true", help="Use async scraping (faster but experimental)")
    sc.set_defaults(func=cmd_scrape)

    # plot command
    pl = subparsers.add_parser("plot", help="Create visualizations of football data")
    pl.add_argument("--type", choices=["treemap", "bar", "form", "radar", "position", "xg", "dashboard", "h2h"], 
                   default="treemap", help="Plot type")
    pl.add_argument("--league-name", default="Premier League", help="League name")
    pl.add_argument("--season", type=int, default=datetime.now().year, help="Season starting year")
    pl.add_argument("--team", help="Team name (required for some plots)")
    pl.add_argument("--opponent", help="Opponent team (for head-to-head plots)")
    pl.add_argument("--metric", help="Metric to plot (for bar charts): gf, ga, points, sh, sot, etc.")
    pl.add_argument("--limit", type=int, help="Limit number of items to display")
    pl.add_argument("--out", help="Output file path (PNG)")
    pl.set_defaults(func=cmd_plot)
    
    # export command
    ex = subparsers.add_parser("export", help="Export data to file formats")
    ex.add_argument("table", help="Table name to export")
    ex.add_argument("--query", help="Custom SQL query (optional)")
    ex.add_argument("--format", choices=["csv", "json", "excel", "parquet"], default="csv", help="Output format")
    ex.add_argument("--out", help="Output file path")
    ex.set_defaults(func=cmd_export)
    
    # analyze command
    an = subparsers.add_parser("analyze", help="Run analysis on teams or matches")
    an.add_argument("--type", choices=["form", "h2h", "advanced", "strengths", "predict", "player", "compare"], 
                   required=True, help="Analysis type")
    an.add_argument("--team", help="Team name for single-team analysis")
    an.add_argument("--teams", help="Comma-separated team names for comparison")
    an.add_argument("--team1", help="First team name for head-to-head analysis")
    an.add_argument("--team2", help="Second team name for head-to-head analysis")
    an.add_argument("--opponent", help="Opponent for match prediction")
    an.add_argument("--player", help="Player name for player analysis")
    an.add_argument("--season", type=int, default=datetime.now().year, help="Season year")
    an.add_argument("--matches", type=int, default=5, help="Number of matches to analyze")
    an.add_argument("--all-comps", action="store_true", help="Include all competitions (not just league)")
    an.add_argument("--json", action="store_true", help="Output results as JSON")
    an.set_defaults(func=cmd_analyze)
    
    # api command
    api = subparsers.add_parser("api", help="Start API server")
    api.add_argument("--host", default=API_HOST, help="API server host")
    api.add_argument("--port", type=int, default=API_PORT, help="API server port")
    api.set_defaults(func=cmd_api)
    
    # dashboard command
    dash = subparsers.add_parser("dashboard", help="Run interactive dashboard")
    dash.set_defaults(func=cmd_dashboard)
    
    # schedule command
    sch = subparsers.add_parser("schedule", help="Schedule automated data updates")
    sch.add_argument("--hour", type=int, default=6, help="Hour to run update (24-hour format)")
    sch.add_argument("--minute", type=int, default=0, help="Minute to run update")
    sch.add_argument("--leagues", help="Comma-separated list of leagues to update")
    sch.set_defaults(func=cmd_schedule)
    
    # cache command
    cache = subparsers.add_parser("cache", help="Manage cache")
    cache.add_argument("--clear", action="store_true", help="Clear cache")
    cache.add_argument("--older-than", type=int, help="Only clear files older than N days")
    cache.set_defaults(func=cmd_cache)

    return ap

###############################################################################
# Entry-point                                                               #
###############################################################################

if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()
    
    try:
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
#!/usr/bin/env python
"""
fbref_toolkit.py
================
An advanced Python module for football analytics that bundles data collection, storage,
and visualization. Features include:

• FBref match + player scraping (daily refresh or bulk)
• Multiple league support with configurable parameters
• League standings tables
• Advanced team form analysis and head-to-head statistics
• Transfermarkt integration for player data enrichment
• Comprehensive visualization suite (treemaps, bar plots, line charts)
• Automated data collection with scheduling
• Exportable data in multiple formats (CSV, JSON)
• API endpoints for data access (optional Fast API integration)
• Streamlit dashboard for interactive analytics
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

# Import modules
from config import (
    POSTGRES_URI, REQUEST_DELAY, CACHE_DIR_PATH, LOG_DIR_PATH, 
    LEAGUES, API_HOST, API_PORT, USER_AGENT
)

from database_utils import (
    get_engine, get_session, ensure_tables_exist, upsert, 
    execute_query, export_data, get_table_info
)

from http_utils import fetch, soupify, clear_cache, HTTPError

from scraper import (
    scrape_team, scrape_player_stats, scrape_league_table,
    scrape_head_to_head, get_squad_urls, scrape_teams_async
)

from analytics import (
    team_form_analysis, calculate_head_to_head_stats,
    calculate_team_advanced_metrics, identify_team_strengths_weaknesses,
    compare_teams, predict_match, analyze_player
)

from visualization import (
    treemap, bar_plot, line_plot, form_heatmap, radar_chart,
    position_chart, head_to_head_comparison, shot_map, dashboard,
    xg_timeline
)

# Configure logging
LOG_DIR_PATH.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR_PATH / "fbref_toolkit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_toolkit")

# Optional imports for scheduling 
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    logger.warning("APScheduler not installed. Scheduling features will be disabled.")

# Optional imports for dashboard/API functionality
try:
    import fastapi
    from fastapi import FastAPI, Query, Depends, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. API features will be disabled.")

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    logger.warning("Streamlit not installed. Dashboard features will be disabled.")

###############################################################################
# Transfermarkt integration (contract expiry + market value)                 #
###############################################################################

def tmkt_contract_expiry(player_id: int) -> Optional[str]:
    """Get player contract expiry date from Transfermarkt.
    
    Args:
        player_id: Transfermarkt player ID
        
    Returns:
        Contract expiry date (YYYY-MM-DD) or None if not found
    """
    import requests
    
    TRANSFERMARKT_URL = "https://www.transfermarkt.com"
    url = f"{TRANSFERMARKT_URL}/transfers/contractEndAjax?ajax=1&id={player_id}"
    
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if response.ok:
            data = response.json()
            # Take first entry
            if data and len(data) > 0:
                return data[0]["dateTo"]  # YYYY-MM-DD
    except Exception as e:
        logger.warning(f"Error fetching contract data for player {player_id}: {e}")
    return None


def tmkt_market_value_history(player_id: int) -> pd.DataFrame:
    """Get player market value history from Transfermarkt.
    
    Args:
        player_id: Transfermarkt player ID
        
    Returns:
        DataFrame with market value history
    """
    import requests
    import pandas as pd
    
    TRANSFERMARKT_URL = "https://www.transfermarkt.com"
    url = f"{TRANSFERMARKT_URL}/player/getMarketValueGraphData/{player_id}"
    
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if response.ok:
            data = response.json()
            
            if "data" in data and "marker" in data["data"]:
                values = []
                for point in data["data"]["marker"]:
                    values.append({
                        "player_id": player_id,
                        "date": point["datum"],
                        "value_eur": point["mw"],
                        "club": point.get("verein", ""),
                        "age": point.get("age", "")
                    })
                return pd.DataFrame(values)
    except Exception as e:
        logger.warning(f"Error fetching market value for player {player_id}: {e}")
    
    return pd.DataFrame()


def find_tmkt_player_id(player_name: str, club_name: Optional[str] = None) -> Optional[int]:
    """Search for a player's Transfermarkt ID.
    
    Args:
        player_name: Player name to search
        club_name: Optional club name to refine search
        
    Returns:
        Transfermarkt player ID or None if not found
    """
    import requests
    from bs4 import BeautifulSoup
    
    TRANSFERMARKT_URL = "https://www.transfermarkt.com"
    search_term = player_name
    if club_name:
        search_term += f" {club_name}"
        
    url = f"{TRANSFERMARKT_URL}/search/players/?query={search_term.replace(' ', '+')}"
    
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if response.ok:
            soup = BeautifulSoup(response.text, "html.parser")
            
            player_table = soup.select_one("table.items")
            if not player_table:
                return None
                
            # Find first player link
            player_link = player_table.select_one("td.hauptlink a")
            if player_link and "href" in player_link.attrs:
                href = player_link["href"]
                # Extract ID from URL pattern like /player/playerid/123456
                parts = href.split("/")
                if len(parts) >= 3:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"Error searching for player {player_name}: {e}")
    
    return None

###############################################################################
# Scheduling and automation                                                 #
###############################################################################

def schedule_daily_update(
    hour: int = 6, 
    minute: int = 0, 
    leagues: List[str] = None
) -> None:
    """Schedule daily data updates using APScheduler.
    
    Args:
        hour: Hour of day to run (24-hour format)
        minute: Minute of hour to run
        leagues: List of league names to update (default: Premier League only)
    """
    if not HAS_SCHEDULER:
        logger.error("APScheduler not installed. Install with: pip install apscheduler")
        return
        
    if leagues is None:
        leagues = ["Premier League"]
        
    logger.info(f"Scheduling daily updates at {hour:02d}:{minute:02d}")
    
    def daily_job():
        """Function to run for the daily update."""
        logger.info("Running daily update")
        
        # Ensure database tables exist
        ensure_tables_exist()
        
        current_year = datetime.now().year
        
        # Update each league
        for league_name in leagues:
            league_id = next((data["id"] for name, data in LEAGUES.items() 
                          if name.lower() == league_name.lower()), None)
            
            if not league_id:
                logger.warning(f"Unknown league: {league_name}")
                continue
                
            logger.info(f"Updating {league_name} data")
            
            try:
                # Update league table
                league_table = scrape_league_table(league_id, current_year)
                if not league_table.empty:
                    upsert(league_table, "league_tables", ["rank", "squad", "season", "league"])
                
                # Update team matches (last 7 matches only)
                for team_url in get_squad_urls(league_id, current_year):
                    try:
                        matches_df = scrape_team(team_url, league_name, current_year, recent_only=True)
                        if not matches_df.empty:
                            upsert(matches_df, "matches", ["match_id"])
                            
                            # Optionally update player data
                            players_df = scrape_player_stats(team_url, current_year)
                            if not players_df.empty:
                                upsert(players_df, "players", ["player_id"])
                    except Exception as e:
                        logger.error(f"Error updating team {team_url}: {e}")
            except Exception as e:
                logger.error(f"Error updating league {league_name}: {e}")
                
        # Clear old cache files (older than 7 days)
        clear_cache(days_old=7)
                
        logger.info("Daily update completed")
    
    # Create scheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(daily_job, 'cron', hour=hour, minute=minute, id='daily_update')
    
    try:
        logger.info("Starting scheduler")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def run_one_time_update(
    league_name: str, 
    season: int, 
    recent_only: bool = False,
    use_async: bool = False
) -> None:
    """Run a one-time update of data.
    
    Args:
        league_name: League name to update
        season: Season year to update
        recent_only: If True, only get last 7 matches
        use_async: If True, use async scraping for better performance
    """
    # Get league ID
    league_id = next((data["id"] for name, data in LEAGUES.items() 
                   if name.lower() == league_name.lower()), None)
    
    if not league_id:
        logger.error(f"Unknown league: {league_name}")
        return
        
    logger.info(f"Running one-time update for {league_name} {season}")
    
    # Ensure database tables exist
    ensure_tables_exist()
    
    # Update league table
    league_table = scrape_league_table(league_id, season)
    if not league_table.empty:
        upsert(league_table, "league_tables", ["rank", "squad", "season", "league"])
    
    # Update team matches
    if use_async:
        import asyncio
        
        # Run async scraping
        logger.info(f"Using async scraping for {league_name}")
        team_dataframes = asyncio.run(scrape_teams_async(league_id, season, recent_only))
        
        # Save results to database
        for df in team_dataframes:
            if not df.empty:
                upsert(df, "matches", ["match_id"])
                logger.info(f"Updated {len(df)} matches for {df['team'].iloc[0]}")
    else:
        # Use regular synchronous scraping
        for team_url in get_squad_urls(league_id, season):
            try:
                matches_df = scrape_team(team_url, league_name, season, recent_only=recent_only)
                if not matches_df.empty:
                    upsert(matches_df, "matches", ["match_id"])
                    logger.info(f"Updated {len(matches_df)} matches for {matches_df['team'].iloc[0]}")
                    
                    # Update player data
                    players_df = scrape_player_stats(team_url, season)
                    if not players_df.empty:
                        upsert(players_df, "players", ["player_id"])
                        logger.info(f"Updated {len(players_df)} player records for {matches_df['team'].iloc[0]}")
            except Exception as e:
                logger.error(f"Error updating team {team_url}: {e}")
    
    logger.info(f"One-time update completed for {league_name} {season}")

###############################################################################
# API Integration                                                           #
###############################################################################

def create_api_app():
    """Create FastAPI app for data access."""
    if not HAS_FASTAPI:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None
        
    app = FastAPI(
        title="FBref API", 
        description="Football data API powered by FBref",
        version="2.0.0"
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def read_root():
        return {"message": "Welcome to FBref API", "version": "2.0.0"}
    
    @app.get("/teams")
    def get_teams(league: Optional[str] = None, season: Optional[int] = None):
        query_parts = ["SELECT DISTINCT team FROM matches"]
        params = {}
        
        if league or season:
            query_parts.append("WHERE")
            conditions = []
            
            if league:
                conditions.append("comp LIKE :league")
                params["league"] = f"%{league}%"
                
            if season:
                conditions.append("season = :season")
                params["season"] = season
                
            query_parts.append(" AND ".join(conditions))
            
        query_parts.append("ORDER BY team")
        query = " ".join(query_parts)
        
        df = execute_query(query, params)
        return {"teams": df["team"].tolist()}
    
    @app.get("/matches")
    def get_matches(
        team: Optional[str] = None, 
        opponent: Optional[str] = None,
        season: Optional[int] = None, 
        limit: int = 10,
        include_stats: bool = False
    ):
        query_parts = ["SELECT * FROM matches"]
        params = {}
        conditions = []
        
        if team:
            conditions.append("team = :team")
            params["team"] = team
            
        if opponent:
            conditions.append("opponent = :opponent")
            params["opponent"] = opponent
            
        if season:
            conditions.append("season = :season")
            params["season"] = season
            
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
            
        query_parts.append("ORDER BY date DESC LIMIT :limit")
        params["limit"] = limit
        
        query = " ".join(query_parts)
        df = execute_query(query, params)
        
        if include_stats and not df.empty and team:
            # Add team form data
            form_data = team_form_analysis(team, matches=5)
            if not form_data.empty:
                form_dict = form_data.iloc[0].to_dict()
                return {
                    "matches": df.to_dict(orient="records"),
                    "team_stats": form_dict
                }
        
        return {"matches": df.to_dict(orient="records")}
    
    @app.get("/league-table")
    def get_league_table(league: str, season: Optional[int] = None):
        if not season:
            season = datetime.now().year
            
        query = """
        SELECT * FROM league_tables 
        WHERE league = :league AND season = :season 
        ORDER BY rank
        """
        
        df = execute_query(query, {"league": league, "season": season})
        return {"table": df.to_dict(orient="records")}
    
    @app.get("/team-form/{team}")
    def get_team_form(team: str, matches: int = 5, all_comps: bool = False):
        form_data = team_form_analysis(team, matches, all_comps)
        if form_data.empty:
            raise HTTPException(status_code=404, detail=f"No form data found for {team}")
        return form_data.iloc[0].to_dict()
    
    @app.get("/head-to-head/{team1}/{team2}")
    def get_head_to_head(team1: str, team2: str, limit: int = 10):
        h2h = calculate_head_to_head_stats(team1, team2, limit)
        if "error" in h2h:
            raise HTTPException(status_code=404, detail=h2h["error"])
        return h2h
    
    @app.get("/team-analysis/{team}")
    def get_team_analysis(team: str, season: int):
        metrics = calculate_team_advanced_metrics(team, season)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No data found for {team} in season {season}")
        return metrics
    
    @app.get("/team-strengths-weaknesses/{team}")
    def get_team_strengths_weaknesses(team: str, season: int):
        analysis = identify_team_strengths_weaknesses(team, season)
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        return analysis
    
    @app.get("/predict-match/{home_team}/{away_team}")
    def get_match_prediction(home_team: str, away_team: str, season: int):
        prediction = predict_match(home_team, away_team, season)
        if "error" in