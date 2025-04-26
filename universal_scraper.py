# Get data rows
            for row in matches_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                        
                        # Extract links where available
                        if cell.select_one('a'):
                            link = cell.select_one('a').get('href', '')
                            row_data[f"{col_name}_link"] = link
                
                if row_data:
                    matches_data.append(row_data)
            
            if not matches_data:
                raise ParseError(f"No match data found for {team_name}")
                
            # Convert to DataFrame
            matches_df = pd.DataFrame(matches_data)
            
            # Add shooting stats if available
            self._add_shooting_stats(soup, matches_df, team_url)
            
            # Add standard stats for corners data
            self._add_standard_stats(soup, matches_df, team_url)
            
            # Add team information
            matches_df['team'] = team_name
            matches_df['league_id'] = league_id
            matches_df['league_name'] = league_name
            
            # Create match_id
            if 'date' in matches_df.columns and 'opponent' in matches_df.columns:
                matches_df['match_id'] = matches_df.apply(
                    lambda row: f"{row['date']}_{team_name}_{row['opponent']}".replace(' ', '_'),
                    axis=1
                )
            
            # Calculate points based on result
            if 'result' in matches_df.columns:
                matches_df['points'] = matches_df['result'].apply(
                    lambda x: 3 if x == 'W' else (1 if x == 'D' else 0)
                )
            
            # Add home/away indicator
            if 'venue' in matches_df.columns:
                matches_df['is_home'] = matches_df['venue'].apply(
                    lambda x: x.lower() == 'home' if pd.notna(x) else None
                )
            
            # Filter for league matches if competition column exists
            if 'comp' in matches_df.columns and league_name:
                league_df = matches_df[matches_df['comp'].str.contains(league_name, case=False, na=False)]
                
                if not league_df.empty:
                    matches_df = league_df
            
            # Add current timestamp
            matches_df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Limit to max_matches
            if len(matches_df) > max_matches:
                matches_df = matches_df.sort_values(by='date', ascending=False).head(max_matches).reset_index(drop=True)
            
            return matches_df
            
        except NetworkError as e:
            # Re-raise network errors
            raise
        except Exception as e:
            raise ParseError(f"Error parsing matches for {team_name}: {str(e)}")
    
    def _add_shooting_stats(self, soup, matches_df, team_url):
        """
        Add shooting stats to the matches DataFrame
        
        Args:
            soup: BeautifulSoup object for the team page
            matches_df: DataFrame with match data
            team_url: URL of the team page
        """
        try:
            # Find link to shooting stats
            shooting_link = None
            for link in soup.select('a'):
                href = link.get('href', '')
                if 'matchlogs/all_comps/shooting' in href:
                    shooting_link = f"https://fbref.com{href}"
                    break
            
            if not shooting_link:
                logger.warning(f"No shooting stats link found for {team_url}")
                return
            
            # Get shooting stats
            shooting_html = self.requester.fetch(shooting_link)
            shooting_soup = BeautifulSoup(shooting_html, 'html.parser')
            
            # Find shooting stats table
            stats_table = None
            for table in shooting_soup.select('table'):
                if table.get('id') and 'shooting' in table.get('id'):
                    stats_table = table
                    break
                
                # Backup method: look for shooting in caption
                if table.select('caption') and 'shooting' in table.select_one('caption').text.lower():
                    stats_table = table
                    break
            
            if not stats_table:
                logger.warning(f"No shooting stats table found for {team_url}")
                return
                
            # Parse the table
            data = []
            headers = []
            
            # Get header row
            header_row = stats_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    col_name = th.get('data-stat', th.text.strip())
                    # Remove any multi-level header indicators
                    if '.' in col_name:
                        col_name = col_name.split('.')[-1]
                    headers.append(col_name)
            
            # Get data rows
            for row in stats_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                
                if row_data:
                    data.append(row_data)
            
            if not data:
                logger.warning(f"No shooting stats data found for {team_url}")
                return
                
            shooting_df = pd.DataFrame(data)
            
            # Merge with match data if date column exists in both
            if 'date' in shooting_df.columns and 'date' in matches_df.columns:
                # For each column in shooting_df (except date), add it to matches_df
                for col in shooting_df.columns:
                    if col != 'date' and col not in matches_df.columns:
                        # Create a date-based lookup dictionary
                        lookup = dict(zip(shooting_df['date'], shooting_df[col]))
                        
                        # Add the column to matches_df
                        matches_df[col] = matches_df['date'].map(lookup)
            
        except Exception as e:
            logger.warning(f"Error adding shooting stats: {e}")
    
    def _add_standard_stats(self, soup, matches_df, team_url):
        """
        Add standard stats (corners, possession) to the matches DataFrame
        
        Args:
            soup: BeautifulSoup object for the team page
            matches_df: DataFrame with match data
            team_url: URL of the team page
        """
        try:
            # Find link to standard stats
            standard_link = None
            for link in soup.select('a'):
                href = link.get('href', '')
                if 'matchlogs/all_comps/stats' in href:
                    standard_link = f"https://fbref.com{href}"
                    break
            
            if not standard_link:
                logger.warning(f"No standard stats link found for {team_url}")
                return
            
            # Get standard stats
            standard_html = self.requester.fetch(standard_link)
            standard_soup = BeautifulSoup(standard_html, 'html.parser')
            
            # Find standard stats table
            stats_table = None
            for table in standard_soup.select('table'):
                if table.get('id') and 'stats' in table.get('id'):
                    stats_table = table
                    break
                
                # Backup method: look for stats in caption
                if table.select('caption') and 'stats' in table.select_one('caption').text.lower():
                    stats_table = table
                    break
            
            if not stats_table:
                logger.warning(f"No standard stats table found for {team_url}")
                return
                
            # Parse the table
            data = []
            headers = []
            
            # Get header row
            header_row = stats_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    col_name = th.get('data-stat', th.text.strip())
                    # Remove any multi-level header indicators
                    if '.' in col_name:
                        col_name = col_name.split('.')[-1]
                    headers.append(col_name)
            
            # Get data rows
            for row in stats_table.select('tbody tr'):
                # Skip non-data rows
                if row.get('class') and 'spacer' in row.get('class'):
                    continue
                
                row_data = {}
                for i, cell in enumerate(row.select('td, th')):
                    if i < len(headers):
                        col_name = headers[i]
                        row_data[col_name] = cell.text.strip()
                
                if row_data:
                    data.append(row_data)
            
            if not data:
                logger.warning(f"No standard stats data found for {team_url}")
                return
                
            standard_df = pd.DataFrame(data)
            
            # Merge with match data if date column exists in both
            if 'date' in standard_df.columns and 'date' in matches_df.columns:
                # Look for corners column
                corners_col = None
                for col in standard_df.columns:
                    if col.lower() in ['ck', 'corners', 'corner_kicks', 'corner']:
                        corners_col = col
                        break
                
                if corners_col:
                    # Create a date-based lookup dictionary
                    lookup = dict(zip(standard_df['date'], standard_df[corners_col]))
                    
                    # Add the column to matches_df
                    matches_df['corners_for'] = matches_df['date'].map(lookup)
                
                # Look for possession column
                poss_col = None
                for col in standard_df.columns:
                    if col.lower() in ['poss', 'possession']:
                        poss_col = col
                        break
                
                if poss_col:
                    # Create a date-based lookup dictionary
                    lookup = dict(zip(standard_df['date'], standard_df[poss_col]))
                    
                    # Add the column to matches_df
                    matches_df['possession'] = matches_df['date'].map(lookup)
            
        except Exception as e:
            logger.warning(f"Error adding standard stats: {e}")

class UniversalScraper:
    """
    Main scraper class that provides a unified interface for all scraping operations.
    Configurable for different leagues, teams, and match-day pairings.
    """
    
    def __init__(self, db_uri=None, cache_dir="data/cache", 
                 output_dir="data/leagues", min_delay=10, max_delay=20,
                 cache_max_age=24):
        """
        Initialize the universal scraper
        
        Args:
            db_uri: PostgreSQL connection URI (optional)
            cache_dir: Directory for HTML cache
            output_dir: Directory for CSV output files
            min_delay: Minimum delay between requests
            max_delay: Maximum delay between requests
            cache_max_age: Maximum cache age in hours
        """
        self.db_uri = db_uri or os.getenv('PG_URI')
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.requester = PoliteRequester(
            cache_dir=cache_dir, 
            max_cache_age=cache_max_age,
            min_delay=min_delay,
            max_delay=max_delay
        )
        
        self.team_parser = TeamParser(self.requester)
        
        if self.db_uri:
            self.db_manager = DatabaseManager(self.db_uri)
        else:
            self.db_manager = None
            logger.warning("No database URI provided - database storage disabled")
    
    def run(self, task_config):
        """
        Run a scraping task based on configuration
        
        Args:
            task_config: Dictionary with task configuration
                Required fields:
                - league: League code (e.g., "EPL", "LALIGA") or "CUSTOM" for direct ID usage
                Optional fields:
                - custom_league_id: League ID when using "CUSTOM" league code
                - custom_league_name: League name when using "CUSTOM" league code
                - teams: List of team pairs for matchday analysis (e.g., [["Arsenal", "Chelsea"], ["Man Utd", "Liverpool"]])
                - lookback: Number of matches to retrieve per team (default: 7)
                - force_refresh: Whether to bypass cache (default: False)
                - store_db: Whether to store results in database (default: True if db_uri provided)
                - fetch_league_info: Whether to fetch league info dynamically from FBref (default: False)
                
        Returns:
            Dictionary with task results including scraped data
            
        Raises:
            ValueError: If task configuration is invalid
            Various scraper exceptions for specific errors
        """
        # Validate configuration
        if not task_config or not isinstance(task_config, dict):
            raise ValueError("Task configuration must be a non-empty dictionary")
        
        if "league" not in task_config:
            raise ValueError("Task configuration must specify a league")
        
        # Get league information
        league_code = task_config["league"]
        
        # Handle custom league ID
        if league_code == "CUSTOM":
            if "custom_league_id" not in task_config:
                raise ValueError("When using 'CUSTOM' league code, you must provide 'custom_league_id'")
            
            league_id = task_config["custom_league_id"]
            league_name = task_config.get("custom_league_name", f"League {league_id}")
            
        # Handle dynamic league fetching
        elif task_config.get("fetch_league_info", False):
            all_leagues = LeagueManager.fetch_all_leagues_from_fbref()
            if league_code in all_leagues:
                league_id = league_code
                league_name = all_leagues[league_code]
            else:
                # Try to find it in predefined leagues
                try:
                    league_info = LeagueManager.get_league_info(league_code)
                    league_id = league_info["id"]
                    league_name = league_info["name"]
                except ValueError:
                    raise ValueError(f"League code '{league_code}' not found in fetched or predefined leagues")
        
        # Use predefined leagues
        else:
            try:
                league_info = LeagueManager.get_league_info(league_code)
            except ValueError as e:
                raise ValueError(f"Invalid league code: {e}")
            
            league_id = league_info["id"]
            league_name = league_info["name"]
        
        # Get options
        lookback = task_config.get("lookback", 7)
        force_refresh = task_config.get("force_refresh", False)
        store_db = task_config.get("store_db", self.db_manager is not None)
        
        # Get league URL and HTML
        league_url = LeagueManager.get_league_url(league_code)
        try:
            league_html = self.requester.fetch(league_url, use_cache=not force_refresh)
        except NetworkError as e:
            raise NetworkError(f"Failed to fetch league page: {e}")
        
        # Process based on task type
        if "teams" in task_config and task_config["teams"]:
            # Team pairs mode (matchday analysis)
            team_pairs = task_config["teams"]
            return self._process_team_pairs(team_pairs, league_html, league_id, league_name, lookback, store_db)
        else:
            # Full league mode
            return self._process_league(league_html, league_id, league_name, lookback, store_db)
    
    def _process_team_pairs(self, team_pairs, league_html, league_id, league_name, lookback, store_db):
        """
        Process team pairs for matchday analysis
        
        Args:
            team_pairs: List of team pairs
            league_html: HTML content of the league page
            league_id: League ID
            league_name: League name
            lookback: Number of matches to retrieve per team
            store_db: Whether to store results in database
            
        Returns:
            Dictionary with results
        """
        all_teams_data = []
        failed_teams = []
        
        # Validate team pairs
        if not isinstance(team_pairs, list):
            raise ValueError("Team pairs must be a list")
        
        # Process each team pair
        for pair in team_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                logger.warning(f"Invalid team pair format: {pair}. Skipping.")
                continue
            
            home_team, away_team = pair
            
            # Process each team
            for team_name in [home_team, away_team]:
                try:
                    # Find team URL
                    team_url = self.team_parser.find_team_url_by_name(team_name, league_html)
                    
                    if not team_url:
                        logger.warning(f"Could not find URL for team: {team_name}")
                        failed_teams.append({"team": team_name, "reason": "URL not found"})
                        continue
                    
                    # Parse team matches
                    team_df = self.team_parser.parse_team_matches(
                        team_url, 
                        league_id, 
                        league_name,
                        max_matches=lookback
                    )
                    
                    if team_df is not None and not team_df.empty:
                        all_teams_data.append(team_df)
                        logger.info(f"Successfully scraped {len(team_df)} matches for {team_name}")
                    else:
                        logger.warning(f"No valid data found for {team_name}")
                        failed_teams.append({"team": team_name, "reason": "No data found"})
                
                except (NetworkError, ParseError) as e:
                    logger.error(f"Error processing team {team_name}: {e}")
                    failed_teams.append({"team": team_name, "reason": str(e)})
                
                except Exception as e:
                    logger.error(f"Unexpected error processing team {team_name}: {e}")
                    failed_teams.append({"team": team_name, "reason": f"Unexpected error: {str(e)}"})
        
        # Combine data from all teams
        if not all_teams_data:
            logger.warning("No team data collected")
            result = {
                "status": "error",
                "message": "No data collected",
                "failed_teams": failed_teams
            }
            return result
        
        combined_df = pd.concat(all_teams_data, ignore_index=True)
        
        # Add metadata
        combined_df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save to CSV
        league_dir = os.path.join(self.output_dir, league_id)
        os.makedirs(league_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        csv_filename = f"{league_id}_{timestamp}_matches.csv"
        csv_path = os.path.join(league_dir, csv_filename)
        
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(combined_df)} matches to {csv_path}")
        
        # Save to database if requested
        if store_db and self.db_manager:
            try:
                stored_count = self.db_manager.store_matches(combined_df)
                logger.info(f"Stored {stored_count} matches in database")
                db_success = True
            except StorageError as e:
                logger.error(f"Database storage error: {e}")
                db_success = False
        else:
            db_success = None
        
        # Return results
        result = {
            "status": "success",
            "teams_processed": len(all_teams_data),
            "matches_collected": len(combined_df),
            "csv_path": csv_path,
            "failed_teams": failed_teams,
            "db_success": db_success
        }
        
        return result
    
    def _process_league(self, league_html, league_id, league_name, lookback, store_db):
        """
        Process a full league
        
        Args:
            league_html: HTML content of the league page
            league_id: League ID
            league_name: League name
            lookback: Number of matches to retrieve per team
            store_db: Whether to store results in database
            
        Returns:
            Dictionary with results
        """
        all_teams_data = []
        failed_teams = []
        
        try:
            # Extract all team URLs
            team_urls = self.team_parser.extract_team_urls(league_html)
            
            # Process each team
            for team_url in team_urls:
                team_name = self.team_parser.extract_team_name(team_url)
                
                try:
                    # Parse team matches
                    team_df = self.team_parser.parse_team_matches(
                        team_url, 
                        league_id, 
                        league_name, 
                        max_matches=lookback
                    )
                    
                    if team_df is not None and not team_df.empty:
                        all_teams_data.append(team_df)
                        logger.info(f"Successfully scraped {len(team_df)} matches for {team_name}")
                    else:
                        logger.warning(f"No valid data found for {team_name}")
                        failed_teams.append({"team": team_name, "reason": "No data found"})
                    
                    # Add delay between teams to avoid rate limiting
                    delay = random.uniform(10, 20)
                    logger.debug(f"Waiting {delay:.2f}s before next team...")
                    time.sleep(delay)
                    
                except (NetworkError, ParseError) as e:
                    logger.error(f"Error processing team {team_name}: {e}")
                    failed_teams.append({"team": team_name, "reason": str(e)})
                
                except Exception as e:
                    logger.error(f"Unexpected error processing team {team_name}: {e}")
                    failed_teams.append({"team": team_name, "reason": f"Unexpected error: {str(e)}"})
        
        except ParseError as e:
            logger.error(f"Error extracting team URLs: {e}")
            return {
                "status": "error",
                "message": f"Failed to extract team URLs: {e}",
            }
        
        # Combine data from all teams
        if not all_teams_data:
            logger.warning("No team data collected")
            result = {
                "status": "error",
                "message": "No data collected",
                "failed_teams": failed_teams
            }
            return result
        
        combined_df = pd.concat(all_teams_data, ignore_index=True)
        
        # Add metadata
        combined_df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save to CSV
        league_dir = os.path.join(self.output_dir, league_id)
        os.makedirs(league_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        csv_filename = f"{league_id}_{timestamp}_matches.csv"
        csv_path = os.path.join(league_dir, csv_filename)
        
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(combined_df)} matches to {csv_path}")
        
        # Also save a "latest" version
        latest_path = os.path.join(league_dir, f"{league_id}_latest.csv")
        combined_df.to_csv(latest_path, index=False)
        
        # Save to database if requested
        if store_db and self.db_manager:
            try:
                stored_count = self.db_manager.store_matches(combined_df)
                logger.info(f"Stored {stored_count} matches in database")
                db_success = True
            except StorageError as e:
                logger.error(f"Database storage error: {e}")
                db_success = False
        else:
            db_success = None
        
        # Return results
        result = {
            "status": "success",
            "teams_processed": len(all_teams_data),
            "matches_collected": len(combined_df),
            "csv_path": csv_path,
            "latest_path": latest_path,
            "failed_teams": failed_teams,
            "db_success": db_success
        }
        
        return result
    
    def to_postgres(self, csv_path, db_uri=None):
        """
        Load a CSV file into PostgreSQL
        
        Args:
            csv_path: Path to CSV file
            db_uri: Database URI (optional, uses instance value if not provided)
            
        Returns:
            Number of rows inserted/updated
            
        Raises:
            StorageError: If database operation fails
        """
        # Initialize DB manager if needed
        if db_uri:
            db_manager = DatabaseManager(db_uri)
        elif self.db_manager:
            db_manager = self.db_manager
        else:
            raise StorageError("No database URI provided")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Store in database
            return db_manager.store_matches(df)
            
        except Exception as e:
            raise StorageError(f"Error loading CSV to database: {e}")
        
        finally:
            # Close connection if we created a new manager
            if db_uri:
                db_manager.close()
    
    def close(self):
        """Clean up resources"""
        if self.db_manager:
            self.db_manager.close()

def main():
    """
    CLI entry point for the universal scraper
    
    Examples:
        # Scrape English Premier League
        python universal_scraper.py --league EPL
        
        # Scrape La Liga with team pairs
        python universal_scraper.py --league LALIGA --pairs Barcelona,RealMadrid Valencia,Sevilla
        
        # Scrape Series A with 5 match lookback
        python universal_scraper.py --league SERIEA --lookback 5
        
        # Use direct league ID (for leagues not in predefined list)
        python universal_scraper.py --league-id 257 --league-name "OFC Nations Cup"
        
        # Fetch all available leagues from FBref
        python universal_scraper.py --fetch-all-leagues
    """
    parser = argparse.ArgumentParser(description="Universal FBref Scraper")
    
    league_group = parser.add_mutually_exclusive_group(required=False)
    league_group.add_argument("--league", "-l", type=str,
                        help="League code (e.g., EPL, LALIGA, SERIEA)")
    league_group.add_argument("--league-id", type=str,
                        help="Direct FBref league ID (e.g., 257 for OFC Nations Cup)")
    
    parser.add_argument("--league-name", type=str,
                        help="League name (required when using --league-id)")
    
    parser.add_argument("--pairs", "-p", type=str, nargs="+",
                        help="Team pairs for matchday analysis (format: Team1,Team2)")
    
    parser.add_argument("--lookback", "-n", type=int, default=7,
                        help="Number of matches to retrieve per team (default: 7)")
    
    parser.add_argument("--force-refresh", "-f", action="store_true",
                        help="Force refresh cache")
    
    parser.add_argument("--no-db", action="store_true",
                        help="Disable database storage")
    
    parser.add_argument("--list-leagues", action="store_true",
                        help="List all predefined supported leagues")
    
    parser.add_argument("--fetch-all-leagues", action="store_true",
                        help="Fetch and display all leagues available on FBref")
    
    parser.add_argument("--config", "-c", type=str,
                        help="Path to JSON/YAML config file")
    
    args = parser.parse_args()
    
    # List predefined leagues if requested
    if args.list_leagues:
        print("Predefined supported leagues:")
        for league in LeagueManager.get_all_leagues():
            print(f"  {league['code']}: {league['name']} ({league['country']})")
        return 0
    
    # Fetch and display all leagues from FBref
    if args.fetch_all_leagues:
        print("Fetching all available leagues from FBref...")
        leagues = LeagueManager.fetch_all_leagues_from_fbref()
        if leagues:
            print("\nAvailable Leagues on FBref:")
            print("=" * 60)
            print(f"{'ID':<6} {'League Name':<50}")
            print("-" * 60)
            
            # Sort by ID for easier reading
            for league_id, name in sorted(leagues.items(), key=lambda x: int(x[0])):
                print(f"{league_id:<6} {name:<50}")
                
            print("=" * 60)
            print(f"Total: {len(leagues)} leagues available")
        else:
            print("Failed to fetch leagues from FBref. Please try again later.")
        return 0
    
    # Create scraper
    scraper = UniversalScraper()
    
    try:
        # Prepare task configuration
        if args.config:
            # Load from config file
            with open(args.config, 'r') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    import yaml
                    task_config = yaml.safe_load(f)
                else:
                    task_config = json.load(f)
        else:
            # Validate that we have either a league code or league ID
            if not args.league and not args.league_id:
                parser.error("You must specify either --league or --league-id")
            
            # Handle direct league ID
            if args.league_id:
                if not args.league_name:
                    parser.error("When using --league-id, you must also provide --league-name")
                
                task_config = {
                    "league": "CUSTOM",
                    "custom_league_id": args.league_id,
                    "custom_league_name": args.league_name,
                    "lookback": args.lookback,
                    "force_refresh": args.force_refresh,
                    "store_db": not args.no_db
                }
            else:
                # Build from CLI arguments with standard league code
                task_config = {
                    "league": args.league,
                    "lookback": args.lookback,
                    "force_refresh": args.force_refresh,
                    "store_db": not args.no_db
                }
            
            # Process team pairs if provided
            if args.pairs:
                task_config["teams"] = []
                for pair_str in args.pairs:
                    teams = pair_str.split(',')
                    if len(teams) == 2:
                        task_config["teams"].append(teams)
                    else:
                        logger.warning(f"Invalid team pair format: {pair_str}. Should be Team1,Team2")
        
        # Run the scraper
        result = scraper.run(task_config)
        
        # Print results
        print(json.dumps(result, indent=2))
        
        if result["status"] == "success":
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        return 1
        
    finally:
        scraper.close()

if __name__ == "__main__":
    sys.exit(main())!/usr/bin/env python3
"""
Universal FBref Scraper

A comprehensive scraper for football data from FBref.com that supports:
- Multi-league scraping with configurable options
- Team pairing analysis for matchday predictions
- Historical match data collection with configurable lookback periods
- Output to CSV files (by league) and PostgreSQL database

This module consolidates functionality from separate scrapers into a unified,
task-driven design that can be easily integrated with scheduling systems.
"""

import os
import sys
import time
import random
import logging
import json
import argparse
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Any
import pandas as pd
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/universal_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("universal_scraper")

# Exception classes
class ScraperException(Exception):
    """Base exception for scraper errors"""
    pass

class NetworkError(ScraperException):
    """Exception raised for network-related errors"""
    pass

class ParseError(ScraperException):
    """Exception raised for HTML parsing errors"""
    pass

class StorageError(ScraperException):
    """Exception raised for data storage errors"""
    pass

class RateLimitError(NetworkError):
    """Exception raised for rate limiting issues"""
    pass

# User-Agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0'
]

class RateLimitHandler:
    """
    Handler for rate limiting to avoid overloading the server.
    Implements adaptive delays, exponential backoff, and domain cooldowns.
    """
    
    def __init__(self, min_delay=10, max_delay=20, cooldown_threshold=3):
        """
        Initialize the rate limit handler
        
        Args:
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            cooldown_threshold: Number of rate limits before enforcing domain cooldown
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cooldown_threshold = cooldown_threshold
        self.rate_limited_count = 0
        self.last_request_time = 0
        
        # Domain-level cooldown tracker
        self.domain_cooldown_until = 0
        
        # Request history for adaptive pacing
        self.request_history = []
        self.rate_limit_history = []
    
    def wait_before_request(self):
        """
        Determine and apply appropriate wait time before making a request
        
        Returns:
            The actual delay applied in seconds
        """
        current_time = time.time()
        
        # Check for domain-level cooldown
        if current_time < self.domain_cooldown_until:
            cooldown_wait = self.domain_cooldown_until - current_time
            logger.info(f"Domain cooldown in effect, waiting {cooldown_wait:.2f}s")
            time.sleep(cooldown_wait)
            current_time = time.time()
        
        # Calculate base delay with FBref's recommended crawl-delay
        base_delay = random.uniform(self.min_delay, self.max_delay)
        
        # Add penalty for recent rate limits
        if self.rate_limited_count > 0:
            penalty = min(self.rate_limited_count * 5, 30)  # Cap at 30s extra
            base_delay += penalty
        
        # Ensure minimum delay since last request
        elapsed = current_time - self.last_request_time
        if elapsed < base_delay:
            wait_time = base_delay - elapsed
            logger.debug(f"Waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
        self.request_history.append(self.last_request_time)
        
        # Clean old history
        self._clean_history()
        
        return base_delay
    
    def _clean_history(self):
        """Remove request records older than 1 hour"""
        cutoff_time = time.time() - 3600
        self.request_history = [t for t in self.request_history if t > cutoff_time]
        self.rate_limit_history = [t for t in self.rate_limit_history if t > cutoff_time]
    
    def handle_rate_limit(self):
        """
        Handle a rate limit response (429)
        
        Returns:
            Backoff time in seconds
        """
        current_time = time.time()
        self.rate_limited_count += 1
        self.rate_limit_history.append(current_time)
        
        # Check if we need a domain-level cooldown
        recent_rate_limits = len([t for t in self.rate_limit_history if t > current_time - 300])
        
        if recent_rate_limits >= self.cooldown_threshold:
            # Implement a longer domain-level cooldown (60-120 seconds)
            cooldown_time = random.uniform(60, 120)
            self.domain_cooldown_until = current_time + cooldown_time
            logger.warning(f"Too many rate limits ({recent_rate_limits} in 5 min). Domain cooldown for {cooldown_time:.2f}s")
            return cooldown_time
        
        # Calculate backoff time (exponential)
        backoff_time = 2 ** min(self.rate_limited_count, 6)  # Cap at 64 seconds
        backoff_time *= random.uniform(0.8, 1.2)  # Add jitter
        
        logger.warning(f"Rate limited, waiting {backoff_time:.2f}s before retry")
        return backoff_time
    
    def reset_after_success(self):
        """Gradually reduce rate limit counter after successful requests"""
        if self.rate_limited_count > 0:
            self.rate_limited_count = max(0, self.rate_limited_count - 0.5)

class PoliteRequester:
    """
    Makes polite requests to web servers with caching, rate limiting,
    and other good citizen behaviors.
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache", 
                 max_cache_age: int = 24,  # Default 24 hour cache
                 min_delay: int = 10,
                 max_delay: int = 20):
        """
        Initialize the requester
        
        Args:
            cache_dir: Directory to store cached responses
            max_cache_age: Maximum age of cached responses in hours
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
        """
        self.cache_dir = cache_dir
        self.max_cache_age = max_cache_age
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize rate limit handler
        self.rate_handler = RateLimitHandler(
            min_delay=min_delay,
            max_delay=max_delay
        )
        
        # Create a session for persistent cookies
        self.session = requests.Session()
    
    def get_cached_response(self, url: str) -> Optional[str]:
        """
        Get cached response for a URL if available and not expired
        
        Args:
            url: URL to check for cached response
            
        Returns:
            Cached HTML content or None if not cached or expired
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        if os.path.exists(cache_file):
            # Check age of cache file
            file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            
            if file_age_hours < self.max_cache_age:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read cache file for {url}: {e}")
        
        return None
    
    def save_to_cache(self, url: str, content: str) -> bool:
        """
        Save response content to cache
        
        Args:
            url: URL associated with the content
            content: HTML content to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False
    
    def get_random_headers(self) -> Dict[str, str]:
        """
        Get random browser-like headers
        
        Returns:
            Dictionary of headers
        """
        user_agent = random.choice(USER_AGENTS)
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        }
        
        return headers
    
    def fetch(self, url: str, use_cache: bool = True, max_retries: int = 5) -> str:
        """
        Fetch URL content with caching, rate-limiting, and retry logic
        
        Args:
            url: URL to fetch
            use_cache: Whether to use/update cache
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content as string
            
        Raises:
            NetworkError: If the request fails after all retries
            RateLimitError: If rate limiting persists after all retries
        """
        # Check cache first if enabled
        if use_cache:
            cached_content = self.get_cached_response(url)
            if cached_content:
                logger.debug(f"Using cached version of {url}")
                return cached_content
        
        # Apply rate limit delay
        self.rate_handler.wait_before_request()
        
        # Make request with retries
        for attempt in range(max_retries):
            try:
                headers = self.get_random_headers()
                
                # Add referrer for more natural requests
                if random.random() < 0.8:
                    # Make it look like internal navigation
                    parts = url.split("/")
                    if len(parts) > 3:
                        base_url = "/".join(parts[:3])
                        possible_referrers = [
                            f"{base_url}/",
                            f"{base_url}/en/",
                            f"{base_url}/en/comps/"
                        ]
                        headers['Referer'] = random.choice(possible_referrers)
                
                logger.debug(f"Requesting {url} (attempt {attempt+1}/{max_retries})")
                
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=30,
                    allow_redirects=True
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    backoff = self.rate_handler.handle_rate_limit()
                    
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"Rate limited after {max_retries} attempts for {url}")
                    
                    time.sleep(backoff)
                    continue
                
                # Handle other errors
                response.raise_for_status()
                
                # Success - reset rate limit counter
                self.rate_handler.reset_after_success()
                
                # Cache the successful response
                if use_cache:
                    self.save_to_cache(url, response.text)
                
                return response.text
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                        raise RateLimitError(f"Rate limited after {max_retries} attempts for {url}")
                    else:
                        raise NetworkError(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                
                backoff = 2 ** attempt * (1 + random.random() * 0.3)
                logger.warning(f"Request failed: {e}. Retrying in {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
        
        # Should never reach here due to exception in the loop
        raise NetworkError(f"Failed to fetch {url} after {max_retries} attempts")

class DatabaseManager:
    """
    Manager for database operations with PostgreSQL.
    Handles schema creation, data insertion, and queries.
    """
    
    def __init__(self, connection_uri=None):
        """
        Initialize database manager
        
        Args:
            connection_uri: PostgreSQL connection URI
        """
        self.connection_uri = connection_uri or os.getenv('PG_URI')
        self.conn = None
    
    def initialize_db(self):
        """
        Initialize database connection and create tables if needed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connection_uri:
                logger.warning("No database URI provided")
                return False
                
            self.conn = psycopg2.connect(self.connection_uri)
            
            # Create tables if they don't exist
            with self.conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recent_matches (
                    match_id VARCHAR(100) PRIMARY KEY,
                    date DATE,
                    team VARCHAR(50),
                    opponent VARCHAR(50),
                    venue VARCHAR(20),
                    result CHAR(1),
                    gf INTEGER,
                    ga INTEGER,
                    points INTEGER,
                    sh INTEGER,
                    sot INTEGER,
                    dist FLOAT,
                    fk FLOAT,
                    pk INTEGER,
                    pkatt INTEGER,
                    possession FLOAT,
                    corners_for INTEGER,
                    corners_against INTEGER,
                    xg FLOAT,
                    xga FLOAT,
                    comp VARCHAR(50),
                    league_id VARCHAR(10),
                    league_name VARCHAR(50),
                    round VARCHAR(50),
                    season INTEGER,
                    is_home BOOLEAN,
                    scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indexes if they don't exist
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_team ON recent_matches(team);
                CREATE INDEX IF NOT EXISTS idx_matches_date ON recent_matches(date);
                CREATE INDEX IF NOT EXISTS idx_matches_league ON recent_matches(league_id);
                """)
                
                self.conn.commit()
                logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False
    
    def store_matches(self, matches_df):
        """
        Store or update matches in database with upsert logic
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            Number of matches stored/updated
            
        Raises:
            StorageError: If database operation fails
        """
        if self.conn is None:
            success = self.initialize_db()
            if not success:
                raise StorageError("Could not initialize database connection")
            
        try:
            # Ensure DataFrame columns are lowercase to match DB
            matches_df.columns = [c.lower() for c in matches_df.columns]
            
            # Convert date strings to dates
            if 'date' in matches_df.columns and matches_df['date'].dtype == 'object':
                matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
            
            # Prepare data for insert/update
            required_columns = ['match_id', 'date', 'team', 'opponent']
            
            # Check if required columns exist
            for col in required_columns:
                if col not in matches_df.columns:
                    raise StorageError(f"Required column '{col}' missing from DataFrame")
            
            # Generate the SQL
            columns = [c for c in matches_df.columns if c != 'scrape_date']
            column_str = ', '.join(columns)
            values_placeholder = ', '.join(['%s'] * len(columns))
            
            # Create upsert query using ON CONFLICT
            query = f"""
            INSERT INTO recent_matches ({column_str})
            VALUES ({values_placeholder})
            ON CONFLICT (match_id) 
            DO UPDATE SET 
            """
            
            # Add all columns to update
            update_clauses = []
            for col in columns:
                if col != 'match_id':  # Don't update the PK
                    update_clauses.append(f"{col} = EXCLUDED.{col}")
            
            query += ', '.join(update_clauses)
            
            # Prepare values
            values = []
            for _, row in matches_df.iterrows():
                row_values = [row[col] if col in row else None for col in columns]
                values.append(row_values)
            
            # Execute the upsert
            with self.conn.cursor() as cursor:
                execute_values(cursor, query, values)
                self.conn.commit()
                
            row_count = len(values)
            logger.info(f"Stored/updated {row_count} matches in the database")
            return row_count
            
        except Exception as e:
            self.conn.rollback()
            raise StorageError(f"Error storing matches in database: {e}")
    
    def get_recent_matches(self, team_name, limit=7):
        """
        Get recent matches for a team
        
        Args:
            team_name: Name of the team
            limit: Maximum number of matches to return
            
        Returns:
            DataFrame with recent matches
            
        Raises:
            StorageError: If database operation fails
        """
        if self.conn is None:
            success = self.initialize_db()
            if not success:
                raise StorageError("Could not initialize database connection")
            
        try:
            query = """
            SELECT * FROM recent_matches
            WHERE team = %s
            ORDER BY date DESC
            LIMIT %s
            """
            
            return pd.read_sql_query(query, self.conn, params=(team_name, limit))
            
        except Exception as e:
            raise StorageError(f"Error retrieving recent matches: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

class LeagueManager:
    """
    Manager for league information and mapping.
    Provides standardized league IDs, names, and URLs.
    """
    
    # Map of standard league codes to FBref IDs
    LEAGUE_MAP = {
        # Major European Leagues
        "EPL": {"id": "9", "name": "Premier League", "country": "England"},
        "LALIGA": {"id": "12", "name": "La Liga", "country": "Spain"},
        "SERIEA": {"id": "11", "name": "Serie A", "country": "Italy"},
        "BUNDESLIGA": {"id": "20", "name": "Bundesliga", "country": "Germany"},
        "LIGUE1": {"id": "13", "name": "Ligue 1", "country": "France"},
        "EREDIVISIE": {"id": "23", "name": "Eredivisie", "country": "Netherlands"},
        "PRIMEIRA": {"id": "32", "name": "Primeira Liga", "country": "Portugal"},
        "CHAMPIONSHIP": {"id": "10", "name": "Championship", "country": "England"},
        
        # North & South American Leagues
        "MLS": {"id": "22", "name": "Major League Soccer", "country": "United States"},
        "BRASILEIRAO": {"id": "24", "name": "Campeonato Brasileiro Série A", "country": "Brazil"},
        "BRASILEIRAO_B": {"id": "38", "name": "Campeonato Brasileiro Série B", "country": "Brazil"},
        "LIGAMX": {"id": "31", "name": "Liga MX", "country": "Mexico"},
        "LIGAARG": {"id": "21", "name": "Liga Profesional de Fútbol Argentina", "country": "Argentina"},
        "COPALIBERT": {"id": "14", "name": "Copa Libertadores de América", "country": "South America"},
        "COPASUDA": {"id": "205", "name": "Copa CONMEBOL Sudamericana", "country": "South America"},
        
        # European Secondary Leagues
        "LEAGUEONE": {"id": "15", "name": "EFL League One", "country": "England"},
        "LEAGUETWO": {"id": "16", "name": "EFL League Two", "country": "England"},
        "EERSTEDIV": {"id": "51", "name": "Eerste Divisie", "country": "Netherlands"},
        "SERIEB": {"id": "18", "name": "Serie B", "country": "Italy"},
        "SEGUNDADIV": {"id": "17", "name": "Spanish Segunda División", "country": "Spain"},
        "LIGUE2": {"id": "60", "name": "Ligue 2", "country": "France"},
        "SCOTPREM": {"id": "40", "name": "Scottish Premiership", "country": "Scotland"},
        "SCOTCHAMP": {"id": "72", "name": "Scottish Championship", "country": "Scotland"},
        
        # Women's Football
        "FAWSL": {"id": "189", "name": "FA Women's Super League", "country": "England"},
        "FRAUENBL": {"id": "183", "name": "Frauen-Bundesliga", "country": "Germany"},
        "LIGAF": {"id": "230", "name": "Liga F", "country": "Spain"},
        "NWSL": {"id": "182", "name": "National Women's Soccer League", "country": "United States"},
        "DAMALSVK": {"id": "187", "name": "Damallsvenskan", "country": "Sweden"},
        "EREDIVW": {"id": "195", "name": "Eredivisie Vrouwen", "country": "Netherlands"},
        
        # International Competitions
        "WORLDCUP": {"id": "1", "name": "FIFA World Cup", "country": "International"},
        "UEFACL": {"id": "8", "name": "UEFA Champions League", "country": "Europe"},
        "UEFAEL": {"id": "19", "name": "UEFA Europa League", "country": "Europe"},
        "UEFACL": {"id": "882", "name": "UEFA Conference League", "country": "Europe"},
        "EURO": {"id": "676", "name": "UEFA European Football Championship", "country": "Europe"},
        "GOLDCUP": {"id": "681", "name": "CONCACAF Gold Cup", "country": "North America"},
        "COPAAMERICA": {"id": "685", "name": "CONMEBOL Copa América", "country": "South America"},
        
        # Cups & Other Competitions
        "FACUP": {"id": "514", "name": "FA Cup", "country": "England"},
        "EFLCUP": {"id": "690", "name": "EFL Cup", "country": "England"},
        "COPADELREY": {"id": "569", "name": "Copa del Rey", "country": "Spain"},
        "COPPAITALIA": {"id": "529", "name": "Coppa Italia", "country": "Italy"},
        "DFBPOKAL": {"id": "521", "name": "DFB-Pokal", "country": "Germany"},
        "COUPEFR": {"id": "518", "name": "Coupe de France", "country": "France"},
        
        # Additional Leagues
        "ELITESERIEN": {"id": "28", "name": "Eliteserien", "country": "Norway"},
        "SUPERLIG": {"id": "26", "name": "Süper Lig", "country": "Turkey"},
        "SUPERLIGA": {"id": "50", "name": "Danish Superliga", "country": "Denmark"},
        "EKSTRAKLASA": {"id": "36", "name": "Ekstraklasa", "country": "Poland"},
        "SUPERETTAN": {"id": "48", "name": "Superettan", "country": "Sweden"},
        "RUSSIANPL": {"id": "30", "name": "Russian Premier League", "country": "Russia"},
        "SAUDIPROLEAGUE": {"id": "70", "name": "Saudi Professional League", "country": "Saudi Arabia"},
        "SUPERLEAGUE": {"id": "27", "name": "Super League Greece", "country": "Greece"},
        "USLCHAMP": {"id": "73", "name": "USL Championship", "country": "United States"},
        
        # Add option for direct ID usage
        "CUSTOM": {"id": "", "name": "", "country": ""}
    }
    
    @classmethod
    def fetch_all_leagues_from_fbref(cls):
        """
        Fetch all available leagues directly from FBref
        
        Returns:
            Dictionary mapping league IDs to names
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import re
            
            url = "https://fbref.com/en/about/coverage"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to fetch coverage page: {response.status_code}")
                return {}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            leagues = {}
            # Find all links that might be competition links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/comps/' in href:
                    match = re.search(r'/comps/(\d+)/', href)
                    if match:
                        league_id = match.group(1)
                        name = link.text.strip()
                        if league_id.isdigit() and name:
                            leagues[league_id] = name
            
            logger.info(f"Fetched {len(leagues)} leagues from FBref")
            return leagues
            
        except Exception as e:
            logger.error(f"Error fetching leagues from FBref: {e}")
            return {}
    
    @classmethod
    def get_league_info(cls, league_code):
        """
        Get standardized league information from code
        
        Args:
            league_code: Standard league code (e.g., "EPL", "LALIGA")
            
        Returns:
            Dictionary with league information
            
        Raises:
            ValueError: If league code is not recognized
        """
        league_code = league_code.upper()
        if league_code not in cls.LEAGUE_MAP:
            valid_codes = ", ".join(cls.LEAGUE_MAP.keys())
            raise ValueError(f"Unknown league code: {league_code}. Valid codes: {valid_codes}")
        
        return cls.LEAGUE_MAP[league_code]
    
    @classmethod
    def get_league_url(cls, league_code):
        """
        Get FBref URL for a league
        
        Args:
            league_code: Standard league code
            
        Returns:
            League URL
        """
        league_info = cls.get_league_info(league_code)
        league_id = league_info["id"]
        league_name = league_info["name"].replace(" ", "-")
        
        return f"https://fbref.com/en/comps/{league_id}/{league_name}-Stats"
    
    @classmethod
    def is_valid_league(cls, league_code):
        """
        Check if a league code is valid
        
        Args:
            league_code: League code to check
            
        Returns:
            True if valid, False otherwise
        """
        return league_code.upper() in cls.LEAGUE_MAP
    
    @classmethod
    def get_all_leagues(cls):
        """
        Get list of all supported leagues
        
        Returns:
            List of dictionaries with league information
        """
        result = []
        for code, info in cls.LEAGUE_MAP.items():
            league_data = info.copy()
            league_data["code"] = code
            result.append(league_data)
        
        return result

class TeamParser:
    """
    Parser for team information, URLs, and match data.
    Handles the extraction of data from FBref HTML pages.
    """
    
    def __init__(self, requester):
        """
        Initialize team parser
        
        Args:
            requester: PoliteRequester instance for making web requests
        """
        self.requester = requester
    
    def extract_team_urls(self, league_html):
        """
        Extract team URLs from league page HTML
        
        Args:
            league_html: HTML content of the league page
            
        Returns:
            List of team URLs
            
        Raises:
            ParseError: If no team URLs could be found
        """
        try:
            soup = BeautifulSoup(league_html, 'html.parser')
            team_urls = []
            
            # First try: standard league table
            tables = soup.select('table.stats_table')
            
            if not tables:
                # Try alternative selectors
                tables = soup.select('#results, #stats_squads_standard_for')
            
            if tables:
                # Process the first table that looks like a standings table
                for table in tables:
                    links = table.select('a[href*="/squads/"]')
                    if links:
                        for link in links:
                            href = link.get('href')
                            if href and '/squads/' in href and href not in team_urls:
                                team_urls.append(f"https://fbref.com{href}")
                        
                        if team_urls:
                            # If we found links, stop processing tables
                            break
            
            # If still no team links found, try a more generic approach
            if not team_urls:
                all_links = soup.select('a[href*="/squads/"]')
                seen_urls = set()
                
                for link in all_links:
                    href = link.get('href', '')
                    if href and '/squads/' in href and 'Stats' in href and href not in seen_urls:
                        team_urls.append(f"https://fbref.com{href}")
                        seen_urls.add(href)
            
            if not team_urls:
                raise ParseError("No team URLs found in league page")
                
            logger.info(f"Found {len(team_urls)} teams")
            return team_urls
            
        except Exception as e:
            raise ParseError(f"Error extracting team URLs: {e}")
    
    def extract_team_name(self, team_url):
        """
        Extract team name from team URL
        
        Args:
            team_url: URL of the team page
            
        Returns:
            Team name
        """
        parts = team_url.split('/')
        for part in reversed(parts):
            if part and part != "Stats":
                return part.replace('-', ' ')
        
        # Last resort: use the last part of the URL
        return parts[-1].replace('-', ' ').replace('Stats', '').strip()
    
    def find_team_url_by_name(self, team_name, league_html):
        """
        Find a team's URL by name in the league page HTML
        
        Args:
            team_name: Name of the team to find
            league_html: HTML content of the league page
            
        Returns:
            Team URL or None if not found
        """
        try:
            soup = BeautifulSoup(league_html, 'html.parser')
            
            # Normalize team name for comparison
            team_name_normalized = team_name.lower().replace('-', ' ').strip()
            
            # Look for team links
            team_links = soup.select('a[href*="/squads/"]')
            
            for link in team_links:
                href = link.get('href', '')
                link_text = link.text.strip()
                
                # Convert URL slug to potential name
                url_parts = href.split('/')
                url_team_name = ''
                for part in reversed(url_parts):
                    if part and part not in ('', 'Stats'):
                        url_team_name = part.replace('-', ' ')
                        break
                
                # Check if either the link text or URL part matches
                if (link_text.lower() == team_name_normalized or 
                    url_team_name.lower() == team_name_normalized):
                    return f"https://fbref.com{href}"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding team URL for {team_name}: {e}")
            return None
    
    def parse_team_matches(self, team_url, league_id, league_name, max_matches=7):
        """
        Parse match data for a team
        
        Args:
            team_url: URL of the team page
            league_id: ID of the league
            league_name: Name of the league
            max_matches: Maximum number of matches to return
            
        Returns:
            DataFrame with match data
            
        Raises:
            ParseError: If parsing fails
            NetworkError: If team page couldn't be fetched
        """
        team_name = self.extract_team_name(team_url)
        
        try:
            # Get team page HTML
            team_html = self.requester.fetch(team_url)
            
            # Parse the team page
            soup = BeautifulSoup(team_html, 'html.parser')
            
            # Find the Scores & Fixtures table
            matches_table = None
            for table in soup.select('table'):
                if table.select('caption'):
                    caption_text = table.select_one('caption').text.strip()
                    if 'Scores & Fixtures' in caption_text:
                        matches_table = table
                        break
            
            if not matches_table:
                raise ParseError(f"Scores & Fixtures table not found for {team_name}")
            
            # Parse the table into a DataFrame
            matches_data = []
            
            # Get header row
            headers = []
            header_row = matches_table.select_one('thead tr')
            if header_row:
                for th in header_row.select('th'):
                    # Get the column name from data-stat attribute or text
                    col_name = th.get('data-stat', th.text.strip())
                    headers.append(col_name)
            
            #