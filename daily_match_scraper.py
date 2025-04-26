import time
import csv
import json
import os
import random
from datetime import date, datetime, timedelta
import cloudscraper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import re

class AdvancedSofaScoreScraper:
    """
    Advanced scraper for SofaScore that uses multiple methods to bypass anti-bot protections
    """
    
    def __init__(self):
        # Create directories for saving data
        self.data_dir = "sofascore_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.daily_dir = os.path.join(self.data_dir, "daily")
        if not os.path.exists(self.daily_dir):
            os.makedirs(self.daily_dir)
            
        self.raw_dir = os.path.join(self.data_dir, "raw")
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        
        # Initialize cloudscraper
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            },
            delay=5,  # Delay solving the JavaScript challenge
            captcha={'provider': 'return_response'}  # Return the response if there's a CAPTCHA
        )
        
        # Initialize session cookies
        self.cookies = {}
        
        # Set up proxy list (replace with your actual proxies if needed)
        self.proxies = []
        # Example: self.proxies = [{'http': 'http://user:pass@proxy1.com:8000', 'https': 'https://user:pass@proxy1.com:8000'}]
    
    def get_random_headers(self):
        """Generate realistic browser headers"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "If-None-Match": f"W/\"{random.randint(10000, 9999999)}\"",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive"
        }
    
    def get_random_proxy(self):
        """Get a random proxy from the proxy list if available"""
        if not self.proxies:
            return None
        return random.choice(self.proxies)
    
    def initialize_browser_session(self):
        """Initialize a browser session and get cookies for API requests"""
        print("Initializing browser session to get cookies...")
        
        try:
            # Set up Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(f"user-agent={random.choice(self.get_random_headers()['User-Agent'])}")
            
            # Initialize Chrome driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Mask WebDriver to avoid detection
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Visit SofaScore homepage to get cookies
            driver.get("https://www.sofascore.com/")
            
            # Wait for the page to load
            time.sleep(5)
            
            # Extract cookies
            cookies = driver.get_cookies()
            
            # Close the browser
            driver.quit()
            
            # Add cookies to the session
            self.cookies = {cookie['name']: cookie['value'] for cookie in cookies}
            
            print(f"✓ Successfully obtained {len(cookies)} cookies")
            return True
            
        except Exception as e:
            print(f"✖ Error initializing browser session: {str(e)}")
            return False
    
    def fetch_events_via_api(self, target_date):
        """
        Attempt to fetch events through the API with cookies and cloudscraper
        
        Args:
            target_date: Date string in YYYY-MM-DD format
            
        Returns:
            List of event dictionaries or None if failed
        """
        # API endpoints to try
        endpoints = [
            f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{target_date}",
            f"https://api.sofascore.com/api/v1/sport/football/events/date/{target_date}",
            f"https://api.sofascore.com/api/v1/football/scheduled-events/date/{target_date}",
            f"https://api.sofascore.com/api/v1/events/date/{target_date}/sport/football",
        ]
        
        # Apply additional headers and cookies to the scraper
        self.scraper.headers.update(self.get_random_headers())
        
        for cookie_name, cookie_value in self.cookies.items():
            self.scraper.cookies.set(cookie_name, cookie_value)
        
        # Try each endpoint
        for endpoint in endpoints:
            try:
                # Get random proxy if available
                proxy = self.get_random_proxy()
                
                # Add a random delay to appear more human-like
                time.sleep(2 + random.random() * 3)
                
                print(f"Trying API endpoint: {endpoint}")
                
                # Make the request with cloudscraper
                response = self.scraper.get(
                    endpoint, 
                    proxies=proxy,
                    timeout=20
                )
                
                # Save response details for debugging
                debug_file = os.path.join(self.raw_dir, f"api_response_{target_date}.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Status: {response.status_code}\n")
                    f.write(f"Headers: {dict(response.headers)}\n")
                    f.write(f"Content: {response.text[:1000]}...\n")
                
                if response.status_code == 403:
                    print(f"  ⚠️ 403 Forbidden for {endpoint}")
                    continue
                
                if response.status_code != 200:
                    print(f"  ⚠️ Status code {response.status_code} for {endpoint}")
                    continue
                
                # Try to parse the JSON response
                try:
                    data = response.json()
                    
                    # Find events data in the response
                    events = None
                    if 'events' in data:
                        events = data['events']
                    elif 'scheduledEvents' in data:
                        events = data['scheduledEvents']
                    elif 'data' in data and isinstance(data['data'], list):
                        events = data['data']
                    
                    if events and len(events) > 0:
                        print(f"  ✓ Found {len(events)} events using API")
                        return events
                except:
                    print(f"  ⚠️ Failed to parse JSON from API response")
                    continue
            
            except Exception as e:
                print(f"  ⚠️ Error with {endpoint}: {str(e)}")
                continue
        
        print(f"  ✖ Failed to fetch events via API for {target_date}")
        return None
    
    def fetch_events_via_browser(self, target_date):
        """
        Fetch events directly using a headless browser as a fallback method
        
        Args:
            target_date: Date string in YYYY-MM-DD format
            
        Returns:
            List of event dictionaries or None if failed
        """
        print(f"Attempting to fetch events via browser for {target_date}...")
        
        try:
            # Set up Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(f"user-agent={random.choice(self.get_random_headers()['User-Agent'])}")
            
            # Initialize Chrome driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Mask WebDriver to avoid detection
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Format the date for URL
            url = f"https://www.sofascore.com/football/{target_date}"
            print(f"  Opening URL: {url}")
            
            # Navigate to the football page for the target date
            driver.get(url)
            
            # Wait for the page to load and events to appear
            wait = WebDriverWait(driver, 30)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            
            # Give extra time for JavaScript to load all events
            time.sleep(10)
            
            # Save the page source for debugging
            page_source = driver.page_source
            source_file = os.path.join(self.raw_dir, f"page_source_{target_date}.html")
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(page_source)
            
            # Try to extract event data from the window.__INITIAL_STATE__ variable
            script = """
            try {
                if (window.__INITIAL_STATE__ && window.__INITIAL_STATE__.events) {
                    return JSON.stringify(window.__INITIAL_STATE__.events);
                }
                if (window.__NEXT_DATA__ && window.__NEXT_DATA__.props && window.__NEXT_DATA__.props.pageProps) {
                    return JSON.stringify(window.__NEXT_DATA__.props.pageProps);
                }
                return null;
            } catch (e) {
                return "Error: " + e.toString();
            }
            """
            
            result = driver.execute_script(script)
            
            # Close the driver
            driver.quit()
            
            if result and not result.startswith("Error:"):
                try:
                    # Parse the JSON data
                    json_data = json.loads(result)
                    
                    # Save the raw data
                    raw_file = os.path.join(self.raw_dir, f"browser_data_{target_date}.json")
                    with open(raw_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
                    
                    # Extract events
                    events = []
                    if isinstance(json_data, list):
                        events = json_data
                    elif isinstance(json_data, dict) and 'events' in json_data:
                        events = json_data['events']
                    
                    if events and len(events) > 0:
                        print(f"  ✓ Found {len(events)} events using browser")
                        return events
                    else:
                        print("  ⚠️ No events found in browser data")
                except Exception as parse_error:
                    print(f"  ⚠️ Error parsing browser data: {str(parse_error)}")
            else:
                print(f"  ⚠️ Failed to extract data from browser: {result}")
            
            # If we didn't get events from JavaScript state, try to parse events from the DOM
            # This is a more complex task and would require parsing the HTML structure
            # which can be added as another fallback mechanism
            
            return None
            
        except Exception as e:
            print(f"  ✖ Error using browser method: {str(e)}")
            return None
    
    def try_fbref_fallback(self, target_date):
        """
        Try to get match data from FBref as a fallback
        
        Args:
            target_date: Date string in YYYY-MM-DD format
            
        Returns:
            List of match dictionaries or empty list if failed
        """
        print(f"Attempting to fetch matches from FBref for {target_date}...")
        
        try:
            # Get the season based on the date
            date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
            
            if date_obj.month >= 8:  # August or later
                season_start = date_obj.year
            else:
                season_start = date_obj.year - 1
                
            season_end = season_start + 1
            season = f"{season_start}-{season_end}"
            
            # Major leagues to fetch
            leagues = [
                {"name": "Premier League", "url": f"https://fbref.com/en/comps/9/{season}/schedule/Premier-League-Scores-and-Fixtures"},
                {"name": "La Liga", "url": f"https://fbref.com/en/comps/12/{season}/schedule/La-Liga-Scores-and-Fixtures"},
                {"name": "Bundesliga", "url": f"https://fbref.com/en/comps/20/{season}/schedule/Bundesliga-Scores-and-Fixtures"},
                {"name": "Serie A", "url": f"https://fbref.com/en/comps/11/{season}/schedule/Serie-A-Scores-and-Fixtures"},
                {"name": "Ligue 1", "url": f"https://fbref.com/en/comps/13/{season}/schedule/Ligue-1-Scores-and-Fixtures"},
                {"name": "Champions League", "url": f"https://fbref.com/en/comps/8/{season}/schedule/Champions-League-Scores-and-Fixtures"}
            ]
            
            all_matches = []
            
            # Format the target date for comparison
            target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
            
            for league in leagues:
                try:
                    print(f"  Fetching {league['name']} schedule...")
                    
                    # Use pandas to read the HTML table
                    tables = pd.read_html(league['url'])
                    
                    if tables and len(tables) > 0:
                        schedule_df = tables[0]
                        
                        # Find matches for the target date
                        if 'Date' in schedule_df.columns:
                            for index, row in schedule_df.iterrows():
                                try:
                                    # Parse the date from FBref format
                                    date_str = str(row['Date'])
                                    if pd.isna(date_str) or date_str == 'nan':
                                        continue
                                    
                                    # FBref date format is typically 'YYYY-MM-DD'
                                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                                        match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                                    elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                                        match_date = datetime.strptime(date_str, "%m/%d/%Y").date()
                                    else:
                                        continue
                                    
                                    # Check if the match is on the target date
                                    if match_date == target_date_obj:
                                        # Extract match details
                                        home_team = row.get('Home', 'Unknown')
                                        away_team = row.get('Away', 'Unknown')
                                        
                                        # Extract time if available
                                        time_str = row.get('Time', 'Unknown')
                                        
                                        # Create match object
                                        match = {
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'league': league['name'],
                                            'time': time_str,
                                            'date': target_date,
                                            'source': 'fbref'
                                        }
                                        
                                        all_matches.append(match)
                                except Exception as row_error:
                                    print(f"  ⚠️ Error processing row: {str(row_error)}")
                                    continue
                    else:
                        print(f"  ⚠️ No data tables found for {league['name']}")
                        
                except Exception as league_error:
                    print(f"  ⚠️ Error fetching {league['name']}: {str(league_error)}")
                    continue
                    
                # Add delay between league requests
                time.sleep(2)
            
            if all_matches:
                print(f"  ✓ Found {len(all_matches)} matches from FBref")
                return all_matches
            else:
                print(f"  ⚠️ No matches found from FBref for {target_date}")
                return []
                
        except Exception as e:
            print(f"  ✖ Error using FBref fallback: {str(e)}")
            return []
    
    def parse_events(self, events, source="api"):
        """
        Parse events from SofaScore API into a standardized format
        
        Args:
            events: List of event dictionaries from the API
            source: Source of the events (api or browser)
            
        Returns:
            List of standardized match dictionaries
        """
        if not events:
            return []
            
        matches = []
        
        for event in events:
            try:
                # Extract home team
                home_team_name = None
                if 'homeTeam' in event and 'name' in event['homeTeam']:
                    home_team_name = event['homeTeam']['name']
                elif 'home' in event and 'name' in event['home']:
                    home_team_name = event['home']['name']
                
                # Extract away team
                away_team_name = None
                if 'awayTeam' in event and 'name' in event['awayTeam']:
                    away_team_name = event['awayTeam']['name']
                elif 'away' in event and 'name' in event['away']:
                    away_team_name = event['away']['name']
                
                # Extract tournament/league
                tournament_name = 'Unknown League'
                if 'tournament' in event and 'name' in event['tournament']:
                    tournament_name = event['tournament']['name']
                elif 'category' in event and 'name' in event['category']:
                    tournament_name = event['category']['name']
                elif 'league' in event and 'name' in event['league']:
                    tournament_name = event['league']['name']
                
                # Skip if we don't have both team names
                if not home_team_name or not away_team_name:
                    continue
                
                # Extract tournament country/region
                country = 'International'
                if 'tournament' in event and 'category' in event['tournament'] and 'name' in event['tournament']['category']:
                    country = event['tournament']['category']['name']
                elif 'category' in event and 'name' in event['category']:
                    country = event['category']['name']
                
                # Extract start time
                start_time = None
                start_time_formatted = 'Unknown'
                if 'startTimestamp' in event:
                    start_time = event['startTimestamp']
                    try:
                        dt = datetime.fromtimestamp(start_time)
                        start_time_formatted = dt.strftime('%H:%M')
                    except:
                        pass
                
                # Extract event ID
                event_id = event.get('id', 'unknown')
                
                # Extract status
                status = 'Unknown'
                if 'status' in event:
                    if isinstance(event['status'], dict) and 'description' in event['status']:
                        status = event['status']['description']
                    elif isinstance(event['status'], str):
                        status = event['status']
                
                # Create standardized match object
                match = {
                    'id': event_id,
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'league': tournament_name,
                    'country': country,
                    'start_timestamp': start_time,
                    'start_time': start_time_formatted,
                    'status': status,
                    'source': source
                }
                
                # Add any additional data if available
                if 'venue' in event:
                    if isinstance(event['venue'], dict) and 'name' in event['venue']:
                        match['venue'] = event['venue']['name']
                    elif isinstance(event['venue'], str):
                        match['venue'] = event['venue']
                
                if 'roundInfo' in event:
                    if isinstance(event['roundInfo'], dict) and 'round' in event['roundInfo']:
                        match['round'] = event['roundInfo']['round']
                
                matches.append(match)
                
            except Exception as e:
                print(f"  ⚠️ Error parsing event: {str(e)}")
                continue
        
        return matches
    
    def save_matches_to_csv(self, matches, filename):
        """Save matches to a CSV file"""
        if not matches:
            print(f"No matches to save to {filename}")
            return
        
        fieldnames = [
            'id', 'home_team', 'away_team', 'league', 'country',
            'start_timestamp', 'start_time', 'status', 'venue', 'round', 'source'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for match in matches:
                writer.writerow(match)
        
        print(f"✓ Saved {len(matches)} matches to {filename}")
    
    def fetch_matches_for_date_range(self, start_date, end_date):
        """
        Fetch all football matches for a date range using multiple methods
        
        Args:
            start_date: Start date (datetime.date or YYYY-MM-DD string)
            end_date: End date (datetime.date or YYYY-MM-DD string)
            
        Returns:
            Dictionary mapping date strings to lists of match dictionaries
        """
        # Convert string dates to datetime.date if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Generate list of dates in the range
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)
        
        # Initialize browser session to get cookies
        self.initialize_browser_session()
        
        all_matches_by_date = {}
        total_matches = 0
        
        # Process each date
        for current_date in date_list:
            date_str = current_date.strftime("%Y-%m-%d")
            
            print(f"\nProcessing date: {date_str}")
            
            # Try all methods in sequence until one works
            events = None
            source = None
            
            # Method 1: Try API with cloudscraper
            events = self.fetch_events_via_api(date_str)
            if events:
                source = "api"
            
            # Method 2: If API fails, try browser method
            if not events:
                events = self.fetch_events_via_browser(date_str)
                if events:
                    source = "browser"
            
            # Parse events if we got them from SofaScore
            matches = []
            if events:
                matches = self.parse_events(events, source)
            
            # Method 3: If SofaScore methods fail, try FBref
            if not matches:
                matches = self.try_fbref_fallback(date_str)
            
            if matches:
                # Save matches for this date
                date_file = os.path.join(self.daily_dir, f"matches_{date_str}.csv")
                self.save_matches_to_csv(matches, date_file)
                
                # Add to overall collection
                all_matches_by_date[date_str] = matches
                total_matches += len(matches)
                
                print(f"  ✓ Processed {len(matches)} matches for {date_str}")
            else:
                print(f"  ✖ No matches found for {date_str} with any method")
        
        # Save all matches to a single CSV
        all_matches = []
        for date_str, matches in all_matches_by_date.items():
            for match in matches:
                match['date'] = date_str
                all_matches.append(match)
        
        if all_matches:
            all_file = os.path.join(self.data_dir, f"all_matches_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv")
            
            # Extended fieldnames for combined file
            fieldnames = [
                'date', 'id', 'home_team', 'away_team', 'league', 'country',
                'start_timestamp', 'start_time', 'status', 'venue', 'round', 'source'
            ]
            
            with open(all_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for match in all_matches:
                    writer.writerow(match)
            
            print(f"\n✓ Saved {len(all_matches)} total matches to {all_file}")
        
        return all_matches_by_date, total_matches
    
    def print_statistics(self, all_matches_by_date):
        """Print statistics about fetched matches"""
        if not all_matches_by_date:
            print("No matches to analyze")
            return
        
        # Flatten all matches
        all_matches = []
        for matches in all_matches_by_date.values():
            all_matches.extend(matches)
        
        if not all_matches:
            print("No matches to analyze")
            return
        
        # Group by source
        sources = {}
        for match in all_matches:
            source = match.get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(match)
        
        # Group by league
        leagues = {}
        for match in all_matches:
            league = match['league']
            if league not in leagues:
                leagues[league] = []
            leagues[league].append(match)
        
        # Group by country
        countries = {}
        for match in all_matches:
            country = match.get('country', 'Unknown')
            if country not in countries:
                countries[country] = []
            countries[country].append(match)
        
        # Print summary
        print("\n=== Match Statistics ===")
        print(f"Total Matches: {len(all_matches)}")
        print(f"Date Range: {min(all_matches_by_date.keys())} to {max(all_matches_by_date.keys())}")
        print(f"Days with Matches: {len(all_matches_by_date)}")
        print(f"Total Leagues: {len(leagues)}")
        print(f"Total Countries/Regions: {len(countries)}")
        
        # Print matches by source
        print("\nMatches by Source:")
        for source, matches in sources.items():
            print(f"  • {source}: {len(matches)} matches")
        
        # Print matches per day
        print("\nMatches per Day:")
        for date_str, matches in sorted(all_matches_by_date.items()):
            print(f"  • {date_str}: {len(matches)} matches")
        
        # Print top leagues
        print("\nTop 10 Leagues by Match Count:")
        top_leagues = sorted(leagues.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for league, matches in top_leagues:
            print(f"  • {league}: {len(matches)} matches")
        
        # Print top countries
        print("\nTop 10 Countries/Regions by Match Count:")
        top_countries = sorted(countries.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for country, matches in top_countries:
            print(f"  • {country}: {len(matches)} matches")

def main():
    """Main function to fetch and process matches"""
    print("=== Advanced SofaScore Football Match Scraper ===")
    
    # Initialize scraper
    scraper = AdvancedSofaScoreScraper()
    
    # Calculate date range for next week
    today = date.today()
    end_date = today + timedelta(days=7)
    
    print(f"Fetching matches from {today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Using multiple fallback methods: API → Browser → FBref")
    
    try:
        # Fetch matches for next week
        all_matches, total_matches = scraper.fetch_matches_for_date_range(today, end_date)
        
        if total_matches > 0:
            # Print statistics
            scraper.print_statistics(all_matches)
            
            print(f"\n✓ Successfully fetched {total_matches} matches for the next week")
            print(f"  Data saved to {scraper.data_dir} directory")
        else:
            print("\n✖ Failed to fetch any matches")
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
    except Exception as e:
        print(f"\n✖ Error: {str(e)}")

if __name__ == "__main__":
    main()