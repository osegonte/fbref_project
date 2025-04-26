#!/usr/bin/env python3
"""
Universal FBref Scraper (CSV-only, match-by-match)

Fetches the last N matches for each requested team, with full detail:
- match_id, date, team, opponent, venue, result, comp, round, season, is_home
- gf, ga, points, xg, xga
- sh, sot, dist, fk, pk, pkatt
- possession, corners_for, corners_against
- league_id, league_name
"""

import os
import sys
import time
import random
import logging
import json
import argparse
import yaml
from datetime import datetime
from io import StringIO
from typing import Dict, Any, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import config  # Contains SCRAPER_CONFIG for defaults

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

# User-agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
]


class RateLimitHandler:
    def __init__(self, min_delay=10, max_delay=20, cooldown_threshold=3):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cooldown_threshold = cooldown_threshold
        self.rate_limited_count = 0
        self.last_request = 0
        self.domain_cooldown_until = 0

    def wait(self):
        now = time.time()
        if now < self.domain_cooldown_until:
            to_wait = self.domain_cooldown_until - now
            logger.info(f"Domain cooldown: sleeping {to_wait:.1f}s")
            time.sleep(to_wait)
            now = time.time()
        base = random.uniform(self.min_delay, self.max_delay)
        if self.rate_limited_count:
            base += min(self.rate_limited_count * 5, 30)
        elapsed = now - self.last_request
        if elapsed < base:
            time.sleep(base - elapsed)
        self.last_request = time.time()

    def backoff(self):
        self.rate_limited_count += 1
        backoff = min(2 ** self.rate_limited_count, 64) * random.uniform(0.8, 1.2)
        if self.rate_limited_count >= self.cooldown_threshold:
            cd = random.uniform(60, 120)
            self.domain_cooldown_until = time.time() + cd
            logger.warning(f"Entering domain cooldown for {cd:.1f}s")
        logger.warning(f"Rate limited; backing off {backoff:.1f}s")
        time.sleep(backoff)


class PoliteRequester:
    def __init__(self,
                 cache_dir: str = "data/cache",
                 cache_max_age: int = 24,
                 min_delay: int = 10,
                 max_delay: int = 20):
        self.cache_dir = cache_dir
        self.cache_max_age = cache_max_age
        self.rate_handler = RateLimitHandler(min_delay, max_delay)
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()

    def _cache_path(self, url: str) -> str:
        from hashlib import md5
        h = md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.html")

    def fetch(self, url: str, use_cache: bool = True, retries: int = 5) -> str:
        path = self._cache_path(url)
        if use_cache and os.path.exists(path):
            age = (time.time() - os.path.getmtime(path)) / 3600
            if age < self.cache_max_age:
                return open(path, encoding='utf-8').read()

        for attempt in range(retries):
            self.rate_handler.wait()
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            try:
                r = self.session.get(url, headers=headers, timeout=30)
                if r.status_code == 429:
                    self.rate_handler.backoff()
                    continue
                r.raise_for_status()
                txt = r.text
                if use_cache:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(txt)
                return txt
            except requests.RequestException as e:
                logger.warning(f"Fetch attempt {attempt+1}/{retries} failed: {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


class LeagueManager:
    LEAGUE_MAP = {
        "EPL":        {"id": "9",  "name": "Premier League"},
        "LALIGA":     {"id": "12", "name": "La Liga"},
        "BUNDESLIGA": {"id": "20", "name": "Bundesliga"},
        # ... add more static leagues as needed ...
    }

    @classmethod
    def fetch_all(cls) -> Dict[str, str]:
        url = "https://fbref.com/en/about/coverage"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        leagues = {}
        for a in soup.select('a[href*="/comps/"]'):
            parts = a['href'].split('/')
            if len(parts) > 3 and parts[2] == 'comps' and parts[3].isdigit():
                leagues[parts[3]] = a.text.strip()
        return leagues

    @classmethod
    def get_info(cls,
                 code: Optional[str] = None,
                 custom_id: Optional[str] = None,
                 custom_name: Optional[str] = None) -> Dict[str, str]:
        # 1) Try static map
        if code:
            key = code.upper()
            if key in cls.LEAGUE_MAP:
                logger.info(f"Found league in static map: {key}")
                return cls.LEAGUE_MAP[key]
            
            # 2) Dynamic fetch - only if not found in static map
            try:
                all_leagues = cls.fetch_all()
                if key in all_leagues:
                    return {"id": key, "name": all_leagues[key]}
                for lid, lname in all_leagues.items():
                    if lname.upper() == key:
                        return {"id": lid, "name": lname}
            except Exception as e:
                logger.error(f"Error fetching leagues: {e}")
                pass
                
        # 3) Custom override
        if custom_id and custom_name:
            return {"id": custom_id, "name": custom_name}
        elif custom_id:
            # Try to get name from league ID
            try:
                all_leagues = cls.fetch_all()
                if custom_id in all_leagues:
                    return {"id": custom_id, "name": all_leagues[custom_id]}
            except Exception:
                pass
            return {"id": custom_id, "name": f"League {custom_id}"}

        # Debug output to help troubleshoot
        if code:
            logger.error(f"League code '{code}' not found in static map: {cls.LEAGUE_MAP.keys()}")
        if custom_id:
            logger.error(f"League ID '{custom_id}' not recognized")
            
        raise ValueError("League not recognized. Use --fetch-all-leagues to view available IDs.")


class TeamParser:
    def __init__(self, requester: PoliteRequester):
        self.req = requester

    def parse_recent(self,
                     league_id: str,
                     league_name: str,
                     team: str,
                     lookback: int = 7) -> pd.DataFrame:
        # Build the Scores & Fixtures URL
        pretty = league_name.replace(" ", "-")
        sched_url = (
            f"https://fbref.com/en/comps/{league_id}/"
            f"schedule/{pretty}-Scores-and-Fixtures"
        )

        html = self.req.fetch(sched_url)
        raw = pd.read_html(StringIO(html))[0]

        # Normalize columns
        df = raw.rename(columns={
            "Date": "date",
            "Home": "home",
            "Away": "away",
            "Score": "score",
            "xG": "xg",
            "xGA": "xga",
            "Sh": "sh",
            "SoT": "sot",
            "Dist": "dist",
            "FK": "fk",
            "PK": "pk",
            "PKatt": "pkatt",
            "Poss": "possession",
        })

        is_home = df["home"] == team
        is_away = df["away"] == team
        df = df[is_home | is_away].copy()

        # Compute the fields you listed
        df["team"]     = team
        df["opponent"] = df["away"].where(is_home, df["home"])
        df["is_home"]  = is_home
        df["venue"]    = df["is_home"].map({True: "Home", False: "Away"})
        goals = df["score"].str.split("–", expand=True)
        gf = goals[0].astype(int)
        ga = goals[1].astype(int)
        df["gf"]     = gf.where(is_home, ga)
        df["ga"]     = ga.where(is_home, gf)
        df["result"] = [
            "W" if g1 > g2 else "D" if g1 == g2 else "L"
            for g1, g2 in zip(df["gf"], df["ga"])
        ]
        df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})

        # Metadata
        df["comp"]        = league_name
        df["season"]      = pd.to_datetime(df["date"]).dt.year
        df["league_id"]   = league_id
        df["league_name"] = league_name
        df["date"]        = pd.to_datetime(df["date"])
        df["match_id"]    = (
            df["date"].dt.strftime("%Y%m%d")
            + "_"
            + df["team"].str.replace(r"\W+", "", regex=True)
            + "_"
            + df["opponent"].str.replace(r"\W+", "", regex=True)
        )

        return df.sort_values("date").iloc[-lookback:].reset_index(drop=True)


class UniversalScraper:
    def __init__(self,
                 cache_dir: str = "data/cache",
                 output_dir: str = "data/leagues",
                 min_delay: int = 10,
                 max_delay: int = 20,
                 cache_max_age: int = 24):
        self.req    = PoliteRequester(cache_dir, cache_max_age, min_delay, max_delay)
        self.parser = TeamParser(self.req)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Running task with parameters: {task}")
        
        info = LeagueManager.get_info(
            code=task.get("league"),
            custom_id=task.get("league_id"),
            custom_name=task.get("league_name"),
        )
        
        league_id   = info["id"]
        league_name = info["name"]
        lookback    = task.get("lookback", 7)
        pairs       = task.get("teams", [])
        
        logger.info(f"Using league: {league_name} (ID: {league_id})")

        dfs, failures = [], []
        for home, away in pairs:
            for team in (home, away):
                try:
                    logger.info(f"Fetching data for {team}")
                    df = self.parser.parse_recent(league_id, league_name, team, lookback)
                    dfs.append(df)
                    logger.info(f"{team}: {len(df)} matches")
                except Exception as e:
                    logger.error(f"Failed to get data for {team}: {e}")
                    failures.append({"team": team, "error": str(e)})

        if not dfs:
            raise RuntimeError("No data scraped for any team")

        combined = pd.concat(dfs, ignore_index=True)
        league_dir = os.path.join(self.output_dir, league_id)
        os.makedirs(league_dir, exist_ok=True)
        fname = f"{league_id}_{datetime.now().strftime('%Y%m%d')}_matches.csv"
        path  = os.path.join(league_dir, fname)
        combined.to_csv(path, index=False)
        logger.info(f"Saved {len(combined)} rows to {path}")

        return {"rows": len(combined), "csv": path, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description="FBref CSV-only scraper")
    grp    = parser.add_mutually_exclusive_group()
    grp.add_argument("--league",    help="Predefined league code")
    grp.add_argument("--league-id", help="Custom league ID")
    parser.add_argument("--league-name", help="Name for custom league")
    parser.add_argument("--lookback",   type=int, default=7)
    parser.add_argument("--pairs",      nargs="+", help="Home,Away pairs")
    parser.add_argument("--config",     help="JSON/YAML task file")
    parser.add_argument("--fetch-all-leagues", action="store_true")
    args = parser.parse_args()

    # Fallback to config.py if no flags given
    if not any([args.league, args.league_id, args.config, args.fetch_all_leagues]):
        base           = config.SCRAPER_CONFIG["base_url"]
        parts          = base.split("/")
        args.league_id = parts[5]
        args.league_name = parts[6].replace("-Stats", "").replace("-", " ")
        args.lookback  = config.SCRAPER_CONFIG["matches_to_keep"]
        print(f"Using default config: league_id={args.league_id}, lookback={args.lookback}")

    if args.fetch_all_leagues:
        print(json.dumps(LeagueManager.fetch_all(), indent=2))
        sys.exit(0)

    if args.config:
        txt  = open(args.config).read()
        task = (yaml.safe_load(txt) if args.config.endswith((".yml", ".yaml"))
                else json.loads(txt))
    else:
        task = {
            "league":       args.league,
            "league_id":    args.league_id,
            "league_name":  args.league_name,
            "lookback":     args.lookback,
        }
        if args.pairs:
            task["teams"] = [p.split(",") for p in args.pairs]

    scraper = UniversalScraper()
    print(f"Starting scrape for {task.get('league') or task.get('league_name') or task.get('league_id')}...")
    result = scraper.run(task)
    print(f"Done. Rows: {result['rows']}  CSV: {result['csv']}")
    if result["failures"]:
        print(f"Failures: {len(result['failures'])}")
        for f in result["failures"]:
            print(f)


if __name__ == "__main__":
    main()