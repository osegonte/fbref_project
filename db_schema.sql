-- PostgreSQL schema for FBref data storage

-- Recent Matches Table
CREATE TABLE IF NOT EXISTS recent_matches (
    match_id VARCHAR(100) PRIMARY KEY,
    date DATE NOT NULL,
    team VARCHAR(50) NOT NULL,
    opponent VARCHAR(50) NOT NULL,
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
    round VARCHAR(50),
    season INTEGER,
    is_home BOOLEAN,
    scrape_date TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_recent_matches_date ON recent_matches(date);
CREATE INDEX IF NOT EXISTS idx_recent_matches_team ON recent_matches(team);
CREATE INDEX IF NOT EXISTS idx_recent_matches_season ON recent_matches(season);

-- Teams Table for team information
CREATE TABLE IF NOT EXISTS teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(50) UNIQUE NOT NULL,
    country VARCHAR(50),
    league VARCHAR(50),
    logo_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players Table for player information
CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    team_id INTEGER REFERENCES teams(team_id),
    position VARCHAR(20),
    nationality VARCHAR(50),
    birth_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (player_name, team_id)
);

-- League Table for storing standings
CREATE TABLE IF NOT EXISTS league_table (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    season INTEGER NOT NULL,
    rank INTEGER,
    matches_played INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    goal_diff INTEGER,
    points INTEGER,
    points_per_match FLOAT,
    xg FLOAT,
    xga FLOAT,
    xg_diff FLOAT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (team_id, season)
);

-- Sample queries for common operations

-- 1. Get all recent matches for a specific team
SELECT * FROM recent_matches
WHERE team = 'Liverpool' 
ORDER BY date DESC
LIMIT 7;

-- 2. Get head-to-head results between two teams
SELECT * FROM recent_matches
WHERE (team = 'Arsenal' AND opponent = 'Chelsea')
   OR (team = 'Chelsea' AND opponent = 'Arsenal')
ORDER BY date DESC;

-- 3. Get team performance in home vs away matches
SELECT 
    team,
    venue,
    COUNT(*) as matches_played,
    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
    SUM(gf) as goals_for,
    SUM(ga) as goals_against,
    SUM(points) as total_points,
    ROUND(AVG(sh),2) as avg_shots,
    ROUND(AVG(sot),2) as avg_shots_on_target
FROM recent_matches
WHERE team = 'Manchester City'
GROUP BY team, venue;

-- 4. Find teams with the highest goal difference in recent matches
SELECT 
    team,
    COUNT(*) as matches_played,
    SUM(gf) as goals_for,
    SUM(ga) as goals_against,
    SUM(gf) - SUM(ga) as goal_diff,
    SUM(points) as total_points
FROM recent_matches
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY team
ORDER BY goal_diff DESC;

-- 5. Compare expected goals (xG) vs actual goals
SELECT 
    team,
    SUM(gf) as actual_goals,
    ROUND(SUM(xg)::numeric, 2) as expected_goals,
    ROUND((SUM(gf) - SUM(xg))::numeric, 2) as goals_vs_xg_diff
FROM recent_matches
GROUP BY team
ORDER BY goals_vs_xg_diff DESC;

-- 6. Get form table (based on last 5 matches per team)
WITH recent_form AS (
    SELECT 
        team,
        date,
        result,
        points,
        ROW_NUMBER() OVER (PARTITION BY team ORDER BY date DESC) as match_num
    FROM recent_matches
)
SELECT 
    team,
    COUNT(*) as matches,
    SUM(points) as points,
    SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses
FROM recent_form
WHERE match_num <= 5
GROUP BY team
ORDER BY points DESC, wins DESC;