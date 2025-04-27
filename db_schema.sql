-- Updated PostgreSQL schema for football data pipeline

-- Fixtures Table for storing upcoming matches
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id SERIAL PRIMARY KEY,
    match_id VARCHAR(100) UNIQUE,
    date DATE NOT NULL,
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    league VARCHAR(100),
    country VARCHAR(50),
    start_time VARCHAR(20),
    start_timestamp BIGINT,
    venue VARCHAR(100),
    status VARCHAR(50),
    source VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries on fixtures
CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date);
CREATE INDEX IF NOT EXISTS idx_fixtures_teams ON fixtures(home_team, away_team);

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

-- Add a table for match predictions
CREATE TABLE IF NOT EXISTS match_predictions (
    prediction_id SERIAL PRIMARY KEY,
    fixture_id INTEGER REFERENCES fixtures(fixture_id),
    home_win_prob FLOAT,
    draw_prob FLOAT,
    away_win_prob FLOAT,
    predicted_home_goals FLOAT,
    predicted_away_goals FLOAT,
    home_form_points INTEGER,
    away_form_points INTEGER,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    UNIQUE (fixture_id, model_version)
);

-- Add a table for pipeline runs to track execution
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id SERIAL PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'failed'
    fixtures_processed INTEGER,
    teams_processed INTEGER,
    matches_processed INTEGER,
    error_message TEXT,
    pipeline_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
    string_agg(result, '' ORDER BY date DESC) as form_string
FROM recent_form
WHERE match_num <= 5
GROUP BY team
ORDER BY points DESC, wins DESC;

-- 7. Find upcoming fixtures for a team
SELECT 
    f.date,
    f.home_team,
    f.away_team,
    f.league,
    f.start_time,
    f.venue
FROM fixtures f
WHERE (f.home_team = 'Liverpool' OR f.away_team = 'Liverpool')
AND f.date >= CURRENT_DATE
ORDER BY f.date ASC, f.start_time ASC;

-- 8. Get matches scheduled for a specific time period
SELECT 
    f.date,
    f.start_time,
    f.home_team,
    f.away_team,
    f.league,
    f.country
FROM fixtures f
WHERE f.date = CURRENT_DATE
AND (
    -- Morning matches (before 12:00)
    (f.start_time < '12:00') OR
    -- Afternoon matches (12:00 - 17:00)
    (f.start_time >= '12:00' AND f.start_time < '17:00') OR
    -- Evening matches (after 17:00)
    (f.start_time >= '17:00')
)
ORDER BY f.start_time ASC;