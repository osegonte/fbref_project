-- Home vs Away performance for a team
-- Replace 'TeamName' with actual team name
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
WHERE team = 'TeamName'
GROUP BY venue
ORDER BY venue;

-- Usage example:
-- Replace 'Liverpool' with the team you want to analyze
-- SELECT 
--     venue,
--     COUNT(*) as matches_played,
--     SUM(CASE WHEN result = 'W' THEN 1 ELSE 0 END) as wins,
--     SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) as draws,
--     SUM(CASE WHEN result = 'L' THEN 1 ELSE 0 END) as losses,
--     SUM(gf) as goals_for,
--     SUM(ga) as goals_against,
--     SUM(gf) - SUM(ga) as goal_diff,
--     SUM(points) as total_points,
--     ROUND(AVG(points)::numeric, 2) as avg_points_per_game,
--     ROUND(AVG(sh)::numeric, 2) as avg_shots,
--     ROUND(AVG(sot)::numeric, 2) as avg_shots_on_target
-- FROM recent_matches
-- WHERE team = 'Liverpool'
-- GROUP BY venue
-- ORDER BY venue;