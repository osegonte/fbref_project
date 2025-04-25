-- Head-to-head results between two teams
-- Replace 'Team1' and 'Team2' with actual team names
SELECT 
    date,
    team as home_team,
    opponent as away_team,
    gf as home_goals,
    ga as away_goals,
    result,
    CASE 
        WHEN result = 'W' THEN team
        WHEN result = 'L' THEN opponent
        ELSE 'Draw'
    END as winner
FROM recent_matches
WHERE (team = 'Team1' AND opponent = 'Team2')
   OR (team = 'Team2' AND opponent = 'Team1')
ORDER BY date DESC;

-- Usage example:
-- Replace 'Manchester City' and 'Liverpool' with the teams you want to analyze
-- SELECT 
--     date,
--     team as home_team,
--     opponent as away_team,
--     gf as home_goals,
--     ga as away_goals,
--     result,
--     CASE 
--         WHEN result = 'W' THEN team
--         WHEN result = 'L' THEN opponent
--         ELSE 'Draw'
--     END as winner
-- FROM recent_matches
-- WHERE (team = 'Manchester City' AND opponent = 'Liverpool')
--    OR (team = 'Liverpool' AND opponent = 'Manchester City')
-- ORDER BY date DESC;