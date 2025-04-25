-- Form table query (last 5 matches per team)
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