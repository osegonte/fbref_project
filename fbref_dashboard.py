"""Interactive dashboard for FBref data."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Import functions from the main toolkit
from fbref_toolkit import (
    pg_engine, team_form_analysis, calculate_head_to_head_stats,
    treemap, bar_plot, line_plot, form_heatmap
)

# Set up page config
st.set_page_config(
    page_title="Football Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Football Analytics Dashboard")

# Sidebar navigation
page = st.sidebar.radio(
    "Select Page",
    ["Team Analysis", "League Tables", "Head-to-Head", "Player Stats"]
)

# Page content
if page == "Team Analysis":
    st.header("Team Analysis")
    
    # Connect to database
    eng = pg_engine()
    
    # Get list of teams
    with eng.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT team FROM matches ORDER BY team"))
        teams = [row[0] for row in result]
    
    # Team selector
    team = st.selectbox("Select Team", teams)
    
    if team:
        # Show team form
        form_df = team_form_analysis(team)
        
        if not form_df.empty:
            st.subheader(f"{team} - Recent Form")
            
            # Form metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points", form_df["points"].iloc[0])
            with col2:
                st.metric("Goals For", form_df["goals_for"].iloc[0])
            with col3:
                st.metric("Goals Against", form_df["goals_against"].iloc[0])
            with col4:
                st.metric("Form", form_df["form_string"].iloc[0])
            
            # Team matches
            with eng.connect() as conn:
                matches_query = text("""
                    SELECT date, opponent, venue, result, gf, ga 
                    FROM matches 
                    WHERE team = :team 
                    ORDER BY date DESC 
                    LIMIT 10
                """)
                result = conn.execute(matches_query, {"team": team})
                matches = [dict(row._mapping) for row in result]
            
            st.subheader("Recent Matches")
            st.table(matches)

elif page == "League Tables":
    st.header("League Tables")
    
    # Connect to database
    eng = pg_engine()
    
    # Get leagues
    with eng.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT league FROM league_tables"))
        leagues = [row[0] for row in result]
    
    league = st.selectbox("Select League", leagues)
    
    if league:
        with eng.connect() as conn:
            table_query = text("""
                SELECT rank, squad, matches_played, wins, draws, losses, 
                       goals_for, goals_against, goal_diff, points
                FROM league_tables
                WHERE league = :league
                ORDER BY rank
            """)
            result = conn.execute(table_query, {"league": league})
            table_data = [dict(row._mapping) for row in result]
        
        st.table(table_data)

# Add other pages as needed