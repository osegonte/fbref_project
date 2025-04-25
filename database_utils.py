"""Database utilities for FBref Toolkit with improved connection management."""
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Any, Generator, Tuple, Union

import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field, validator

from config import POSTGRES_URI

# Configure logging
logger = logging.getLogger("fbref_toolkit.database")

# SQLAlchemy engine singleton
_engine = None

def get_engine() -> Engine:
    """Get or create a SQLAlchemy engine with connection pooling.
    
    Returns:
        SQLAlchemy engine instance
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            POSTGRES_URI,
            future=True,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600  # Recycle connections older than 1 hour
        )
        logger.info(f"Created database engine with connection pooling")
    return _engine

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session using context manager for automatic cleanup.
    
    Yields:
        SQLAlchemy session
        
    Example:
        with get_session() as session:
            results = session.execute(text("SELECT * FROM matches"))
    """
    engine = get_engine()
    session_factory = sessionmaker(engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

# Data validation models
class Match(BaseModel):
    """Validation model for match data."""
    match_id: str
    date: str
    team: str
    opponent: str
    result: str
    gf: Optional[int] = None
    ga: Optional[int] = None
    venue: str
    is_home: bool
    points: Optional[int] = None
    xg: Optional[float] = None
    xga: Optional[float] = None
    comp: str
    season: int
    
    @validator('result')
    def validate_result(cls, v):
        if v not in ('W', 'D', 'L'):
            raise ValueError(f"Result must be W, D, or L, got {v}")
        return v
    
    @validator('gf', 'ga', 'points')
    def validate_numeric(cls, v, values, **kwargs):
        if v is not None and v < 0:
            raise ValueError(f"Value cannot be negative: {v}")
        return v

class Player(BaseModel):
    """Validation model for player data."""
    player_id: str
    player: str
    team: str
    nation: Optional[str] = None
    pos: Optional[str] = None
    age: Optional[float] = None
    minutes: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    season: int
    stats_type: str

# Tables definition
TABLES_SQL = {
    "matches": """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            date DATE,
            team TEXT,
            opponent TEXT,
            result TEXT,
            gf INTEGER,
            ga INTEGER,
            venue TEXT,
            is_home BOOLEAN,
            points INTEGER,
            sh INTEGER,
            sot INTEGER,
            corner_for INTEGER,
            corner_against INTEGER,
            poss NUMERIC,
            xg NUMERIC,
            xga NUMERIC,
            comp TEXT,
            season INTEGER,
            scrape_date DATE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "players": """
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            player TEXT,
            team TEXT,
            nation TEXT,
            pos TEXT,
            age NUMERIC,
            minutes INTEGER,
            goals INTEGER,
            assists INTEGER,
            season INTEGER,
            stats_type TEXT,
            scrape_date DATE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "league_tables": """
        CREATE TABLE IF NOT EXISTS league_tables (
            id SERIAL PRIMARY KEY,
            rank INTEGER,
            squad TEXT,
            matches_played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_diff INTEGER,
            points INTEGER,
            points_per_match NUMERIC,
            xg NUMERIC,
            xga NUMERIC,
            xg_diff NUMERIC,
            league TEXT,
            country TEXT,
            season INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "player_valuations": """
        CREATE TABLE IF NOT EXISTS player_valuations (
            id SERIAL PRIMARY KEY,
            player_id INTEGER,
            date DATE,
            value_eur INTEGER,
            club TEXT,
            age TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """
}

INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_matches_team ON matches (team)",
    "CREATE INDEX IF NOT EXISTS idx_matches_season ON matches (season)",
    "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (date)",
    "CREATE INDEX IF NOT EXISTS idx_players_team ON players (team)",
    "CREATE INDEX IF NOT EXISTS idx_league_tables_season ON league_tables (season)",
    "CREATE INDEX IF NOT EXISTS idx_league_tables_league ON league_tables (league)",
]

# Triggers for updated_at columns
TRIGGERS_SQL = """
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_matches_updated_at') THEN
        CREATE TRIGGER update_matches_updated_at
        BEFORE UPDATE ON matches
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_column();
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_players_updated_at') THEN
        CREATE TRIGGER update_players_updated_at
        BEFORE UPDATE ON players
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_column();
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_league_tables_updated_at') THEN
        CREATE TRIGGER update_league_tables_updated_at
        BEFORE UPDATE ON league_tables
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_column();
    END IF;
END
$$;
"""

def ensure_tables_exist() -> None:
    """Create database tables if they don't exist."""
    engine = get_engine()
    
    with engine.begin() as conn:
        # Create tables
        for table_name, table_sql in TABLES_SQL.items():
            conn.execute(text(table_sql))
            logger.debug(f"Ensured table exists: {table_name}")
        
        # Create indexes
        for index_sql in INDEXES_SQL:
            conn.execute(text(index_sql))
        
        # Create triggers for updated_at columns
        conn.execute(text(TRIGGERS_SQL))
    
    logger.info("Database schema initialized")

def validate_data(
    df: pd.DataFrame, 
    model_class: type, 
    drop_invalid: bool = False
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate DataFrame rows against a Pydantic model.
    
    Args:
        df: DataFrame to validate
        model_class: Pydantic model class to use for validation
        drop_invalid: If True, drop invalid rows; if False, raise exception
        
    Returns:
        Tuple of (validated_dataframe, list_of_validation_errors)
        
    Raises:
        ValueError: If drop_invalid is False and validation errors are found
    """
    if df.empty:
        return df, []
    
    errors = []
    valid_rows = []
    
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            # Validate row against model
            model_instance = model_class(**row_dict)
            # Add validated row to list
            valid_rows.append(model_instance.dict())
        except Exception as e:
            error = {
                "row_index": i,
                "error": str(e),
                "data": row_dict
            }
            errors.append(error)
    
    if errors and not drop_invalid:
        error_msg = f"Found {len(errors)} validation errors. First error: {errors[0]['error']}"
        raise ValueError(error_msg)
    
    if errors:
        logger.warning(f"Dropped {len(errors)} invalid rows during validation")
        
    # Create new DataFrame from valid rows
    validated_df = pd.DataFrame(valid_rows)
    return validated_df, errors

def upsert(
    df: pd.DataFrame, 
    table: str, 
    key_cols: Sequence[str],
    validate: bool = True,
    model_class: Optional[type] = None
) -> int:
    """Insert or update records in database table with validation.
    
    Args:
        df: DataFrame with data to upsert
        table: Target table name
        key_cols: List of column names that form the primary key
        validate: Whether to validate data before upserting
        model_class: Pydantic model class to use for validation
        
    Returns:
        Number of rows upserted
        
    Raises:
        ValueError: If validation fails and validate=True
    """
    if df.empty:
        logger.warning(f"Empty DataFrame provided for upsert to {table}")
        return 0
    
    # Determine appropriate validation model
    if validate and not model_class:
        if table == "matches":
            model_class = Match
        elif table == "players":
            model_class = Player
    
    # Validate data if requested
    if validate and model_class:
        df, errors = validate_data(df, model_class)
        if df.empty:
            logger.error(f"All rows failed validation for {table}")
            return 0
    
    # Clean DataFrame before upsert
    # Replace NaN with None for database compatibility
    df = df.replace({pd.NA: None})
    
    engine = get_engine()
    with engine.begin() as conn:
        tmp_table = f"tmp_{table}"
        
        # Create a temporary table
        df.to_sql(tmp_table, conn, index=False, if_exists="replace")
        
        cols = ", ".join(df.columns)
        keys = ", ".join(key_cols)
        # Only update non-key columns
        sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in df.columns if c not in key_cols)
        
        # Execute upsert
        if sets:  # Only if there are columns to update
            query = f"""
                INSERT INTO {table} ({cols}) 
                SELECT {cols} FROM {tmp_table}
                ON CONFLICT ({keys}) 
                DO UPDATE SET {sets};
                DROP TABLE {tmp_table};
            """
        else:
            # If only key columns are present, just do nothing on conflict
            query = f"""
                INSERT INTO {table} ({cols}) 
                SELECT {cols} FROM {tmp_table}
                ON CONFLICT ({keys}) DO NOTHING;
                DROP TABLE {tmp_table};
            """
            
        conn.execute(text(query))
        logger.info(f"Upserted {len(df)} rows to {table}")
        return len(df)

def execute_query(
    query: str, 
    params: Optional[Dict[str, Any]] = None, 
    as_dataframe: bool = True
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute a SQL query and return results.
    
    Args:
        query: SQL query to execute
        params: Optional parameters for the query
        as_dataframe: If True, return results as DataFrame; otherwise as list of dicts
    
    Returns:
        Query results as DataFrame or list of dictionaries
    """
    with get_session() as session:
        if params:
            result = session.execute(text(query), params)
        else:
            result = session.execute(text(query))
            
        if as_dataframe:
            # Convert to DataFrame
            columns = result.keys()
            return pd.DataFrame([dict(zip(columns, row)) for row in result])
        else:
            # Return as list of dicts
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result]

def get_table_info(table_name: str) -> Dict[str, Any]:
    """Get information about a table's structure.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Dictionary with table information
    """
    engine = get_engine()
    inspector = inspect(engine)
    
    if not inspector.has_table(table_name):
        return {"exists": False}
    
    # Get table columns
    columns = []
    for column in inspector.get_columns(table_name):
        columns.append({
            "name": column["name"],
            "type": str(column["type"]),
            "nullable": column["nullable"],
            "default": str(column["default"]) if column["default"] else None,
        })
    
    # Get primary keys
    pk = inspector.get_pk_constraint(table_name)
    
    # Get indexes
    indexes = []
    for index in inspector.get_indexes(table_name):
        indexes.append({
            "name": index["name"],
            "columns": index["column_names"],
            "unique": index["unique"],
        })
    
    # Count rows
    with get_session() as session:
        count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    
    return {
        "exists": True,
        "columns": columns,
        "primary_key": pk["constrained_columns"] if pk else [],
        "indexes": indexes,
        "row_count": count
    }

def export_data(
    table: str, 
    query: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    format: str = "csv",
    output_path: Optional[str] = None
) -> str:
    """Export data from database to file.
    
    Args:
        table: Table name to export
        query: Optional custom SQL query
        params: Optional parameters for the custom query
        format: Output format (csv, json, excel, parquet)
        output_path: Optional output file path
        
    Returns:
        Path to output file
    """
    # Get the data
    engine = get_engine()
    
    if query:
        if params:
            df = pd.read_sql(query, engine, params=params)
        else:
            df = pd.read_sql(query, engine)
    else:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
    
    if df.empty:
        logger.warning(f"No data found for export from {table}")
        return None
        
    # Generate default filename if not provided
    if not output_path:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/{table}_{timestamp}.{format}"
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to specified format
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", date_format="iso")
    elif format == "excel":
        df.to_excel(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {format}")
        
    logger.info(f"Exported {len(df)} rows to {output_path}")
    return output_path