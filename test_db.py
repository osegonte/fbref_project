from config import POSTGRES_URI
from sqlalchemy import create_engine, text

print(f"Connecting to: {POSTGRES_URI}")
engine = create_engine(POSTGRES_URI)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Connection successful!")
        
        # Check if tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        tables = [row[0] for row in result]
        print(f"Tables in database: {tables}")
        
except Exception as e:
    print(f"Connection failed: {e}")