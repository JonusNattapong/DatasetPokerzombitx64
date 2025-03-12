"""
Database connector module for storing and retrieving poker data.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, List, Union, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, Text, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()


class PokerHand(Base):
    """SQLAlchemy model for poker hands table."""
    
    __tablename__ = 'poker_hands'
    
    id = Column(Integer, primary_key=True)
    buyin = Column(String)
    tourn_id = Column(Integer)
    table = Column(String)
    hand_id = Column(Integer)
    date = Column(Date)
    time = Column(String)
    table_size = Column(Integer)
    level = Column(Integer)
    playing = Column(Integer)
    seat = Column(Integer)
    name = Column(String)
    stack = Column(Float)
    position = Column(String)
    action_pre = Column(String)
    action_flop = Column(String)
    action_turn = Column(String)
    action_river = Column(String)
    all_in = Column(Boolean)
    cards = Column(String)
    board_flop = Column(String)
    board_turn = Column(String)
    board_river = Column(String)
    combination = Column(String)
    pot_pre = Column(Float)
    pot_flop = Column(Float)
    pot_turn = Column(Float)
    pot_river = Column(Float)
    ante = Column(Float)
    blinds = Column(Float)
    bet_pre = Column(Float)
    bet_flop = Column(Float)
    bet_turn = Column(Float)
    bet_river = Column(Float)
    result = Column(String)
    balance = Column(Float)


class DBConnector:
    """Class for database operations with poker data."""
    
    def __init__(self, db_path: str = 'poker_data.db'):
        """
        Initialize database connector.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """Establish connection to the database."""
        try:
            # Create engine
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def store_dataframe(self, df: pd.DataFrame, table_name: str = 'poker_hands', 
                        if_exists: str = 'append', index: bool = False) -> int:
        """
        Store a DataFrame in the database.
        
        Args:
            df: DataFrame to store
            table_name: Name of the table to store the data in
            if_exists: How to behave if the table exists ('fail', 'replace', or 'append')
            index: Whether to store the DataFrame index
            
        Returns:
            Number of rows stored
        """
        try:
            # Convert date column to datetime if it exists
            if 'date' in df.columns and df['date'].dtype == object:
                df['date'] = pd.to_datetime(df['date'])
            
            # Store DataFrame in database
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            
            logger.info(f"Stored {len(df)} rows in table: {table_name}")
            return len(df)
        except Exception as e:
            logger.error(f"Failed to store data in table {table_name}: {str(e)}")
            raise
    
    def query_dataframe(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing query results
        """
        try:
            result = pd.read_sql_query(query, self.engine)
            logger.info(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def get_player_data(self, player_name: str) -> pd.DataFrame:
        """
        Get all hands for a specific player.
        
        Args:
            player_name: Name of the player to get hands for
            
        Returns:
            DataFrame containing player's hands
        """
        query = f"SELECT * FROM poker_hands WHERE name = '{player_name}'"
        return self.query_dataframe(query)
    
    def get_player_stats(self, min_hands: int = 10) -> pd.DataFrame:
        """
        Get statistics for all players.
        
        Args:
            min_hands: Minimum number of hands played to include a player
            
        Returns:
            DataFrame containing player statistics
        """
        query = f"""
        SELECT 
            name, 
            COUNT(*) as hands_played,
            SUM(balance) as total_balance,
            AVG(balance) as avg_balance
        FROM 
            poker_hands
        GROUP BY 
            name
        HAVING 
            hands_played >= {min_hands}
        ORDER BY 
            avg_balance DESC
        """
        return self.query_dataframe(query)
    
    def get_position_stats(self, player_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get statistics by position.
        
        Args:
            player_name: Optional player name to filter by
            
        Returns:
            DataFrame containing position statistics
        """
        where_clause = f"WHERE name = '{player_name}'" if player_name else ""
        query = f"""
        SELECT 
            position, 
            COUNT(*) as hands_played,
            SUM(balance) as total_balance,
            AVG(balance) as avg_balance
        FROM 
            poker_hands
        {where_clause}
        GROUP BY 
            position
        ORDER BY 
            position
        """
        return self.query_dataframe(query)
    
    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info(f"Closed connection to database: {self.db_path}")


def import_csv_to_db(csv_path: str, db_path: str, table_name: str = 'poker_hands', 
                     if_exists: str = 'replace') -> int:
    """
    Import a CSV file into the database.
    
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the SQLite database file
        table_name: Name of the table to store the data in
        if_exists: How to behave if the table exists ('fail', 'replace', or 'append')
        
    Returns:
        Number of rows imported
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Initialize connector
        connector = DBConnector(db_path)
        
        # Store data
        rows_stored = connector.store_dataframe(df, table_name, if_exists)
        
        # Close connection
        connector.close()
        
        return rows_stored
    except Exception as e:
        logger.error(f"Failed to import CSV to database: {str(e)}")
        raise
