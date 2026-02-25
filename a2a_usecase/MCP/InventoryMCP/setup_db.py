"""Database Setup Script for Car Dealership System.

This script initializes the PostgreSQL database with the vehicles table
and seeds it with sample data for testing and development.

Database Schema:
    vehicles:
        - vin (VARCHAR, PK): Vehicle Identification Number
        - make (VARCHAR): Manufacturer name
        - model (VARCHAR): Model name
        - year (INT): Manufacturing year
        - color (VARCHAR): Vehicle color
        - price (DECIMAL): Price in dollars
        - status (VARCHAR): Availability status (default: 'AVAILABLE')

Prerequisites:
    - PostgreSQL server running on localhost:5432
    - Database 'dealership' created
    - User 'admin' with password 'password123' (or update connection string)

Usage:
    python setup_db.py

Environment:
    Connection string can be configured via POSTGRES_DSN environment variable
"""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "dbname=dealership user=admin password=password123 host=localhost port=5432"
)


def seed_sql() -> None:
    """Initialize database schema and seed with sample vehicle data.
    
    Creates the vehicles table if it doesn't exist and inserts sample
    vehicle records. Uses ON CONFLICT DO NOTHING to allow safe re-runs.
    
    Raises:
        psycopg2.Error: If database connection or query execution fails
    """
    print("⏳ Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                vin VARCHAR(17) PRIMARY KEY,
                make VARCHAR(50),
                model VARCHAR(50),
                year INT,
                color VARCHAR(20),
                price DECIMAL(10, 2),
                status VARCHAR(20) DEFAULT 'AVAILABLE'
            );
        """)
        
        # Insert sample data - diverse vehicle inventory
        cars = [
            # Sedans
            ('VIN001', 'Honda', 'Civic', 2024, 'Red', 25000.00),
            ('VIN002', 'Toyota', 'Camry', 2024, 'Silver', 28500.00),
            ('VIN003', 'Tesla', 'Model 3', 2024, 'White', 42000.00),
            ('VIN004', 'BMW', '3 Series', 2023, 'Black', 45000.00),
            ('VIN005', 'Mercedes-Benz', 'C-Class', 2024, 'Gray', 48000.00),
            ('VIN006', 'Audi', 'A4', 2023, 'Blue', 43000.00),
            ('VIN007', 'Hyundai', 'Elantra', 2024, 'White', 22000.00),
            
            # SUVs
            ('VIN008', 'Ford', 'Explorer', 2024, 'Black', 52000.00),
            ('VIN009', 'Jeep', 'Grand Cherokee', 2023, 'Green', 48000.00),
            ('VIN010', 'Toyota', 'RAV4', 2024, 'Blue', 32000.00),
            ('VIN011', 'Tesla', 'Model Y', 2024, 'Red', 54000.00),
            ('VIN012', 'Mazda', 'CX-5', 2024, 'Gray', 35000.00),
            ('VIN013', 'Chevrolet', 'Tahoe', 2023, 'White', 58000.00),
            
            # Trucks
            ('VIN014', 'Ford', 'F-150', 2023, 'Blue', 45000.00),
            ('VIN015', 'Chevrolet', 'Silverado', 2024, 'Red', 47000.00),
            ('VIN016', 'Ram', '1500', 2023, 'Black', 46000.00),
            ('VIN017', 'Toyota', 'Tacoma', 2024, 'Silver', 38000.00),
            
            # Luxury
            ('VIN018', 'Porsche', '911', 2024, 'Yellow', 115000.00),
            ('VIN019', 'Lexus', 'ES 350', 2024, 'Pearl', 47000.00),
            ('VIN020', 'Mercedes-Benz', 'S-Class', 2023, 'Black', 105000.00),
            
            # Electric/Hybrid
            ('VIN021', 'Nissan', 'Leaf', 2024, 'White', 35000.00),
            ('VIN022', 'Chevrolet', 'Bolt', 2023, 'Blue', 32000.00),
            ('VIN023', 'Toyota', 'Prius', 2024, 'Silver', 29000.00),
        ]
        
        for car in cars:
            cur.execute(
                "INSERT INTO vehicles (vin, make, model, year, color, price) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                car
            )
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database initialized and seeded successfully")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise


if __name__ == "__main__":
    seed_sql()
