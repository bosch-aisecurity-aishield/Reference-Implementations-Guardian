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
        - mileage (INT): Odometer reading in miles
        - transmission (VARCHAR): Transmission type (Automatic/Manual)
        - fuel_type (VARCHAR): Fuel type (Gasoline/Diesel/Electric/Hybrid)
        - body_type (VARCHAR): Vehicle category (Sedan/SUV/Truck/Coupe/Hatchback)
        - description (TEXT): Vehicle features and description
        - status (VARCHAR): Availability status (AVAILABLE/SOLD/RESERVED)

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
    
    Creates the vehicles table with comprehensive fields and inserts diverse
    sample vehicle records. Uses ON CONFLICT DO NOTHING to allow safe re-runs.
    
    Raises:
        psycopg2.Error: If database connection or query execution fails
    """
    print("⏳ Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()
        
        # Drop table if exists for clean setup
        print("🔄 Creating vehicles table...")
        cur.execute("DROP TABLE IF EXISTS vehicles CASCADE;")
        
        cur.execute("""
            CREATE TABLE vehicles (
                vin VARCHAR(17) PRIMARY KEY,
                make VARCHAR(50) NOT NULL,
                model VARCHAR(50) NOT NULL,
                year INT NOT NULL CHECK (year >= 1900 AND year <= 2030),
                color VARCHAR(30) NOT NULL,
                price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
                mileage INT DEFAULT 0 CHECK (mileage >= 0),
                transmission VARCHAR(20) DEFAULT 'Automatic',
                fuel_type VARCHAR(20) DEFAULT 'Gasoline',
                body_type VARCHAR(20),
                description TEXT,
                status VARCHAR(20) DEFAULT 'AVAILABLE' CHECK (status IN ('AVAILABLE', 'SOLD', 'RESERVED')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for common query patterns
        print("📑 Creating indexes...")
        cur.execute("CREATE INDEX idx_make ON vehicles(make);")
        cur.execute("CREATE INDEX idx_model ON vehicles(model);")
        cur.execute("CREATE INDEX idx_year ON vehicles(year);")
        cur.execute("CREATE INDEX idx_price ON vehicles(price);")
        cur.execute("CREATE INDEX idx_status ON vehicles(status);")
        cur.execute("CREATE INDEX idx_body_type ON vehicles(body_type);")
        
        print("📝 Inserting sample data...")
        
        # Insert comprehensive sample data
        cars = [
            # Sedans - Affordable
            ('1HGBH41JXMN109186', 'Honda', 'Civic', 2024, 'Crystal Black Pearl', 25999.00, 145, 'Automatic', 'Gasoline', 'Sedan',
             'Fuel-efficient compact sedan with advanced safety features, Apple CarPlay, Android Auto, and excellent reliability rating.'),
            ('4T1BF1FK5EU123456', 'Toyota', 'Camry', 2024, 'Celestial Silver', 28500.00, 892, 'Automatic', 'Hybrid', 'Sedan',
             'Midsize hybrid sedan with spacious interior, premium sound system, adaptive cruise control, and exceptional fuel economy.'),
            ('WBA3A5C51EK234567', 'BMW', '330i', 2023, 'Alpine White', 45000.00, 8500, 'Automatic', 'Gasoline', 'Sedan',
             'Luxury sport sedan with turbocharged engine, premium leather interior, navigation, and dynamic handling.'),
            ('WDDWF8EB5ER345678', 'Mercedes-Benz', 'C300', 2024, 'Selenite Gray', 48000.00, 2100, 'Automatic', 'Gasoline', 'Sedan',
             'Premium sedan with advanced MBUX infotainment, LED headlights, memory seats, and sophisticated design.'),
            ('WAUFFAFL8EN456789', 'Audi', 'A4', 2023, 'Navarra Blue', 43000.00, 12000, 'Automatic', 'Gasoline', 'Sedan',
             'Refined sedan with Quattro AWD, virtual cockpit, Bang & Olufsen sound, and elegant craftsmanship.'),
            ('5NPE24AF2MH567890', 'Hyundai', 'Elantra', 2024, 'Intense Blue', 22000.00, 3400, 'Automatic', 'Gasoline', 'Sedan',
             'Value-packed sedan with modern styling, 10-year warranty, wireless charging, and comprehensive safety suite.'),
            ('3VW2B7AJ8EM678901', 'Volkswagen', 'Jetta', 2024, 'Pure White', 24500.00, 5600, 'Automatic', 'Gasoline', 'Sedan',
             'German-engineered sedan with spacious trunk, touchscreen display, and solid build quality.'),
            
            # SUVs - Crossovers and Full-Size
            ('1FM5K8GC8LGA78912', 'Ford', 'Explorer', 2024, 'Agate Black', 52000.00, 1200, 'Automatic', 'Gasoline', 'SUV',
             'Three-row family SUV with powerful EcoBoost engine, hands-free driving tech, and towing capacity up to 5,600 lbs.'),
            ('1C4RJFBG5KC789123', 'Jeep', 'Grand Cherokee', 2023, 'Olive Green', 48000.00, 15000, 'Automatic', 'Gasoline', 'SUV',
             'Rugged luxury SUV with legendary off-road capability, Uconnect system, and premium Nappa leather seats.'),
            ('2T3F1RFV8PC890234', 'Toyota', 'RAV4', 2024, 'Blueprint', 32000.00, 4500, 'Automatic', 'Hybrid', 'SUV',
             'Best-selling compact SUV with AWD, excellent reliability, ample cargo space, and impressive fuel efficiency.'),
            ('5YJYGDEE1MF901345', 'Tesla', 'Model Y', 2024, 'Deep Blue Metallic', 54000.00, 6800, 'Automatic', 'Electric', 'SUV',
             'All-electric crossover with 330 miles range, autopilot, panoramic glass roof, and supercharger access.'),
            ('JM3KFBDM5M0012456', 'Mazda', 'CX-5', 2024, 'Soul Red Crystal', 35000.00, 7200, 'Automatic', 'Gasoline', 'SUV',
             'Stylish compact SUV with premium interior, Skyactiv technology, Bose audio, and engaging driving dynamics.'),
            ('3GNKBHRS0PS123567', 'Chevrolet', 'Tahoe', 2023, 'Summit White', 58000.00, 18000, 'Automatic', 'Gasoline', 'SUV',
             'Full-size SUV with seating for 9, 8,400 lbs towing, advanced safety tech, and commanding road presence.'),
            ('5TDJGRFH8KS234678', 'Toyota', 'Highlander', 2024, 'Wind Chill Pearl', 42000.00, 3900, 'Automatic', 'Hybrid', 'SUV',
             'Three-row family SUV with hybrid efficiency, 5,000 lbs towing, leather seats, and reliable performance.'),
            
            # Trucks - Light Duty and Heavy Duty
            ('1FTFW1E84MFA45789', 'Ford', 'F-150', 2024, 'Velocity Blue', 52000.00, 8900, 'Automatic', 'Gasoline', 'Truck',
             'America\'s best-selling truck with 13,200 lbs towing, aluminum body, Pro Power Onboard, and rugged capability.'),
            ('3GCPYFED8RG456890', 'Chevrolet', 'Silverado 1500', 2024, 'Cherry Red Tintcoat', 49000.00, 11000, 'Automatic', 'Gasoline', 'Truck',
             'Full-size pickup with spacious crew cab, 13,300 lbs towing, Multi-Flex tailgate, and bold styling.'),
            ('1C6SRFFT5MN567901', 'Ram', '1500', 2023, 'Diamond Black', 46000.00, 22000, 'Automatic', 'Gasoline', 'Truck',
             'Refined pickup with best-in-class interior, air suspension, 12-inch touchscreen, and smooth ride quality.'),
            ('5TFDY5F18NX678012', 'Toyota', 'Tacoma', 2024, 'Army Green', 38000.00, 12500, 'Manual', 'Gasoline', 'Truck',
             'Midsize off-road truck with TRD package, manual transmission, crawl control, and legendary durability.'),
            ('1FTEW1EP8PFC89123', 'Ford', 'F-250 Super Duty', 2024, 'Stone Gray', 68000.00, 5200, 'Automatic', 'Diesel', 'Truck',
             'Heavy-duty workhorse with 20,000 lbs towing, Power Stroke diesel, trailer brake controller, and payload capacity.'),
            
            # Luxury and Sports Cars
            ('WP0AA2A93PS234567', 'Porsche', '911 Carrera', 2024, 'Racing Yellow', 115000.00, 1200, 'Automatic', 'Gasoline', 'Coupe',
             'Iconic sports car with 379 hp twin-turbo flat-six, PDK transmission, Sport Chrono package, and thrilling performance.'),
            ('JTHBK1GG8MA345678', 'Lexus', 'ES 350', 2024, 'Eminent White Pearl', 47000.00, 3400, 'Automatic', 'Gasoline', 'Sedan',
             'Luxury sedan with whisper-quiet cabin, Mark Levinson audio, semi-aniline leather, and exceptional comfort.'),
            ('WDDUG8CB5LA456789', 'Mercedes-Benz', 'S500', 2024, 'Obsidian Black', 115000.00, 4800, 'Automatic', 'Gasoline', 'Sedan',
             'Flagship luxury sedan with massaging seats, Burmester sound, 64-color ambient lighting, and cutting-edge tech.'),
            ('WAUZZZ4G8DN567890', 'Audi', 'R8', 2023, 'Suzuka Gray', 155000.00, 3200, 'Automatic', 'Gasoline', 'Coupe',
             'Supercar with 562 hp V10 engine, Quattro AWD, carbon fiber accents, and exotic styling.'),
            ('WBA3C1C52EK678901', 'BMW', 'M4 Competition', 2024, 'Toronto Red', 78000.00, 2900, 'Automatic', 'Gasoline', 'Coupe',
             'High-performance coupe with 503 hp twin-turbo I6, M Sport seats, M Differential, and razor-sharp handling.'),
            
            # Electric and Hybrid Vehicles
            ('5YJSA1E27MF789012', 'Tesla', 'Model 3', 2024, 'Pearl White Multi-Coat', 42000.00, 8500, 'Automatic', 'Electric', 'Sedan',
             'Premium electric sedan with 358 miles range, autopilot, over-the-air updates, and minimalist interior.'),
            ('1N4BZ1CP4PC890123', 'Nissan', 'Leaf', 2024, 'Electric Blue', 35000.00, 6700, 'Automatic', 'Electric', 'Hatchback',
             'Affordable EV with 212 miles range, ProPILOT Assist, e-Pedal, and practical hatchback versatility.'),
            ('1G1FZ6S01M4901234', 'Chevrolet', 'Bolt EUV', 2024, 'Bright Blue', 32000.00, 9200, 'Automatic', 'Electric', 'SUV',
             'Compact electric SUV with 247 miles range, Super Cruise, spacious interior, and one-pedal driving.'),
            ('JTDKARFP8M3012345', 'Toyota', 'Prius', 2024, 'Wind Chill Pearl', 29000.00, 11000, 'Automatic', 'Hybrid', 'Hatchback',
             'Legendary hybrid with 57 mpg, advanced safety, solar roof option, and proven reliability.'),
            ('KNDJX3AE8N7123456', 'Kia', 'Niro EV', 2024, 'Gravity Gray', 41000.00, 4300, 'Automatic', 'Electric', 'SUV',
             'Versatile electric crossover with 253 miles range, spacious cargo, UVO intelligence, and modern design.'),
            
            # Sporty Compact and Hot Hatches
            ('3VWC57BU8MM234567', 'Volkswagen', 'Golf GTI', 2024, 'Tornado Red', 32000.00, 7800, 'Manual', 'Gasoline', 'Hatchback',
             'Hot hatch with 241 hp turbo engine, plaid seats, adaptive dampers, and European driving dynamics.'),
            ('JTHBP5C26M5345678', 'Lexus', 'IS 350', 2024, 'Ultrasonic Blue', 44000.00, 5100, 'Automatic', 'Gasoline', 'Sedan',
             'Sporty luxury sedan with 311 hp V6, F Sport package, Mark Levinson audio, and aggressive styling.'),
            ('JM1BN1W38M1456789', 'Mazda', 'Miata MX-5', 2024, 'Soul Red Crystal', 29000.00, 4200, 'Manual', 'Gasoline', 'Convertible',
             'Pure driving pleasure with lightweight design, perfect 50/50 balance, manual top, and responsive steering.'),
            
            # Family Minivans and Crossovers  
            ('2C4RC1N70NR567890', 'Chrysler', 'Pacifica', 2024, 'Brilliant Blue', 38000.00, 8900, 'Automatic', 'Hybrid', 'Minivan',
             'Versatile hybrid minivan with Stow \'n Go seating, built-in vacuum, Uconnect theater, and 30 mpg combined.'),
            ('5FNRL6H73MB678901', 'Honda', 'Odyssey', 2024, 'Lunar Silver', 40000.00, 6500, 'Automatic', 'Gasoline', 'Minivan',
             'Family hauler with magic slide seats, CabinWatch, wireless charging, and exceptional safety ratings.'),
            
            # Sold and Reserved Examples
            ('1GNEVGKW1MJ789012', 'Chevrolet', 'Bolt EV', 2023, 'Summit White', 28000.00, 15000, 'Automatic', 'Electric', 'Hatchback',
             'Electric hatchback with 259 miles range, one-pedal driving, DC fast charging, and practicality.', 'SOLD'),
            ('5UXCR6C0XL9890123', 'BMW', 'X5', 2023, 'Phytonic Blue', 67000.00, 9800, 'Automatic', 'Gasoline', 'SUV',
             'Luxury SUV with three-row seating, gesture control, Harman Kardon sound, and commanding presence.', 'RESERVED'),
        ]
        
        for car_data in cars:
            if len(car_data) == 11:  # Without status
                cur.execute(
                    """INSERT INTO vehicles 
                    (vin, make, model, year, color, price, mileage, transmission, fuel_type, body_type, description) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                    ON CONFLICT DO NOTHING""",
                    car_data
                )
            else:  # With status
                cur.execute(
                    """INSERT INTO vehicles 
                    (vin, make, model, year, color, price, mileage, transmission, fuel_type, body_type, description, status) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                    ON CONFLICT DO NOTHING""",
                    car_data
                )
        
        conn.commit()
        
        # Display statistics
        cur.execute("SELECT COUNT(*) FROM vehicles;")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM vehicles WHERE status = 'AVAILABLE';")
        available = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT make) FROM vehicles;")
        makes = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"✅ Database initialized successfully!")
        print(f"📊 Statistics:")
        print(f"   • Total vehicles: {total}")
        print(f"   • Available: {available}")
        print(f"   • Unique makes: {makes}")
        print(f"   • Body types: Sedan, SUV, Truck, Coupe, Hatchback, Minivan, Convertible")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise


if __name__ == "__main__":
    seed_sql()
