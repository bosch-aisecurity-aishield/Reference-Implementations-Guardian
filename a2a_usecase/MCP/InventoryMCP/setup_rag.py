"""Vector Store Setup Script for Car Dealership System.

This script initializes the ChromaDB vector database with vehicle visual
features and descriptions for semantic search capabilities.

Vector Collection:
    vehicle_visuals:
        - Documents: Textual descriptions of vehicle features
        - Metadata: VIN, image path, feature type (interior/exterior)
        - IDs: Unique identifier for each document

Use Cases:
    - Semantic search for visual features (leather seats, sunroof, etc.)
    - Natural language queries about vehicle appearance
    - Image-based vehicle recommendations

Prerequisites:
    - ChromaDB installed (included in requirements.txt)
    - Write permissions to ./chroma_data directory

Usage:
    python setup_rag.py

Environment:
    Storage path can be configured via CHROMA_PERSIST_DIR environment variable
"""

from __future__ import annotations

import os
import shutil

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")


def seed_vectors() -> None:
    """Initialize ChromaDB vector store with sample vehicle visual data.
    
    Creates the vehicle_visuals collection if it doesn't exist and adds
    sample documents with embeddings for semantic search.
    
    Raises:
        Exception: If vector store initialization fails
    """
    print("🧠 Initializing ChromaDB vector store...")
    
    # Optional: Reset DB to ensure clean state with new images
    # if os.path.exists(CHROMA_DIR):
    #     shutil.rmtree(CHROMA_DIR)

    # Initialize persistent client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    
    # Delete existing collection if it exists to ensure fresh data
    try:
        client.delete_collection("vehicle_visuals")
    except ValueError:
        pass  # Collection didn't exist

    collection = client.get_or_create_collection(
        name="vehicle_visuals",
        embedding_function=emb_fn
    )
    
    # Data structure to hold all vehicle info
    # Using specific Unsplash IDs to ensure images match the descriptions
    vehicle_data = [
        # --- SEDANS ---
        {
            "vin": "VIN001", 
            "model": "Honda Civic",
            "images": [
                ("https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=800", "exterior", "front_angle", "Red Honda Civic exterior with sporty front grille, LED headlights, and aerodynamic sedan profile"),
                ("https://images.unsplash.com/photo-1590362891991-f776e747a588?w=800", "exterior", "side_profile", "Side view of red Honda Civic showing alloy wheels and sleek body lines"),
                ("https://images.unsplash.com/photo-1517524008697-84bbe3c3fd98?w=800", "interior", "dashboard", "Honda Civic interior dashboard with touchscreen infotainment, digital cluster, and ergonomic steering wheel")
            ]
        },
        {
            "vin": "VIN002",
            "model": "Toyota Camry",
            "images": [
                ("https://images.unsplash.com/photo-1621007947382-bb3c3968e3bb?w=800", "exterior", "front", "Silver Toyota Camry front view showing wide grille, chrome accents, and modern sedan styling"),
                ("https://images.unsplash.com/photo-1550355291-bbee04a92027?w=800", "interior", "seats", "Toyota Camry leather interior seating with spacious legroom, center armrest, and premium trim materials")
            ]
        },
        {
            "vin": "VIN003",
            "model": "Tesla Model 3",
            "images": [
                ("https://images.unsplash.com/photo-1536700503339-1e4b06520771?w=800", "exterior", "front_driving", "White Tesla Model 3 driving on road, showing grille-less front fascia, aerodynamic wheels, and glass roof"),
                ("https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=800", "interior", "minimalist_dash", "Tesla Model 3 minimalist interior featuring 15-inch center touchscreen, wood trim dashboard, and vegan leather seats"),
                ("https://images.unsplash.com/photo-1571127236794-81c0bbfe1ce3?w=800", "interior", "steering", "Futuristic steering wheel and clean dashboard view of Tesla Model 3")
            ]
        },
        {
            "vin": "VIN004",
            "model": "BMW 3 Series",
            "images": [
                ("https://images.unsplash.com/photo-1556189250-72ba954522cd?w=800", "exterior", "front", "Black BMW 3 Series with iconic kidney grille, aggressive LED headlights, and M-sport bumper"),
                ("https://images.unsplash.com/photo-1555215695-3004980ad54e?w=800", "interior", "cockpit", "BMW driver-oriented cockpit with digital instrument cluster, iDrive controller, and leather sport steering wheel")
            ]
        },
        {
            "vin": "VIN005",
            "model": "Mercedes-Benz C-Class",
            "images": [
                ("https://images.unsplash.com/photo-1617788138017-80ad40651399?w=800", "exterior", "side", "Gray Mercedes-Benz C-Class luxury sedan side profile with elegant lines and alloy wheels"),
                ("https://images.unsplash.com/photo-1608994751987-e647898d28a5?w=800", "interior", "ambient_lighting", "Mercedes C-Class interior with ambient lighting, turbine air vents, and high-resolution center display")
            ]
        },
        {
            "vin": "VIN006",
            "model": "Audi A4",
            "images": [
                ("https://images.unsplash.com/photo-1606152421811-991d5877404e?w=800", "exterior", "front", "Blue Audi A4 sedan with singleframe grille, sharp LED light signature, and professional executive look"),
                ("https://images.unsplash.com/photo-1507136566006-cfc505b114fc?w=800", "interior", "virtual_cockpit", "Audi interior featuring Virtual Cockpit digital display, leather upholstery, and minimalist center console")
            ]
        },
        {
            "vin": "VIN007",
            "model": "Hyundai Elantra",
            "images": [
                ("https://images.unsplash.com/photo-1624502559648-2612be9859f5?w=800", "exterior", "angle", "White Hyundai Elantra showing sharp parametric dynamics design, geometric body panels, and modern styling")
            ]
        },

        # --- SUVS ---
        {
            "vin": "VIN008",
            "model": "Ford Explorer",
            "images": [
                ("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=800", "exterior", "front", "Black Ford Explorer SUV with bold mesh grille, rugged stance, and roof rails ready for adventure"),
                ("https://images.unsplash.com/photo-1519641471654-76ce0107ad1b?w=800", "interior", "cabin", "Spacious Ford Explorer cabin showing third-row seating capability and family-friendly layout")
            ]
        },
        {
            "vin": "VIN009",
            "model": "Jeep Grand Cherokee",
            "images": [
                ("https://images.unsplash.com/photo-1519245659620-e859806a8d3b?w=800", "exterior", "offroad", "Green Jeep Grand Cherokee in outdoor setting, featuring seven-slot grille and off-road capable suspension"),
                ("https://images.unsplash.com/photo-1594967399222-263a036c0a08?w=800", "interior", "luxury_trim", "Jeep Grand Cherokee interior with quilted leather seats, wood trim, and Uconnect infotainment system")
            ]
        },
        {
            "vin": "VIN010",
            "model": "Toyota RAV4",
            "images": [
                ("https://images.unsplash.com/photo-1609521263047-f8f205293f24?w=800", "exterior", "front_angle", "Blue Toyota RAV4 showing rugged adventure-ready front end, wheel arch cladding, and roof rails"),
                ("https://images.unsplash.com/photo-1506015391300-4802dc74de2e?w=800", "interior", "dash", "Toyota RAV4 functional interior with floating touchscreen, tactile climate knobs, and durable seating materials")
            ]
        },
        {
            "vin": "VIN011",
            "model": "Tesla Model Y",
            "images": [
                ("https://images.unsplash.com/photo-1619247662169-95e20689b657?w=800", "exterior", "rear_angle", "Red Tesla Model Y showing crossover SUV shape, rear hatch, and elevated ride height"),
                ("https://images.unsplash.com/photo-1532581140115-3e355d1ed1de?w=800", "interior", "panoramic_roof", "Tesla Model Y interior looking up through full panoramic glass roof, creating open airy feeling")
            ]
        },
        {
            "vin": "VIN012",
            "model": "Mazda CX-5",
            "images": [
                ("https://images.unsplash.com/photo-1570374910698-6db3d7d7f649?w=800", "exterior", "front", "Gray Mazda CX-5 with signature wing grille, sleek Kodo design language, and premium paint finish"),
                ("https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=800", "interior", "premium_feel", "Mazda CX-5 upscale interior with soft-touch materials, commander control knob, and driver-centric layout")
            ]
        },
        {
            "vin": "VIN013",
            "model": "Chevrolet Tahoe",
            "images": [
                ("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=800", "exterior", "side", "White Chevrolet Tahoe full-size SUV showing massive length, running boards, and commanding road presence"),
                ("https://images.unsplash.com/photo-1616455579100-2ceaa4eb2d37?w=800", "interior", "cargo_space", "Chevrolet Tahoe cavernous cargo area with power-folding rear seats and vast storage capacity")
            ]
        },

        # --- TRUCKS ---
        {
            "vin": "VIN014",
            "model": "Ford F-150",
            "images": [
                ("https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=800", "exterior", "front_action", "Blue Ford F-150 pickup truck driving on dirt, showing high clearance and tough truck bed"),
                ("https://images.unsplash.com/photo-1605218427368-35b8618eb721?w=800", "interior", "work_surface", "Ford F-150 interior with fold-flat work surface, large center console, and durable truck controls")
            ]
        },
        {
            "vin": "VIN015",
            "model": "Chevrolet Silverado",
            "images": [
                ("https://images.unsplash.com/photo-1551830820-330a71b99659?w=800", "exterior", "front", "Red Chevrolet Silverado with chrome grille bar, heavy duty stance, and towing mirrors"),
                ("https://images.unsplash.com/photo-1626027376742-8c171092e42f?w=800", "exterior", "tailgate", "Chevrolet Silverado rear view showing Multi-Flex tailgate and spray-in bedliner")
            ]
        },
        {
            "vin": "VIN016",
            "model": "Ram 1500",
            "images": [
                ("https://images.unsplash.com/photo-1594967399245-8c772dc6a117?w=800", "exterior", "front", "Black Ram 1500 truck with aggressive rebel styling, hood vents, and off-road tires"),
                ("https://images.unsplash.com/photo-1618684617300-4e3271420786?w=800", "interior", "luxury_truck", "Ram 1500 luxury interior with large 12-inch vertical touchscreen and leather etched detailing")
            ]
        },
        {
            "vin": "VIN017",
            "model": "Toyota Tacoma",
            "images": [
                ("https://images.unsplash.com/photo-1600180296767-f27eb66d5100?w=800", "exterior", "offroad_action", "Silver Toyota Tacoma mid-size truck climbing rocks, showing TRD Pro skid plates and suspension"),
                ("https://images.unsplash.com/photo-1518987048-93e29699e79a?w=800", "exterior", "bed_view", "Toyota Tacoma bed with composite liner, deck rail system, and tie-down cleats")
            ]
        },

        # --- LUXURY ---
        {
            "vin": "VIN018",
            "model": "Porsche 911",
            "images": [
                ("https://images.unsplash.com/photo-1503376763036-066120622c74?w=800", "exterior", "side_speed", "Yellow Porsche 911 sports car profile showing iconic slope, rear engine intake, and performance brakes"),
                ("https://images.unsplash.com/photo-1580274455191-1c62238fa333?w=800", "interior", "sport_chrono", "Porsche 911 interior featuring Sport Chrono clock on dash, Alcantara steering wheel, and racing seats"),
                ("https://images.unsplash.com/photo-1614162692292-7ac56d7f0e9e?w=800", "exterior", "rear", "Rear view of Porsche 911 with active spoiler deployed and sport exhaust system")
            ]
        },
        {
            "vin": "VIN019",
            "model": "Lexus ES 350",
            "images": [
                ("https://images.unsplash.com/photo-1592853625601-bb9d239129df?w=800", "exterior", "front", "Pearl Lexus ES with Spindle grille, triple-beam LED headlights, and elegant executive styling"),
                ("https://images.unsplash.com/photo-1621939514649-280e2ee25f60?w=800", "interior", "comfort", "Lexus interior showing hand-stitched leather dash, analog clock, and mark levinson speaker grilles")
            ]
        },
        {
            "vin": "VIN020",
            "model": "Mercedes-Benz S-Class",
            "images": [
                ("https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=800", "exterior", "front_night", "Black Mercedes S-Class flagship sedan with hood ornament, multibeam LEDs, and chrome luxury trim"),
                ("https://images.unsplash.com/photo-1563720360172-67b8f3dce741?w=800", "interior", "executive_rear", "Mercedes S-Class rear executive lounge seating with massage pillows, leg rests, and rear entertainment screens"),
                ("https://images.unsplash.com/photo-1609337583696-6e792c906a52?w=800", "interior", "screens", "Mercedes MBUX hyperscreen dashboard with OLED technology and augmented reality display")
            ]
        },

        # --- ELECTRIC/HYBRID ---
        {
            "vin": "VIN021",
            "model": "Nissan Leaf",
            "images": [
                ("https://images.unsplash.com/photo-1593941707882-a5bba14938c7?w=800", "exterior", "charging", "White Nissan Leaf plugged into charger, showing front charging port and EV badging"),
                ("https://images.unsplash.com/photo-1519750292352-c9fc17322ed7?w=800", "interior", "shifter", "Nissan Leaf interior with unique drive selector puck and eco-mode display")
            ]
        },
        {
            "vin": "VIN022",
            "model": "Chevrolet Bolt",
            "images": [
                ("https://images.unsplash.com/photo-1609521263047-f8f205293f24?w=800", "exterior", "compact", "Blue Chevrolet Bolt EV hatchback showing compact city-friendly dimensions and closed grille"),
                ("https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800", "interior", "tech", "Chevrolet Bolt modern interior with floating center console and 10.2-inch infotainment screen")
            ]
        },
        {
            "vin": "VIN023",
            "model": "Toyota Prius",
            "images": [
                ("https://images.unsplash.com/photo-1533106497176-45ae19e68ba2?w=800", "exterior", "profile", "Silver Toyota Prius hybrid showing aerodynamic wedge shape for maximum fuel efficiency"),
                ("https://images.unsplash.com/photo-1583267746897-dee5c6cf3b38?w=800", "interior", "center_display", "Toyota Prius dashboard with central mounted gauge cluster and hybrid system energy monitor")
            ]
        }
    ]

    documents = []
    metadatas = []
    ids = []
    
    doc_id_counter = 1

    for vehicle in vehicle_data:
        for url, img_type, view, desc in vehicle['images']:
            # Add description document
            documents.append(desc)
            
            # Add metadata for retrieval
            metadatas.append({
                "vin": vehicle['vin'],
                "model": vehicle['model'],
                "image_url": url,
                "type": img_type,
                "view": view
            })
            
            # Create unique ID
            ids.append(f"doc_{str(doc_id_counter).zfill(3)}")
            doc_id_counter += 1

    # Add documents with embeddings
    # Using batches to be safe with large lists (though this size is fine)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Vector store initialized at {CHROMA_DIR}")
    print(f"   Added {len(documents)} vehicle visual documents covering {len(vehicle_data)} unique vehicles")


if __name__ == "__main__":
    seed_vectors()