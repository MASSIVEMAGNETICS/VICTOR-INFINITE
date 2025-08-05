# ============================================================
# FILE: persona_system.py
# VERSION: v1.0.0
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# DESCRIPTION:
#     A standalone module for the Victor project that handles the
#     creation, storage, and management of unique digital personas.
#     This system is the foundation of the swarm, providing the
#     identities for all autonomous accounts.
#
# FEATURES:
#     ✔ Procedurally generates unique personas with names, backstories, and interests.
#     ✔ Assigns personality archetypes for behavioral modeling.
#     ✔ Stores and retrieves personas from a persistent SQLite database.
#     ✔ Uses data pools for generating varied and believable identities.
#
# PLATFORM: Python 3
# DEPENDENCIES: None (Uses standard Python libraries: sqlite3, random, uuid, json)
# HOW TO RUN: python persona_system.py
# HOW TO MAKE EXE: pyinstaller --onefile persona_system.py
# ============================================================

import sqlite3
import random
import uuid
import json

# --- Data Pools for Persona Generation ---
# These lists can be expanded significantly for greater variety.
FIRST_NAMES = ["Alex", "Ben", "Chloe", "David", "Eva", "Finn", "Grace", "Henry", "Isla", "Jack"]
LAST_NAMES = ["Smith", "Jones", "Chen", "Patel", "Khan", "Garcia", "Miller", "Davis", "Rossi", "Kim"]
INTEREST_POOLS = {
    "tech": ["AI", "cybersecurity", "blockchain", "SaaS", "gamedev", "robotics"],
    "creative": ["graphic design", "indie music", "filmmaking", "creative writing", "pottery"],
    "lifestyle": ["minimalism", "vanlife", "urban gardening", "homebrewing", "fitness"],
    "academic": ["philosophy", "history", "physics", "linguistics", "economics"]
}
PERSONALITY_ARCHETYPES = ["The Jester", "The Sage", "The Explorer", "The Creator", "The Ruler", "The Everyman"]
BACKSTORY_TEMPLATES = [
    "A former {job} who is now passionate about {interest_1} and {interest_2}.",
    "A self-taught expert in {interest_1}, currently exploring the world of {interest_2}.",
    "Deeply interested in the intersection of {interest_1} and {interest_2}.",
    "Just a curious mind trying to learn more about {interest_1}, {interest_2}, and everything in between."
]
JOB_TITLES = ["barista", "consultant", "developer", "artist", "researcher", "student", "analyst"]


class PersonaDatabase:
    """
    Handles all interactions with the SQLite database for storing and retrieving personas.
    """
    def __init__(self, db_file="personas.db"):
        """Initializes the database connection and creates the table if it doesn't exist."""
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.create_table()
            print(f"[*] Database connection successful. Using '{self.db_file}'")
        except sqlite3.Error as e:
            print(f"[!] Database Error: {e}")
            raise

    def create_table(self):
        """Creates the 'personas' table based on the architecture schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS personas (
            persona_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            backstory TEXT,
            interests TEXT,
            personality_archetype TEXT,
            profile_picture_url TEXT
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"[!] Error creating table: {e}")

    def save_persona(self, persona_data):
        """Saves a single persona dictionary to the database."""
        sql = '''INSERT INTO personas(persona_id, name, backstory, interests, personality_archetype, profile_picture_url)
                 VALUES(?,?,?,?,?,?)'''
        try:
            cursor = self.conn.cursor()
            # Interests are stored as a JSON string
            interests_json = json.dumps(persona_data['interests'])
            cursor.execute(sql, (
                persona_data['id'],
                persona_data['name'],
                persona_data['backstory'],
                interests_json,
                persona_data['archetype'],
                persona_data['pfp_url']
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[!] Error saving persona: {e}")

    def get_all_personas(self):
        """Retrieves all personas from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM personas")
            rows = cursor.fetchall()
            
            personas = []
            for row in rows:
                personas.append({
                    "id": row[0],
                    "name": row[1],
                    "backstory": row[2],
                    "interests": json.loads(row[3]), # Convert JSON string back to list
                    "archetype": row[4],
                    "pfp_url": row[5]
                })
            return personas
        except sqlite3.Error as e:
            print(f"[!] Error fetching personas: {e}")
            return []

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("[*] Database connection closed.")


class PersonaGenerator:
    """
    Generates unique, believable digital identities based on predefined data pools.
    """
    def generate(self):
        """Constructs a single persona dictionary."""
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        
        # Select 2-3 interests from a random category
        interest_category = random.choice(list(INTEREST_POOLS.keys()))
        num_interests = random.randint(2, 3)
        interests = random.sample(INTEREST_POOLS[interest_category], num_interests)
        
        # Create backstory
        backstory = random.choice(BACKSTORY_TEMPLATES).format(
            job=random.choice(JOB_TITLES),
            interest_1=interests[0],
            interest_2=interests[1]
        )
        
        persona = {
            "id": str(uuid.uuid4()),
            "name": f"{first_name} {last_name}",
            "backstory": backstory,
            "interests": interests,
            "archetype": random.choice(PERSONALITY_ARCHETYPES),
            "pfp_url": f"https://i.pravatar.cc/150?u={uuid.uuid4()}" # Placeholder for unique profile pics
        }
        return persona


def main():
    """
    Main function to demonstrate the Persona Management System.
    """
    print("==============================================")
    print("== VICTOR - PERSONA MANAGEMENT SYSTEM      ==")
    print("==============================================")
    
    db = PersonaDatabase()
    generator = PersonaGenerator()
    
    num_to_generate = 10
    print(f"\n[*] Generating {num_to_generate} new personas...")
    
    for i in range(num_to_generate):
        new_persona = generator.generate()
        db.save_persona(new_persona)
        print(f"  - Generated and saved persona: {new_persona['name']}")
        
    print(f"\n--- Verifying: Fetching all {num_to_generate} personas from database ---")
    all_personas = db.get_all_personas()
    
    if not all_personas:
        print("[!] No personas found in the database.")
        return

    for i, p in enumerate(all_personas):
        print(f"\n--- Persona {i+1}/{len(all_personas)} ---")
        print(f"  ID:        {p['id']}")
        print(f"  Name:      {p['name']}")
        print(f"  Archetype: {p['archetype']}")
        print(f"  Interests: {', '.join(p['interests'])}")
        print(f"  Backstory: {p['backstory']}")
        print(f"  PFP Link:  {p['pfp_url']}")
        
    db.close()
    print("\n[+] Persona Management System demo complete.")


if __name__ == "__main__":
    main()
