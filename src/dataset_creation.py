import pandas as pd
import random

# Define core templates for the 5 intents
templates = {
    "Rescue": [
        "SOS: {count} people trapped on a roof in {loc}. Water rising fast.",
        "URGENT: Landslide in {loc}. {count} houses buried. Need rescue team immediately.",
        "My {relation} is stuck in a tree in {loc} (immediate danger).",
        "We are trapped in the attic in {loc}. {count} kids. Water is entering.",
        "Helicopter rescue needed for a group of hikers in {loc}.",
        "Boat capsized in {loc}. {count} people in the water. Need help!",
        "Emergency: {loc} area flooded. People stranded on top of a van.",
        "SOS: {count} people stranded on a rock in {river}. Water rising rapidly."
    ],
    "Medical": [
        "Help! My {relation} is bedridden and the water is knee deep. Location: {loc}.",
        "Rescue needed: Pregnant woman in labor at {loc}. Cannot reach hospital.",
        "Emergency: Tree fell on a house in {loc}. {count} injured.",
        "Warning: Dengue risk is high in {loc} due to stagnant water.",
        "Gampaha hospital is requesting urgent medicine for {count} patients.",
        "Need ambulance to {loc} for an elderly person with chest pain.",
        "My {relation} has a high fever and we are cut off by floods in {loc}."
    ],
    "Supply": [
        "Does anyone have extra dry rations for the camp in {loc}?",
        "Gampaha hospital is requesting drinking water and food for patients.",
        "We need milk powder and diapers for {count} babies at the {loc} camp.",
        "Donation needed: {count} packets of rice and curry for lunch in {loc}.",
        "We need blankets and clothes for the children at the {loc} school.",
        "Requesting 500 liters of drinking water for the shelter in {loc}."
    ],
    "Utility": [
        "BREAKING: Water levels in {river} ({loc}) have reached {level} meters. Warning!",
        "Update: {road} road cleared near {loc}. Traffic moving slowly.",
        "Is the highway open? I need to get to the airport from {loc}.",
        "River overflow in {loc}. Level {level}m. Critical warning issued.",
        "Electricity restored in some parts of {loc}. Infrastructure stable.",
        "Kalutara bridge is at risk of collapse. Police have closed it.",
        "Road block at the entrance to the highway in {loc}. Avoid the area."
    ],
    "Other": [
        "Just saw on news that {loc} town is flooded. Hope everyone is safe.",
        "Just arrived at the shelter in {loc}. It's crowded but safe.",
        "The government has allocated relief funds for the {loc} district.",
        "My dog is missing in {loc}. Has anyone seen him?",
        "Please pray for Sri Lanka. Stay safe everyone.",
        "Donation drive happening at the town hall in {loc} tomorrow.",
        "I lost my ID card in the flood in {loc}. What should I do?"
    ]
}

# Parameters for data randomization
locations = ["Colombo", "Gampaha", "Ja-Ela", "Badulla", "Kelaniya", "Ratnapura", "Kandy", "Wattala", "Matara", "Kegalle", "Nuwara Eliya", "Wellampitiya", "Kolonnawa"]
relations = ["uncle", "grandmother", "sister", "neighbor", "grandfather", "brother"]
rivers = ["Kelani River", "Kalu Ganga", "Mahaweli River", "Nilwala River"]
roads = ["Kandy Road", "Galle Road", "Low Level Road", "High Level Road"]

data = []

# Generate 1000 rows
for i in range(10000):
    intent = random.choice(list(templates.keys()))
    template = random.choice(templates[intent])
    
    # Fill template slots
    message = template.format(
        count=random.randint(2, 20),
        loc=random.choice(locations),
        relation=random.choice(relations),
        river=random.choice(rivers),
        level=round(random.uniform(4.5, 12.0), 1),
        road=random.choice(roads)
    )
    data.append({"message": message, "intent": intent})

# Convert to DataFrame and Export
df = pd.DataFrame(data)
df.to_csv("data/crisis_intent_data.csv", index=False)

print(f"Successfully created 'crisis_intent_data.csv' with {len(df)} rows.")