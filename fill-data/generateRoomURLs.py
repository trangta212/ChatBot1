import json

# Load the preprocessed room information
with open('simplified_rooms.json', 'r', encoding='utf-8') as f:
    rooms = json.load(f)

# Add URLs to each room
for index, room in enumerate(rooms, start=3400):
    room_id = index
    room['url'] = f"http://localhost:3000/user/room-details/{room_id}"

# Save the updated rooms back to the file
with open('simplified_rooms.json', 'w', encoding='utf-8') as f:
    json.dump(rooms, f, ensure_ascii=False, indent=2)

print(f"✅ URLs have been added to {len(rooms)} rooms in preprocessed_roomInformation.json") 