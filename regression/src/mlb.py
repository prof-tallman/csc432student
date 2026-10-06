import os
import json
import requests

BASE_URL = 'https://statsapi.mlb.com/api/v1'
MLB_EP_TEAMS_LISTING = 'teams'
MLB_EP_TEAMS_ROSTER  = 'teams/{team_id}/roster'
MLB_EP_GAMES_SCHEDULE = 'schedule'
MLB_EP_GAMES_BOXSCORE = 'game/{game_id}/boxscore'
MLB_EP_GAMES_FULLFEED = 'game/{game_id}/feed/live'

def save_json_data(filename, data):
    with open(filename, 'w') as fout:
        json_string_data = json.dumps(data)
        fout.write(json_string_data)
        
def load_json_data(filename):
    with open(filename) as fin:
        json_data = json.load(fin)
        return json_data

def download_json_data(endpoint, query={}, filename=None):
    
    if filename != None and os.path.exists(filename):
        print(f">>> Loading JSON from cached file {filename}")
        with open(filename, 'r') as file:
            return json.load(file)
        
    url = f"{BASE_URL}/{endpoint}"
    response = requests.request("GET", url, params=query)
    print(f">>> GET {response.request.url}")
    json_data = response.json()
    
    if filename != None:
        save_json_data(filename, json_data)

    return json_data


# ALL TEAMS
params = {'sportId':1, 'season':2026}
data = download_json_data(MLB_EP_TEAMS_LISTING, 
                          params, 
                          'mlb_teams.json')
csv_data = []
for team in data['teams']:
    record = { 'id':team['id'], 'year':team['season'], 'name':team['teamName'],
               'abb':team['abbreviation'], 'location':team['locationName'],
               }
    csv_data.append(record)

# ALL PLAYERS
#endpoint = f"MLB_EP_TEAMS_ROSTER"
#data = download_json_data(MLB_EP_TEAMS_LISTING, 
#                          {'sportId':1, 'season':2026}, 
#                          'mlb_teams.json')


params = {
    "sportId": 1,
    "teamId": 119,
    "startDate": "2026-10-05",
    "endDate": "2026-10-12",
    "hydrate": "probablePitcher"
}
endpoint = MLB_EP_GAMES_SCHEDULE
data = download_json_data(endpoint, params, None)

for day in data['dates']:
    for game in day['games']:
        print(game['gamePk'], game['gameDate'], game['teams']['away']['team']['name'], '@', game['teams']['home']['team']['name'])
        print('Away:', game['teams']['away'].get('probablePitcher', 'Unknown Pitcher'))
        print('Home:', game['teams']['home'].get('probablePitcher', 'Unknown Pitcher'))