import requests
from requests.auth import AuthBase
import base64
import pandas as pd
from utils import get_lubw_config

# Custom authentication class to handle UTF-8 encoding for the password
class UTF8BasicAuth(AuthBase):
    """Attaches HTTP Basic Authentication Header with UTF-8 encoding."""
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __call__(self, r):
        # Encode credentials with UTF-8
        auth_str = f'{self.username}:{self.password}'
        b64_encoded = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
        r.headers['Authorization'] = f'Basic {b64_encoded}'
        return r

# Authentication details
username = "hs-heilbronn"
password = "Vc+CW€Fr\P"

# Base URL for the API
base_url = "https://mersyzentrale.de/www/Datenweitergabe/Luft/data.php"

# Define components for each station
station_components = {
    "DEBW015": ['PM10', 'PM2.5', 'NO2', 'O3', 'TEMP', 'RLF', 'NSCH', 'STRG', 'WIV'],
    "DEBW152": ['NO2', 'CO']
}


# Function to fetch data for a station and time range
def fetch_station_data(station, start_time, end_time):
    # Get the components for the specified station
    components = station_components.get(station)

    if components is None:
        raise ValueError(f"Unknown station: {station}")

    all_data = {}  # Dictionary to store component data, keyed by datetime

    # Loop over components for the station
    for component in components:
        params = {
            'komponente': component,
            'von': start_time,
            'bis': end_time,
            'station': station
        }

        # Continue fetching data until no nextLink is provided
        next_link = None

        while True:
            try:
                # If there's a next link, use it; otherwise, use the base URL with params
                if next_link:
                    response = requests.get(next_link, auth=UTF8BasicAuth(username, password))
                else:
                    response = requests.get(base_url, params=params, auth=UTF8BasicAuth(username, password))

                response.raise_for_status()  # Raise an error for bad responses (4XX, 5XX)
                response.encoding = 'utf-8'

                data = response.json()

                # Debugging: Print out the structure of the response for inspection
                print(f"Component: {component} | Station: {station} | Data:")
                print(data)  # Print the actual response

                # Ensure the data is valid and contains 'messwerte'
                if 'messwerte' not in data or not isinstance(data['messwerte'], list):
                    print(f"No 'messwerte' found for component {component} at station {station}")
                    break  # Stop if there's no data for this component

                # Process each measurement (messwert)
                for entry in data['messwerte']:
                    # Extract 'startZeit' as the datetime and 'wert' as the value
                    dt = entry['startZeit']  # Use 'startZeit' for datetime
                    value = entry['wert']  # Use 'wert' for the value

                    # Add datetime as a key, and create an empty dict for that datetime if not present
                    if dt not in all_data:
                        all_data[dt] = {'datetime': dt}

                    # Add the value for the current component to the dictionary
                    all_data[dt][component] = value

                # Check for the nextLink to see if more data needs to be fetched
                next_link = data.get('nextLink')

                # Break if no more nextLink is present
                if not next_link:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for component {component} at station {station}: {e}")
                return None

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(all_data.values()))

    # Sort by datetime if needed
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime').reset_index(drop=True)

    return df


# Example Usage:
station = "DEBW015"
start_time = "2024-11-15T00:00:00"
end_time = "2024-11-16T00:00:00"

df = fetch_station_data(station, start_time, end_time)

# Display the DataFrame
# print(df)

# Optionally, save to CSV:x
# df.to_csv(f"/home/garc/DEBW152_20241115-20241231.csv", index=False, encoding="utf-8")