import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

# Function to display the JavaScript widget
def display_location_widget():
    # Load the HTML file content
    with open("location_widget.html", "r") as file:
        html_code = file.read()
    components.html(html_code, height=100)

# Function to display map with Folium
def display_map(lat, lon):
    # Create a Folium map centered on the given latitude and longitude
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        [lat, lon],
        popup="Reported Damage Location",
        icon=folium.Icon(color="red")
    ).add_to(m)
    # Display the map in the Streamlit app
    st_folium(m, width=700, height=500)

# Streamlit app title
st.title("Roadway Infrastructure Monitoring System")

st.sidebar.header("Upload Image/Video or Provide a URL")

# Display the location widget
st.subheader("Get Your Location")
display_location_widget()

# JavaScript to Python communication
# Capture the message from the JavaScript code
message = st.experimental_get_query_params()
if 'latitude' in message and 'longitude' in message:
    latitude = float(message['latitude'][0])
    longitude = float(message['longitude'][0])
    st.write(f"Detected Location: Latitude {latitude}, Longitude {longitude}")

    # Display the map with the detected coordinates
    display_map(latitude, longitude)

