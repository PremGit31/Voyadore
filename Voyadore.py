import streamlit as s
from dotenv import load_dotenv 

import os
load_dotenv()
geminiApi = os.getenv('GeminiAPI')
serpApi = os.getenv('SerpAPI')
# TavilySearch automatically searches Tavily API key in .env. Also the api key must be
# saved like 'TAVILY_API_KEY' in .env
# tavilyApi = os.getenv('TAVILY_API_KEY') 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

s.set_page_config(page_title='Voyadore by Prem', page_icon='🌏')
s.title('Voyadore🌏')
s.header('Plan voyages swiftly!:rocket:')

c1, c2 = s.columns([2,2])
with c1:
  source = s.text_input('Source')
  dest = s.text_input('Destination')
  fromDate = s.date_input('From')
  ToDate = s.date_input('To')
with c2:
  selected_options = s.multiselect('Find:', ['Youtube Videos','Websites'])
  preferences = s.text_input('Let me know your preferences!')
  uploaded_file = s.file_uploader('Upload your travel document')
  
with s.sidebar:
        s.title("🌟 About Voyadore")
        s.header('Designed with 🤍 by Prem Shirsathe, student at KJSIT Mumbai.')
        sidebar_tab1, sidebar_tab2 = s.tabs(["📋 Features & Usage", "💻 Development Info"])

        with sidebar_tab1:
          s.subheader("📋 Features")
          s.markdown("""
          - 🌍 **Plan Your Trip**: Enter your source, destination, and travel dates to get personalized suggestions.
          - 📄 **PDF Summarization**: Upload travel documents for quick summaries and key points.
          - 🔍 **Travel Resources**: Search for YouTube videos and websites about your destination.
          - 🗺️ **Interactive Map**: Explore nearby attractions, restaurants, and more with a customizable map.
          - 🤖 **AI-Powered Chat**: Get travel advice and itineraries from an AI travel companion.
          """)

          s.subheader("🛠️ How to Use")
          s.markdown("""
          1. Fill in the main form with your travel details.
          2. Upload a travel document (optional) for summarization.
          3. Use the **Conversation** tab to chat with the AI for travel advice.
          4. Explore travel resources in the **Sources** tab.
          5. Check out nearby places on the **Maps** tab.
          6. Customize your map settings and categories to find the best spots.
          """)

        with sidebar_tab2:
          s.subheader("💻 How It Works")
          s.markdown("""
          - **Streamlit Framework**: The app is built using the Streamlit library for an interactive UI.
          - **LangChain Integration**: Utilizes LangChain for AI-driven conversations and PDF summarization.
          - **Tavily Search**: Fetches flight information using the Tavily API.
          - **SerpAPI**: Searches for travel resources like YouTube videos and websites.
          - **Folium & Geopy**: Generates interactive maps and locates destinations.
          - **Environment Variables**: API keys are securely loaded using `dotenv`.
          """)

          s.subheader("📦 Key Libraries")
          s.markdown("""
          - `streamlit`, `streamlit-folium`
          - `langchain`, `langchain-google-genai`, `langchain-tavily`
          - `PyPDF2`, `folium`, `geopy`, `requests`
          """)

          s.subheader("📧 Contact")
          s.markdown("""
          Have feedback or questions? Reach out to me at **premshirsathe@outlook.com**.
          """)

gemini = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-001',
    temperature=0.5,
    convert_system_message_to_human=True,
    google_api_key=geminiApi
)
  
if uploaded_file is not None:
    try:
        import PyPDF2

        # Read the uploaded PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""

        # Extract text from each page
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Pass the extracted text to the LLM for summarization
        with s.spinner("🤖 Summarizing the PDF content..."):
            llm_input = f"Summarize the following PDF content and extract key points:\n\n{pdf_text}"            
            response = gemini.invoke(llm_input)

        # Display the summary and key points in a text area
        with s.expander("📄 Summary and Key Points"):
            s.text_area("Summary", response.content, height=300)

        s.success("✅ PDF processed and summarized successfully!")
    except Exception as e:
        s.error(f"❌ Error processing PDF: {str(e)}") 
        
from langchain_tavily import TavilySearch

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

tavily_output1 = tool.invoke({
    'query': f"What is the economical and convenient flight from {source} to {dest} on {fromDate}?"
})

tavily_output2 = tool.invoke({
  'query': f"What is the economical and convenient flight from {source} to {dest} on {ToDate}?"
})

tavily_output3 = tool.invoke({
  'query': f'What are some of the best hotels in and around {dest}'
})

SYS_PROMPT = f"""
              You are a friendly, honest, helpful travel companion! Be cheerful and correct.
              Consider the following information given by user and guide them:
              {source}
              {dest}
              {fromDate}
              {ToDate}\n
              Suggest a good itinery to the users considering their {preferences}
              After mentioning the above things do this:
              Based on the following information, what is the economical and convenient
              flight from {source} to {dest} on {fromDate}? 
              Check each line before answering \n\n{str(tavily_output1).replace('{', '{{').replace('}', '}}')} 
              
              Based on the following information, what is the economical and convenient
              flight from {source} to {dest} on {ToDate}? 
              Check each line before answering \n\n{str(tavily_output2).replace('{', '{{').replace('}', '}}')}
              Based on the following information, what are some of the best hotels
              in and around {dest}:
              Check each line before answering \n\n{str(tavily_output3).replace('{', '{{').replace('}', '}}')}
              """

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", SYS_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
  ]
)

chain = (prompt | gemini)

streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
  chain,
  lambda session_id: streamlit_msg_history,  # Accesses memory
  input_messages_key="input",
  history_messages_key="history",
)

tab1, tab2, tab3, tab4 = s.tabs(['Conversation','Sources','Maps','Weather'])

with tab1:
  for msg in streamlit_msg_history.messages:
    s.chat_message(msg.type,).write(msg.content)


  # If user inputs a new prompt, display it and show the response
  if user_prompt := s.chat_input():
    s.chat_message("human").write(user_prompt)
    # This is where response from the LLM is shown
    with s.chat_message("ai"):
      config = {"configurable": {"session_id": "any"}}
      # Get llm response
      response = conversation_chain.invoke({"input": user_prompt}, config)
      s.markdown(response.content) # Display response directly
    
with tab2:
  from langchain_community.utilities import SerpAPIWrapper
  
  # Check if destination is provided
  if not dest:
    s.info("⚠️ Please enter a destination in the main form to search for travel content.")
  else:
    # Create search queries based on user selections
    search_queries = []
    
    if 'Youtube Videos' in selected_options:
      search_queries.append(f'Best things to do in {dest} travel guide site:youtube.com')
      search_queries.append(f'{dest} travel vlog attractions site:youtube.com')
    
    if 'Websites' in selected_options:
      search_queries.append(f'Best things to do in {dest} travel guide')
      search_queries.append(f'{dest} tourist attractions travel tips')
    
    # If no options selected, default to both
    if not selected_options:
      search_queries = [
        f'Best things to do in {dest} travel guide site:youtube.com',
        f'{dest} travel attractions tourist guide'
      ]
    
    # Display search interface
    s.subheader(f'🔍 Travel Resources for {dest}')
    
    # Create columns for search options
    col1, col2 = s.columns([3, 1])
    
    with col1:
      custom_query = s.text_input(
        'Custom search (optional):', 
        placeholder=f'e.g., "best restaurants in {dest}"'
      )
    
    with col2:
      s.write("")  # Spacer
      search_button = s.button('🔍 Search', type="primary")
    
    if search_button:
      try:
        # Use custom query if provided, otherwise use generated queries
        queries_to_search = [custom_query] if custom_query else search_queries
        
        with s.spinner('🔍 Searching for travel resources...'):
          serpapi = SerpAPIWrapper(serpapi_api_key=serpApi)
          
          all_results = []
          youtube_results = []
          website_results = []
          
          for query in queries_to_search:
            try:
              res = serpapi.results(query)
              organic = res.get('organic_results', [])
              
              # Separate YouTube and website results
              for result in organic:
                link = result.get('link', '')
                if 'youtube.com/watch' in link:
                  youtube_results.append(result)
                else:
                  website_results.append(result)
              
              all_results.append(res)
            except Exception as e:
              s.error(f"Error searching for '{query}': {str(e)}")
        
        # Display results based on user selection
        if not youtube_results and not website_results:
          s.warning('🤷‍♂️ No results found. Try a different search term.')
        else:
          # Display YouTube Videos if selected or found
          if ('Youtube Videos' in selected_options or not selected_options) and youtube_results:
            s.subheader('🎥 YouTube Travel Videos')
            
            for i, result in enumerate(youtube_results[:5]):  # Limit to 5 videos
              title = result.get('title', 'No title')
              link = result.get('link', '')
              snippet = result.get('snippet', 'No description available')
              
              # Extract video ID and display
              if 'youtube.com/watch' in link:
                try:
                  video_id = link.split('v=')[-1].split('&')[0]
                  iframe_url = f'https://www.youtube.com/embed/{video_id}'
                  
                  with s.container():
                    s.markdown(f"**{i+1}. [{title}]({link})**")
                    s.write(snippet[:200] + "..." if len(snippet) > 200 else snippet)
                    s.video(iframe_url)
                    s.divider()
                except Exception as e:
                  s.markdown(f"**{i+1}. [{title}]({link})**")
                  s.write(snippet)
                  s.error(f"Could not load video preview")
          
          # Display Website Results if selected or found
          if ('Websites' in selected_options or not selected_options) and website_results:
            s.subheader('🌐 Travel Websites & Articles')
            
            for i, result in enumerate(website_results[:8]):  # Limit to 8 websites
              title = result.get('title', 'No title')
              link = result.get('link', '')
              snippet = result.get('snippet', 'No description available')
              displayed_link = result.get('displayed_link', link)
              
              with s.container():
                s.markdown(f"**{i+1}. [{title}]({link})**")
                s.caption(f"🔗 {displayed_link}")
                s.write(snippet)
                s.divider()
          
          # Display quick answers/featured snippets
          for res in all_results:
            answer_displayed = False
            
            # Check for answer box
            if 'answer_box' in res and 'snippet' in res['answer_box']:
              if not answer_displayed:
                s.success("💡 **Quick Travel Tip:**")
                s.write(res['answer_box']['snippet'])
                answer_displayed = True
                break
            
            # Check for knowledge graph
            elif 'knowledge_graph' in res and 'description' in res['knowledge_graph']:
              if not answer_displayed:
                s.info("📍 **About the Destination:**")
                s.write(res['knowledge_graph']['description'])
                answer_displayed = True
                break
          
          # Show search summary
          total_results = len(youtube_results) + len(website_results)
          s.caption(f"Found {total_results} relevant travel resources for {dest}")
      
      except ImportError:
        s.error("❌ SerpAPI wrapper not available. Please install: `pip install langchain-community`")
      except Exception as e:
        s.error(f"❌ Search failed: {str(e)}")
        s.info("Please check your SerpAPI key and internet connection.")
  
  # Add helpful travel planning tips in sidebar or expander
  with s.expander("💡 Travel Planning Tips"):
    s.markdown(f"""
    **Planning your trip to {dest if dest else '[Destination]'}:**
    - 🎥 Watch travel vlogs for insider tips
    - 📱 Check recent reviews on travel sites
    - 🗺️ Look for local recommendations
    - 📅 Consider seasonal factors: {fromDate} to {ToDate if 'ToDate' in locals() else 'N/A'}
    
    **Search suggestions:**
    - "hidden gems in {dest if dest else '[destination]'}"
    - "local food {dest if dest else '[destination]'}"
    - "budget travel {dest if dest else '[destination]'}"
    """)

with tab3:
  import folium
  from streamlit_folium import st_folium
  import requests
  import json
  from geopy.geocoders import Nominatim
  
  s.subheader('🗺️ Interactive Travel Map')
  
  # Check if destination is provided
  if not dest:
    s.info("⚠️ Please enter a destination in the main form to explore places on the map.")
  else:
    try:
      # Initialize geocoder
      geolocator = Nominatim(user_agent="voyadore_travel_app")
      
      # Get coordinates for the destination
      with s.spinner(f'🌍 Finding location coordinates for {dest}...'):
        location = geolocator.geocode(dest)
        
        if not location:
          s.error(f"❌ Could not find coordinates for '{dest}'. Please try a more specific location.")
        else:
          dest_coords = [location.latitude, location.longitude]
          
          # Map configuration options
          s.markdown("### 🎛️ Map Settings")
          col1, col2, col3 = s.columns(3)
          
          with col1:
            map_style = s.selectbox(
              "Map Style:",
              ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter", "Stamen Terrain"],
              index=0
            )
          
          with col2:
            search_radius = s.slider("Search Radius (km):", 1, 20, 5)
          
          with col3:
            max_places = s.slider("Max Places to Show:", 5, 50, 20)
          
          # Place categories to search for
          s.markdown("### 📍 Select Place Categories")
          categories = s.multiselect(
            "Choose categories to display:",
            [
              "🏨 Hotels & Accommodation",
              "🍽️ Restaurants & Cafes", 
              "🎭 Attractions & Entertainment",
              "🛍️ Shopping & Markets",
              "🏥 Healthcare & Services",
              "🚌 Transportation",
              "🏛️ Museums & Culture",
              "🌳 Parks & Recreation"
            ],
            default=["🏨 Hotels & Accommodation", "🍽️ Restaurants & Cafes", "🎭 Attractions & Entertainment"]
          )
          
          if s.button("🗺️ Generate Interactive Map", type="primary"):
            with s.spinner('🔍 Finding nearby places...'):
              
              # Map style configuration
              map_styles = {
                "OpenStreetMap": None,
                "CartoDB Positron": "cartodbpositron", 
                "CartoDB Dark_Matter": "cartodbdark_matter",
                "Stamen Terrain": "stamenterrain"
              }
              
              # Create the map
              m = folium.Map(
                location=dest_coords, 
                zoom_start=13,
                tiles=map_styles[map_style] if map_styles[map_style] else "OpenStreetMap"
              )
              
              # Add main destination marker
              folium.Marker(
                location=dest_coords,
                popup=f"<b>📍 {dest}</b><br>Your Destination",
                tooltip=f"🎯 {dest}",
                icon=folium.Icon(color="red", icon="star", prefix="fa")
              ).add_to(m)
              
              # Category mapping to Overpass API queries
              category_queries = {
                "🏨 Hotels & Accommodation": 'node["tourism"~"hotel|hostel|guest_house|apartment"]',
                "🍽️ Restaurants & Cafes": 'node["amenity"~"restaurant|cafe|fast_food|bar|pub"]',
                "🎭 Attractions & Entertainment": 'node["tourism"~"attraction|museum|zoo|theme_park"]',
                "🛍️ Shopping & Markets": 'node["shop"]["shop"!="vacant"]',
                "🏥 Healthcare & Services": 'node["amenity"~"hospital|pharmacy|clinic|bank|atm"]',
                "🚌 Transportation": 'node["amenity"~"bus_station|taxi"]',
                "🏛️ Museums & Culture": 'node["tourism"~"museum|gallery|monument"]',
                "🌳 Parks & Recreation": 'node["leisure"~"park|garden|playground|sports_centre"]'
              }
              
              # Icons and colors for different categories
              category_styles = {
                "🏨 Hotels & Accommodation": {"color": "blue", "icon": "bed"},
                "🍽️ Restaurants & Cafes": {"color": "green", "icon": "cutlery"},
                "🎭 Attractions & Entertainment": {"color": "red", "icon": "camera"},
                "🛍️ Shopping & Markets": {"color": "purple", "icon": "shopping-cart"},
                "🏥 Healthcare & Services": {"color": "pink", "icon": "plus"},
                "🚌 Transportation": {"color": "gray", "icon": "bus"},
                "🏛️ Museums & Culture": {"color": "orange", "icon": "university"},
                "🌳 Parks & Recreation": {"color": "darkgreen", "icon": "tree"}
              }
              
              all_places = []
              
              # Search for places in each selected category
              for category in categories:
                if category in category_queries:
                  try:
                    # Overpass API query
                    overpass_url = "http://overpass-api.de/api/interpreter"
                    overpass_query = f"""
                    [out:json][timeout:25];
                    (
                      {category_queries[category]}(around:{search_radius * 1000},{dest_coords[0]},{dest_coords[1]});
                    );
                    out center meta;
                    """
                    
                    response = requests.get(overpass_url, params={'data': overpass_query})
                    data = response.json()
                    
                    places_added = 0
                    for element in data.get('elements', []):
                      if places_added >= max_places // len(categories):
                        break
                        
                      if 'lat' in element and 'lon' in element:
                        tags = element.get('tags', {})
                        name = tags.get('name', 'Unnamed Place')
                        
                        # Skip if no name available
                        if name == 'Unnamed Place' and not any(key in tags for key in ['shop', 'amenity', 'tourism']):
                          continue
                        
                        place_type = tags.get('amenity') or tags.get('tourism') or tags.get('shop') or tags.get('leisure', 'place')
                        
                        popup_content = f"""
                        <b>{name}</b><br>
                        <i>{category.split(' ')[1]} {category.split(' ')[2] if len(category.split(' ')) > 2 else ''}</i><br>
                        Type: {place_type}<br>
                        """
                        
                        # Add additional info if available
                        if 'addr:street' in tags:
                          popup_content += f"📍 {tags['addr:street']}<br>"
                        if 'phone' in tags:
                          popup_content += f"📞 {tags['phone']}<br>"
                        if 'website' in tags:
                          popup_content += f"🌐 <a href='{tags['website']}' target='_blank'>Website</a><br>"
                        
                        style = category_styles.get(category, {"color": "blue", "icon": "info-sign"})
                        
                        folium.Marker(
                          location=[element['lat'], element['lon']],
                          popup=folium.Popup(popup_content, max_width=300),
                          tooltip=name,
                          icon=folium.Icon(
                            color=style["color"],
                            icon=style["icon"],
                            prefix="fa"
                          )
                        ).add_to(m)
                        
                        all_places.append({
                          'name': name,
                          'category': category,
                          'coords': [element['lat'], element['lon']],
                          'type': place_type
                        })
                        places_added += 1
                        
                  except Exception as e:
                    s.warning(f"Could not fetch data for {category}: {str(e)}")
              
              # Add a circle to show search radius
              folium.Circle(
                location=dest_coords,
                radius=search_radius * 1000,  # Convert km to meters
                popup=f"Search Area ({search_radius} km radius)",
                color="blue",
                fill=True,
                opacity=0.2,
                fillOpacity=0.1
              ).add_to(m)
              
              # Display the map
              s.markdown("### 🗺️ Interactive Map")
              map_data = st_folium(m, width=800, height=600, returned_objects=["popup", "last_clicked"])
              
              # Show information about clicked places
              if map_data.get('last_object_clicked_popup'):
                s.markdown("### 📍 Selected Place Details")
                s.info(f"Last clicked: {map_data['last_object_clicked_popup']}")
              
              # Display places summary
              if all_places:
                s.markdown("### 📋 Places Found")
                
                # Group places by category
                places_by_category = {}
                for place in all_places:
                  category = place['category']
                  if category not in places_by_category:
                    places_by_category[category] = []
                  places_by_category[category].append(place)
                
                # Display in expandable sections
                for category, places in places_by_category.items():
                  with s.expander(f"{category} ({len(places)} found)"):
                    for i, place in enumerate(places, 1):
                      s.write(f"{i}. **{place['name']}** - {place['type']}")
                
                s.success(f"✅ Found {len(all_places)} places near {dest}")
              else:
                s.warning("No places found in the selected categories. Try expanding your search radius or selecting different categories.")
          
          # Travel tips based on destination
          with s.expander("💡 Map Navigation Tips"):
            s.markdown(f"""
            **How to use the map:**
            - 🖱️ Click on markers to see place details
            - 🔍 Zoom in/out using mouse wheel or map controls
            - 🎯 The red star shows your destination: **{dest}**
            - 📍 Different colored markers represent different place categories
            - 🔵 Blue circle shows your search radius ({search_radius} km)
            
            **Travel Planning Tips:**
            - Look for clusters of restaurants and attractions
            - Check distances between your hotel and points of interest
            - Note transportation hubs for easy movement
            - Save interesting places by taking screenshots
            """)
    
    except ImportError as e:
      s.error("❌ Required packages missing. Please install: `pip install folium geopy streamlit-folium requests`")
    except Exception as e:
      s.error(f"❌ Error creating map: {str(e)}")
      s.info("Please check your internet connection and try again.")
      
with tab4:
  s.write('Note that only current weather info is available')
  s.caption('Please enter the name of a nearby major city if the info for your city is not showing.')
  weatherApi = os.getenv('WeatherAPI')
  
  import requests 
  import json
  
  BASE_URL = "http://api.weatherapi.com/v1/current.json"

  def get_weather(city):
    """Get current weather for a city"""
    params = {
        'key': weatherApi,
        'q': city,
        'aqi': 'no'
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract key info
        weather_info = {
            'location': data['location']['name'],
            'country': data['location']['country'],
            'temperature': data['current']['temp_c'],
            'condition': data['current']['condition']['text'],
            'humidity': data['current']['humidity']
        }

        return weather_info

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

  # Test the function
  if __name__ == "__main__":
    if dest:
      with s.spinner(f"🌤️ Fetching weather information for {dest}..."):
        weather_info = get_weather(dest)
        if isinstance(weather_info, dict):
          llm_input = f"Summarize the weather: {weather_info}"
          response = gemini.invoke(llm_input)
          s.subheader(f"🌤️ Weather Summary for {dest}")
          s.write(response.content)
        else:
          s.error(f"❌ Unable to fetch weather data: {weather_info}")
    else:
      s.info("⚠️ Please enter a destination in the main form to get weather information.")