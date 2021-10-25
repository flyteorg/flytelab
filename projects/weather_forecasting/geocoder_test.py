from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="custom-flytelab-foobar")
location = geolocator.geocode("175 5th Avenue NYC")
print(location)
