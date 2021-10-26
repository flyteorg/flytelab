from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="custom-flytelab", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=10)
location = geolocator.geocode("175 5th Avenue NYC")
print(location)
