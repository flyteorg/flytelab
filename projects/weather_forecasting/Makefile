venv:
	@virtualenv ./.venv/weather-forecasting

deps:
	@pip install -r requirements.txt

env.txt:
	@touch env.txt
	@echo "NOAA_API_KEY='<API_KEY>'"

.PHONY: env-export
env-export:
	@eval $(sed 's/^/export /g' env.txt)
