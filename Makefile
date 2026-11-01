D := examples/data
R := examples/reports

.PHONY: install dev data demo test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

data:
	python examples/make_sample_data.py

demo: data   ## run every command on the sample data → examples/reports/
	dskit profile  $(D)/customers_churn.csv --target churn -o $(R)/01_profile.html
	-dskit validate $(D)/customers_churn.csv --rules examples/rules/customers_churn.yaml -o $(R)/02_validation.html
	dskit clean    $(D)/customers_churn.csv -o $(D)/customers_churn_clean.csv --report $(R)/03_cleaning.html
	dskit baseline $(D)/customers_churn.csv --target churn -o $(R)/04_baseline_churn.html --model-out examples/models/churn.joblib
	dskit baseline $(D)/house_prices.csv --target price_idr -o $(R)/05_baseline_house_prices.html
	dskit predict  examples/models/churn.joblib $(D)/customers_churn.csv -o $(D)/churn_scored.csv
	dskit forecast $(D)/daily_sales.csv --date date --value orders --horizon 30 -o $(R)/06_forecast.html --csv $(D)/forecast.csv
	dskit abtest   $(D)/ab_test.csv --group group --metric converted --segment device --guardrail revenue -o $(R)/07_abtest.html
	dskit rfm      $(D)/transactions.csv --customer customer_id --date order_date --amount amount --currency Rp -o $(R)/08_rfm.html --csv $(D)/rfm_customers.csv

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf reports data/processed examples/models .pytest_cache
