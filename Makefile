create-data:
	python src/_download_data.py && \
	python src/data_cleaning.py
