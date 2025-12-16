PYTHONPATH := src

setup:
	python -m pip install -r requirements.txt

index:
	PYTHONPATH=$(PYTHONPATH) python src/index.py

app:
	PYTHONPATH=$(PYTHONPATH) streamlit run app/streamlit_app.py

eval:
	PYTHONPATH=$(PYTHONPATH) python src/eval.py

.PHONY: setup index app eval