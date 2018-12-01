dev-init: 
	python3 -m venv p36 --system-site-packages
	pip install -U pip
	pip install --upgrade -r requirements.txt

