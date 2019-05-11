dev-init-mac: 
	python3 -m venv p36 --system-site-packages
	pip install -U pip
	pip install --upgrade -r requirements.txt

prepare-ubuntu:
	sudo apt-get install python3-venv

dev-init-ubuntu:
	sudo python3 -m venv p36 --system-site-packages
	sudo pip install -U pip
	sudo pip install --upgrade -r requirements.txt
