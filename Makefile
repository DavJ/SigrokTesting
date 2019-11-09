dev-init-mac: 
	python3 -m venv p36 --system-site-packages
	pip install -U pip
	pip install --upgrade -r requirements.txt

prepare-ubuntu:
	sudo apt-get install python3-venv

dev-init-ubuntu:
	pip3 install --user virtualenv
	virtualenv --python=/usr/bin/python3.6 python36	
	source ./python36/bin/activate
	sudo pip install --upgrade --ignore-installed  -r requirements.txt
