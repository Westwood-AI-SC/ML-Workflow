# Update package list
sudo apt-get update -y
sudo apt-get upgrade -y

# Install necessary packages
chmod +x ./scripts/install_dependencies.sh
./scripts/install_dependencies.sh

# create .venv
virtualenv .venv --python=python3
source .venv/bin/activate
pip install -r requirements.txt

