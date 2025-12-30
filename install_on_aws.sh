#!/bin/bash
set -e

echo "=========================================="
echo "🚀 TENDER AUTOMATION INSTALLER FOR AWS"
echo "=========================================="

# 1. Update System
echo "[1/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Dependencies (Git, Python, Pip, Venv)
echo "[2/7] Installing Python & Git..."
sudo apt install -y git python3-pip python3-venv

# 3. Install Docker & Docker Compose
echo "[3/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt install -y docker.io docker-compose
    sudo systemctl enable --now docker
    # Add ubuntu user to docker group
    sudo usermod -aG docker ubuntu || true
else
    echo "Docker already installed."
fi

# 4. Install MySQL (Server)
echo "[4/7] Installing MySQL Server..."
if ! command -v mysql &> /dev/null; then
    sudo apt install -y mysql-server
    sudo systemctl start mysql.service
else
    echo "MySQL already installed."
fi

# 5. Database Setup (Create User & DB)
echo "[5/7] Configuring Database..."
# Change 'StrongPassword@123' to something strictly secret in production!
sudo mysql -e "CREATE DATABASE IF NOT EXISTS tender_automation_with_ai;"
sudo mysql -e "CREATE USER IF NOT EXISTS 'tender_user'@'%' IDENTIFIED BY 'StrongPassword@123';"
sudo mysql -e "GRANT ALL PRIVILEGES ON tender_automation_with_ai.* TO 'tender_user'@'%';"
sudo mysql -e "FLUSH PRIVILEGES;"

# Configure MySQL to listen on all interfaces (for Docker access)
# WARNING: In production, use AWS Security Groups to restrict access to port 3306!
echo "[INFO] Updating MySQL bind-address to 0.0.0.0 for Docker access..."
sudo sed -i 's/bind-address.*/bind-address = 0.0.0.0/' /etc/mysql/mysql.conf.d/mysqld.cnf
sudo systemctl restart mysql

# 6. Python Environment Setup
echo "[6/7] Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing Requirements..."
# Install standard libs
pip install -r requirements.txt
# Install Playwright browsers for the scraper
playwright install --with-deps chromium

# 7. Create Tables
echo "[7/7] Creating Database Tables..."
python create_missing_table.py

echo "=========================================="
echo "✅ INSTALLATION COMPLETE!"
echo "=========================================="
echo "Next Steps:"
echo "1. Log out and log back in (to apply Docker group changes)."
echo "2. Activate venv: source venv/bin/activate"
echo "3. Start Workers: docker-compose up -d --scale worker=20"
echo "4. Start Scraper: python run.py"
