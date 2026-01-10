#!/bin/bash
#
# GPU Metrics Setup Script for macOS
# This script configures passwordless sudo for powermetrics
# Required for GPU power draw measurements
#

echo "========================================="
echo "GPU Metrics Setup Script (macOS)"
echo "========================================="
echo ""

# Check if running on macOS
if [[ ! "$OSTYPE" == "darwin"* ]]; then
  echo "❌ Error: This script is for macOS only"
  echo "   For Linux/Windows, ensure nvidia-smi is installed"
  exit 1
fi

# Check if powermetrics is available
if ! command -v powermetrics &> /dev/null; then
  echo "❌ Error: powermetrics not found"
  echo "   This is a system tool on macOS and should be available"
  exit 1
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
  echo "⚠️  This script needs to modify sudoers file"
  echo "   You may be prompted for your password"
  echo ""
  # Re-run with sudo using the full script path
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  sudo bash "$SCRIPT_PATH"
  exit $?
fi

# Get the original user (if running via sudo)
SUDO_USER=${SUDO_USER:-$(whoami)}
ORIGINAL_UID=$(id -u $SUDO_USER 2>/dev/null)

if [[ -z "$ORIGINAL_UID" ]]; then
  echo "❌ Error: Could not determine user"
  exit 1
fi

echo "Setting up passwordless sudo for powermetrics..."
echo "User: $SUDO_USER"
echo ""

# Create temporary sudoers file
TEMP_SUDOERS=$(mktemp)
trap "rm -f $TEMP_SUDOERS" EXIT

# Copy current sudoers
sudo cp /etc/sudoers "$TEMP_SUDOERS"

# Check if already configured
if sudo grep -q "^$SUDO_USER ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" "$TEMP_SUDOERS"; then
  echo "✓ Already configured! Passwordless sudo for powermetrics is active."
  exit 0
fi

# Add new line to sudoers
echo "$SUDO_USER ALL=(ALL) NOPASSWD: /usr/bin/powermetrics" | sudo tee -a "$TEMP_SUDOERS" > /dev/null

# Validate new sudoers file
if ! sudo visudo -c -f "$TEMP_SUDOERS" > /dev/null 2>&1; then
  echo "❌ Error: Failed to validate sudoers file"
  exit 1
fi

# Copy validated sudoers back to /etc/sudoers
sudo cp "$TEMP_SUDOERS" /etc/sudoers

echo ""
echo "✓ Setup complete!"
echo ""
echo "Verification:"
echo "  Run: sudo -n powermetrics -s gpu_power -n 1"
echo "  This should work without prompting for password"
echo ""
echo "To test GPU metrics collection:"
echo "  node metrics-server.js &"
echo "  node -e \"const GPU = require('./gpu_metrics_collector.js'); const g = new GPU(); g.startMonitoring(); setTimeout(() => { const m = g.stopMonitoring(); console.log(m); }, 5000);\""
echo ""
