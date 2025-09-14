#!/bin/bash

# Script to create a zip file with the complete notebook files

echo "Creating zip file for sharing..."

# First run the fix script to ensure notebooks are properly updated
./fix_notebooks.sh

# Create a temp directory
mkdir -p /tmp/credit-card-fraud-share

# Copy all important files
cp -r api models notebooks frontend *.sh *.csv *.txt /tmp/credit-card-fraud-share/

# Create zip file
cd /tmp
zip -r credit-card-fraud-complete.zip credit-card-fraud-share

# Move zip back to project directory
mv credit-card-fraud-complete.zip /Users/chethan/Desktop/credit-card-fraud/

# Clean up
rm -rf /tmp/credit-card-fraud-share

echo "Zip file created at: /Users/chethan/Desktop/credit-card-fraud/credit-card-fraud-complete.zip"
echo "You can now share this zip file with your friend. All notebooks will be complete!"
