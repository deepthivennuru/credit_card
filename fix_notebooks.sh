#!/bin/bash

# Script to fix notebook sharing issues

echo "Fixing notebook sharing issues..."

# Backup the original notebooks if needed (just in case)
echo "Creating backup of original notebooks..."
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/01_exploratory_data_analysis.ipynb" "/Users/chethan/Desktop/credit-card-fraud/notebooks/01_exploratory_data_analysis.ipynb.bak" 2>/dev/null
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/02_model_development.ipynb" "/Users/chethan/Desktop/credit-card-fraud/notebooks/02_model_development.ipynb.bak" 2>/dev/null

# Copy the fixed notebooks to replace the original ones
echo "Copying fixed notebooks..."
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/fixed_eda.ipynb" "/Users/chethan/Desktop/credit-card-fraud/notebooks/01_exploratory_data_analysis.ipynb"
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/fixed_model.ipynb" "/Users/chethan/Desktop/credit-card-fraud/notebooks/02_model_development.ipynb"

# Also update backup_notebooks folder to ensure consistency
echo "Updating backup notebooks folder..."
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/fixed_eda.ipynb" "/Users/chethan/Desktop/credit-card-fraud/backup_notebooks/01_exploratory_data_analysis.ipynb"
cp -f "/Users/chethan/Desktop/credit-card-fraud/notebooks/fixed_model.ipynb" "/Users/chethan/Desktop/credit-card-fraud/backup_notebooks/02_model_development.ipynb"

echo "Notebook fix completed!"
echo "You can now zip your project and share it with your friend."
echo "All notebooks should now display properly with all cells included."
