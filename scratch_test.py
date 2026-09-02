import os
import sys
import pandas as pd
import numpy as np

# Let's import the data generation part from app.py
sys.path.append('k:/app/3. market 코스피 전망')
import app

print("total_risk_index:", app.total_risk_index)
print("base_risk:", app.base_risk)
print("applied_base_risk:", app.applied_base_risk)
print("final_panic_score:", app.final_panic_score)
print("hist_risks[-1]:", app.hist_risks[-1])
print("Is total_risk_index NaN?", np.isnan(app.total_risk_index))
