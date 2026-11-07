"""Creates the bundled sample datasets (run once: python data/generate_sample_data.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

OUT = Path(__file__).parent

# 1. Classic "Play Tennis" (purely categorical - ideal for ID3)
pd.DataFrame(
    [r.split(",") for r in """Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No""".splitlines()],
    columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"],
).to_csv(OUT / "play_tennis.csv", index=False)

# 2. Iris (purely numeric - ideal for C4.5)
iris = load_iris(as_frame=True)
df = iris.data.copy()
df["species"] = iris.target.map(dict(enumerate(iris.target_names)))
df.to_csv(OUT / "iris.csv", index=False)

# 3. Synthetic credit-approval data (mixed types + missing values)
rng = np.random.default_rng(42)
n = 600
age = rng.integers(21, 66, n)
income = rng.normal(55000, 18000, n).clip(15000, None).round(-2)
loan = rng.normal(15000, 7000, n).clip(1000, None).round(-2)
employment = rng.choice(["salaried", "self-employed", "unemployed", "student"], n, p=[.5, .25, .1, .15])
home = rng.choice(["own", "rent", "mortgage"], n, p=[.3, .4, .3])
history = rng.choice(["good", "average", "poor"], n, p=[.5, .3, .2])
score = (
    (income / loan) * 0.8
    + np.select([history == "good", history == "average"], [2.0, 0.8], 0.0)
    + np.select([employment == "salaried", employment == "self-employed", employment == "student"], [1.2, 0.6, 0.1], -1.0)
    + (age > 30) * 0.4 + rng.normal(0, 0.7, n)
)
approved = np.where(score > np.median(score), "approved", "rejected")
credit = pd.DataFrame({"age": age, "income": income, "loan_amount": loan, "employment": employment,
                       "home_ownership": home, "credit_history": history, "decision": approved})
for col, frac in [("income", 0.04), ("employment", 0.03)]:
    credit.loc[rng.random(n) < frac, col] = np.nan
credit.to_csv(OUT / "credit_approval_synthetic.csv", index=False)
print("Sample datasets written to", OUT)
