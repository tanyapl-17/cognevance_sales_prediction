
import numpy as np
import pandas as pd

# Reproducible synthetic dataset
rng = np.random.default_rng(42)
n = 1000

dates = pd.date_range("2024-01-01", periods=n, freq="D")

products = rng.choice(
    ["Laptop", "Phone", "Headphones", "Tablet", "Smartwatch"],
    size=n,
    p=[0.15, 0.30, 0.25, 0.15, 0.15]
)

regions = rng.choice(
    ["North", "South", "East", "West", "Central"],
    size=n
)

category_price = {
    "Laptop": 55000,
    "Phone": 22000,
    "Headphones": 2500,
    "Tablet": 18000,
    "Smartwatch": 7000
}

base_cost_ratio = {
    "Laptop": 0.78,
    "Phone": 0.72,
    "Headphones": 0.55,
    "Tablet": 0.70,
    "Smartwatch": 0.60
}

prices = np.array([
    category_price[p] for p in products
]) * rng.uniform(0.85, 1.15, n)

units = rng.integers(1, 16, n)

# Seasonal and weekend effects
day_of_year = dates.dayofyear.to_numpy()
seasonality = 1 + 0.20 * np.sin(
    2 * np.pi * day_of_year / 365
)

weekend_effect = np.where(
    dates.dayofweek.to_numpy() >= 5, 1.15, 1.0
)

discount = rng.choice(
    [0, 5, 10, 15, 20],
    size=n,
    p=[0.25, 0.20, 0.30, 0.15, 0.10]
)

# Revenue with demand variation and noise
demand_factor = (
    seasonality
    * weekend_effect
    * (1 + discount / 100 * 0.35)
)

units = np.maximum(
    1,
    np.round(units * demand_factor).astype(int)
)

gross_revenue = prices * units
revenue = gross_revenue * (1 - discount / 100)

cost_ratio = np.array([
    base_cost_ratio[p] for p in products
])

cost = gross_revenue * cost_ratio
profit = revenue - cost

advertising_spend = rng.uniform(100, 5000, n)

data = pd.DataFrame({
    "Date": dates,
    "Product": products,
    "Region": regions,
    "Units_Sold": units,
    "Unit_Price": prices.round(2),
    "Discount_Percent": discount,
    "Advertising_Spend": advertising_spend.round(2),
    "Revenue": revenue.round(2),
    "Cost": cost.round(2),
    "Profit": profit.round(2)
})

data.to_csv("sales_data.csv", index=False)

print("Sales dataset generated successfully!")
print(f"Rows: {len(data)}")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
print(f"Total revenue: ₹{data['Revenue'].sum():,.2f}")
print("Saved as sales_data.csv")
print(data.head())
