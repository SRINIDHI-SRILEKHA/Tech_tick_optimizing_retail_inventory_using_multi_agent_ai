import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("demand_forecasting.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Weekday"] = df["Date"].dt.weekday
df["Month"] = df["Date"].dt.month

# One-hot encode categorical columns
categorical_cols = ["Promotions", "Seasonality Factors", "External Factors", "Demand Trend", "Customer Segments"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded.drop(columns=["Date"], inplace=True)

# Step 2: Split features (X) and target (y)
X = df_encoded.drop(columns=["Sales Quantity"])
y = df_encoded["Sales Quantity"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
X_test = X_test.copy()
X_test["Predicted Demand"] = model.predict(X_test)

# Extract Product ID & Store ID for viewing (for matching later)
X_test["Product ID"] = df.loc[X_test.index, "Product ID"]
X_test["Store ID"] = df.loc[X_test.index, "Store ID"]

# Show prediction
print("\n🔮 Predicted Demand (sample):")
print(X_test[["Product ID", "Store ID", "Predicted Demand"]].head())

# Save for next step
X_test[["Product ID", "Store ID", "Predicted Demand"]].to_csv("predicted_demand.csv", index=False)
print("✅ Prediction saved to predicted_demand.csv")

# Step 6: Load inventory file
inventory_df = pd.read_csv("inventory_monitoring.csv")

# Step 7: Load predicted demand
predicted_df = pd.read_csv("predicted_demand.csv")

# Step 8: Merge demand + inventory on Product ID and Store ID
merged_df = pd.merge(predicted_df, inventory_df, on=["Product ID", "Store ID"])

# Step 9: Add Action column (Reorder or OK)
merged_df["Action"] = merged_df.apply(
    lambda row: "Reorder" if row["Stock Levels"] < row["Predicted Demand"] else "Stock OK",
    axis=1
)

# Step 10: Show result
print("\n📦 Inventory Optimization Result (Sample):")
print(merged_df[["Product ID", "Store ID", "Stock Levels", "Predicted Demand", "Action"]].head())

# Save for next step
merged_df.to_csv("inventory_optimized.csv", index=False)
print("✅ Inventory result saved to inventory_optimized.csv")

# Step 11: Load pricing data
pricing_df = pd.read_csv("pricing_optimization.csv")

# Step 12: Prepare features
X_price = pricing_df.drop(columns=["Sales Volume"])
y_price = pricing_df["Sales Volume"]

# Step 13: Train model
model_price = LinearRegression()
model_price.fit(X_price, y_price)

# Step 14: Predict sales based on price
pricing_df["Predicted Sales"] = model_price.predict(X_price)

# Step 15: Suggest Top Prices
top_prices = pricing_df[["Product ID", "Store ID", "Price", "Predicted Sales"]].sort_values(
    by="Predicted Sales", ascending=False
)

print("\n💰 Top Optimal Prices Suggested:")
print(top_prices.head())

# Save final price suggestion file
top_prices.to_csv("optimal_prices.csv", index=False)
print("✅ Optimal prices saved to optimal_prices.csv")

# Step 16: Load both outputs
inventory_final = pd.read_csv("inventory_optimized.csv")
price_final = pd.read_csv("optimal_prices.csv")

# Step 17: Merge on Product ID + Store ID
final_df = pd.merge(inventory_final, price_final[["Product ID", "Store ID", "Price"]], 
                    on=["Product ID", "Store ID"], how="left")

# Step 18: Rename column
final_df.rename(columns={"Price": "Optimal Price"}, inplace=True)

# Step 19: Save final summary
final_df.to_csv("final_retail_summary.csv", index=False)
print("\n📁 ✅ Final Summary Saved as 'final_retail_summary.csv'")

