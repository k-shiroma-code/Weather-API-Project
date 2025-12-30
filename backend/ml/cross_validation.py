"""
================================================================================
🔌 CALIFORNIA GRID LOAD FORECASTING - TIME SERIES CROSS-VALIDATION
================================================================================
Implements proper time series CV (expanding/sliding window) to get robust
performance estimates. Standard k-fold CV doesn't work for time series
because it leaks future information into training.

Methods:
1. Expanding Window CV - Train on all past data, test on next period
2. Sliding Window CV - Fixed training window size

Author: Portfolio Project for SCE Grid Load Forecasting
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(filepath='data/processed/california_data_processed.csv'):
    """Load data and create features"""
    print("\n" + "=" * 80)
    print("🔌 CALIFORNIA GRID LOAD FORECASTING - CROSS-VALIDATION")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    print(f"\n✅ Data loaded: {len(df):,} observations")
    
    # Aggregate to daily
    df['Date'] = df['Time'].dt.date
    daily = df.groupby('Date').agg({
        'Electric_demand': 'mean', 'Temperature': 'mean', 'Humidity': 'mean',
        'Wind_speed': 'mean', 'GHI': 'mean', 'PV_production': 'mean', 
        'Wind_production': 'mean', 'Season': 'first', 'Day_of_the_week': 'first'
    }).reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.set_index('Date')
    
    # Feature engineering
    daily['Temp_squared'] = daily['Temperature'] ** 2
    daily['Is_weekend'] = (daily['Day_of_the_week'] >= 5).astype(int)
    daily['Total_renewable'] = daily['PV_production'] + daily['Wind_production']
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 21]:
        daily[f'demand_lag_{lag}'] = daily['Electric_demand'].shift(lag)
    
    # Rolling features
    daily['demand_rolling_7d_mean'] = daily['Electric_demand'].rolling(7).mean()
    daily['demand_rolling_7d_std'] = daily['Electric_demand'].rolling(7).std()
    daily['demand_rolling_14d_mean'] = daily['Electric_demand'].rolling(14).mean()
    daily['temp_rolling_3d_mean'] = daily['Temperature'].rolling(3).mean()
    daily['temp_rolling_7d_mean'] = daily['Temperature'].rolling(7).mean()
    
    # Calendar features
    daily['day_of_year'] = daily.index.dayofyear
    daily['month'] = daily.index.month
    daily['week_of_year'] = daily.index.isocalendar().week.astype(int)
    daily['day_sin'] = np.sin(2 * np.pi * daily['day_of_year'] / 365)
    daily['day_cos'] = np.cos(2 * np.pi * daily['day_of_year'] / 365)
    daily['week_sin'] = np.sin(2 * np.pi * daily['week_of_year'] / 52)
    daily['week_cos'] = np.cos(2 * np.pi * daily['week_of_year'] / 52)
    
    # Drop NaN from lag features
    daily = daily.dropna()
    print(f"📊 Daily observations after feature engineering: {len(daily)}")
    print(f"📅 Date range: {daily.index.min().date()} to {daily.index.max().date()}")
    
    return daily

# ============================================================================
# TIME SERIES CROSS-VALIDATION
# ============================================================================

def time_series_cv_splits(data, n_splits=5, test_size=30, gap=0, method='expanding'):
    """
    Generate time series cross-validation splits.
    
    Parameters:
    -----------
    data : DataFrame with DatetimeIndex
    n_splits : number of CV folds
    test_size : days in each test set
    gap : days between train and test (to simulate forecast horizon)
    method : 'expanding' (growing train set) or 'sliding' (fixed window)
    
    Returns:
    --------
    List of (train_idx, test_idx) tuples
    """
    n = len(data)
    splits = []
    
    # Calculate minimum training size
    if method == 'expanding':
        min_train_size = n - (n_splits * test_size) - ((n_splits - 1) * gap)
    else:  # sliding
        min_train_size = 365  # At least 1 year for sliding window
    
    for i in range(n_splits):
        # Test set position (from end, working backwards)
        test_end = n - (i * (test_size + gap))
        test_start = test_end - test_size
        
        # Train set position
        if method == 'expanding':
            train_start = 0
        else:  # sliding
            train_start = max(0, test_start - gap - min_train_size)
        
        train_end = test_start - gap
        
        if train_end > train_start and test_start >= 0:
            train_idx = list(range(train_start, train_end))
            test_idx = list(range(test_start, test_end))
            splits.append((train_idx, test_idx))
    
    # Reverse to get chronological order
    splits = splits[::-1]
    
    return splits

def evaluate_fold(y_true, y_pred):
    """Calculate metrics for a single fold"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

def run_cv_for_model(data, feature_cols, model_class, model_params, splits, model_name):
    """Run cross-validation for a single model"""
    results = []
    all_predictions = []
    all_actuals = []
    all_dates = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        X_train = train[feature_cols].values
        y_train = train['Electric_demand'].values
        X_test = test[feature_cols].values
        y_test = test['Electric_demand'].values
        
        # Scale features for Ridge
        if model_class == Ridge:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train and predict
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Store results
        metrics = evaluate_fold(y_test, predictions)
        metrics['fold'] = fold_idx + 1
        metrics['train_size'] = len(train_idx)
        metrics['test_start'] = test.index.min().date()
        metrics['test_end'] = test.index.max().date()
        results.append(metrics)
        
        # Store predictions for plotting
        all_predictions.extend(predictions)
        all_actuals.extend(y_test)
        all_dates.extend(test.index.tolist())
    
    return results, all_predictions, all_actuals, all_dates

# ============================================================================
# MAIN CROSS-VALIDATION PIPELINE
# ============================================================================

def run_cross_validation(data, n_splits=5, test_size=30, method='expanding'):
    """Run full cross-validation pipeline for all models"""
    
    print(f"\n" + "=" * 80)
    print(f"📊 TIME SERIES CROSS-VALIDATION")
    print(f"=" * 80)
    print(f"   Method: {method.upper()} window")
    print(f"   Folds: {n_splits}")
    print(f"   Test size: {test_size} days per fold")
    
    # Generate splits
    splits = time_series_cv_splits(data, n_splits=n_splits, test_size=test_size, 
                                    gap=0, method=method)
    
    print(f"\n📋 Cross-Validation Splits:")
    print("-" * 70)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_start = data.index[train_idx[0]].date()
        train_end = data.index[train_idx[-1]].date()
        test_start = data.index[test_idx[0]].date()
        test_end = data.index[test_idx[-1]].date()
        print(f"   Fold {i+1}: Train [{train_start} → {train_end}] ({len(train_idx)} days) | "
              f"Test [{test_start} → {test_end}] ({len(test_idx)} days)")
    
    # Define feature sets
    weather_features = ['Temperature', 'Humidity', 'Wind_speed', 'Temp_squared', 
                       'Is_weekend', 'day_sin', 'day_cos']
    
    all_features = [f for f in [
        'Temperature', 'Humidity', 'Wind_speed', 'GHI', 'PV_production', 'Wind_production',
        'Temp_squared', 'Is_weekend', 'Total_renewable',
        'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_7', 'demand_lag_14',
        'demand_rolling_7d_mean', 'demand_rolling_7d_std', 'demand_rolling_14d_mean',
        'temp_rolling_3d_mean', 'temp_rolling_7d_mean', 
        'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month'
    ] if f in data.columns]
    
    # Define models
    models = {
        'Ridge + Weather': {
            'class': Ridge,
            'params': {'alpha': 1.0},
            'features': weather_features
        },
        'Random Forest': {
            'class': RandomForestRegressor,
            'params': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
                      'random_state': 42, 'n_jobs': -1},
            'features': all_features
        },
        'Gradient Boosting': {
            'class': GradientBoostingRegressor,
            'params': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                      'subsample': 0.8, 'random_state': 42},
            'features': all_features
        }
    }
    
    # Run CV for each model
    all_results = {}
    all_cv_predictions = {}
    
    print(f"\n" + "=" * 80)
    print(f"🏃 RUNNING CROSS-VALIDATION")
    print("=" * 80)
    
    for model_name, model_config in models.items():
        print(f"\n📈 {model_name}...")
        
        results, predictions, actuals, dates = run_cv_for_model(
            data=data,
            feature_cols=model_config['features'],
            model_class=model_config['class'],
            model_params=model_config['params'],
            splits=splits,
            model_name=model_name
        )
        
        all_results[model_name] = results
        all_cv_predictions[model_name] = {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates
        }
        
        # Print fold results
        results_df = pd.DataFrame(results)
        print(f"\n   Fold Results:")
        print(f"   {'Fold':<6} {'Test Period':<25} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'MAPE (%)':<10}")
        print(f"   {'-'*65}")
        for _, row in results_df.iterrows():
            period = f"{row['test_start']} → {row['test_end']}"
            print(f"   {int(row['fold']):<6} {period:<25} {row['mae']:>8,.1f}    {row['rmse']:>8,.1f}    {row['mape']:>6.2f}")
        
        # Summary stats
        print(f"\n   📊 Summary:")
        print(f"      Mean MAE:  {results_df['mae'].mean():,.2f} ± {results_df['mae'].std():,.2f} MW")
        print(f"      Mean RMSE: {results_df['rmse'].mean():,.2f} ± {results_df['rmse'].std():,.2f} MW")
        print(f"      Mean MAPE: {results_df['mape'].mean():.2f} ± {results_df['mape'].std():.2f}%")
    
    return all_results, all_cv_predictions, splits

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cv_results(all_results, save_path='figures/11_cv_results.png'):
    """Plot cross-validation results comparison"""
    print(f"\n📉 Creating CV results plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Ridge + Weather': '#3498db', 'Random Forest': '#27ae60', 'Gradient Boosting': '#e74c3c'}
    
    # Plot 1: MAE across folds
    ax = axes[0, 0]
    for model_name, results in all_results.items():
        df = pd.DataFrame(results)
        ax.plot(df['fold'], df['mae'], marker='o', linewidth=2, markersize=8,
               label=model_name, color=colors.get(model_name, 'gray'))
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('MAE (MW)', fontsize=11)
    ax.set_title('MAE Across CV Folds', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(list(all_results.values())[0]) + 1))
    
    # Plot 2: RMSE across folds
    ax = axes[0, 1]
    for model_name, results in all_results.items():
        df = pd.DataFrame(results)
        ax.plot(df['fold'], df['rmse'], marker='s', linewidth=2, markersize=8,
               label=model_name, color=colors.get(model_name, 'gray'))
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('RMSE (MW)', fontsize=11)
    ax.set_title('RMSE Across CV Folds', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(list(all_results.values())[0]) + 1))
    
    # Plot 3: Mean metrics comparison (bar chart)
    ax = axes[1, 0]
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35
    
    mean_mae = [pd.DataFrame(all_results[m])['mae'].mean() for m in model_names]
    std_mae = [pd.DataFrame(all_results[m])['mae'].std() for m in model_names]
    mean_rmse = [pd.DataFrame(all_results[m])['rmse'].mean() for m in model_names]
    std_rmse = [pd.DataFrame(all_results[m])['rmse'].std() for m in model_names]
    
    bars1 = ax.bar(x - width/2, mean_mae, width, yerr=std_mae, label='MAE', 
                   color='#3498db', capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, mean_rmse, width, yerr=std_rmse, label='RMSE', 
                   color='#e74c3c', capsize=5, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Error (MW)', fontsize=11)
    ax.set_title('Mean CV Error (± Std Dev)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MAPE comparison with confidence intervals
    ax = axes[1, 1]
    mean_mape = [pd.DataFrame(all_results[m])['mape'].mean() for m in model_names]
    std_mape = [pd.DataFrame(all_results[m])['mape'].std() for m in model_names]
    
    bars = ax.bar(model_names, mean_mape, yerr=std_mape, 
                  color=[colors.get(m, 'gray') for m in model_names],
                  capsize=5, alpha=0.8, edgecolor='black')
    
    for bar, val, std in zip(bars, mean_mape, std_mape):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
               f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_title('Mean CV MAPE (± Std Dev)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Time Series Cross-Validation Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

def plot_cv_predictions(all_cv_predictions, save_path='figures/12_cv_predictions.png'):
    """Plot actual vs predicted for all CV folds"""
    print(f"\n📈 Creating CV predictions plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Ridge + Weather': '#3498db', 'Random Forest': '#27ae60', 'Gradient Boosting': '#e74c3c'}
    
    for ax, (model_name, data) in zip(axes, all_cv_predictions.items()):
        actuals = np.array(data['actuals'])
        predictions = np.array(data['predictions'])
        
        ax.scatter(actuals, predictions, alpha=0.5, color=colors.get(model_name, 'gray'), s=30)
        
        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel('Actual Demand (MW)', fontsize=11)
        ax.set_ylabel('Predicted Demand (MW)', fontsize=11)
        ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Validation: Actual vs Predicted (All Folds)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

def plot_cv_timeline(data, splits, all_cv_predictions, save_path='figures/13_cv_timeline.png'):
    """Plot predictions over time for best model"""
    print(f"\n📅 Creating CV timeline plot...")
    
    # Use Random Forest predictions
    rf_data = all_cv_predictions['Random Forest']
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot full actual data
    ax.plot(data.index, data['Electric_demand'], color='black', linewidth=1, 
            alpha=0.5, label='Actual (Full)')
    
    # Plot predictions for each fold with different colors
    dates = pd.to_datetime(rf_data['dates'])
    predictions = np.array(rf_data['predictions'])
    actuals = np.array(rf_data['actuals'])
    
    # Color by fold
    fold_colors = plt.cm.Set2(np.linspace(0, 1, len(splits)))
    
    start_idx = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_size = len(test_idx)
        fold_dates = dates[start_idx:start_idx + fold_size]
        fold_preds = predictions[start_idx:start_idx + fold_size]
        fold_actuals = actuals[start_idx:start_idx + fold_size]
        
        ax.plot(fold_dates, fold_actuals, color='black', linewidth=2)
        ax.plot(fold_dates, fold_preds, color=fold_colors[fold_idx], linewidth=2,
               linestyle='--', label=f'Fold {fold_idx + 1} Pred')
        
        start_idx += fold_size
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Electric Demand (MW)', fontsize=11)
    ax.set_title('Random Forest: CV Predictions Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

def print_summary(all_results):
    """Print final summary report"""
    print("\n" + "=" * 80)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("=" * 80)
    
    print("\n📊 FINAL CV RESULTS SUMMARY:")
    print("-" * 80)
    print(f"{'Model':<25} {'MAE (MW)':<20} {'RMSE (MW)':<20} {'MAPE (%)':<15}")
    print("-" * 80)
    
    summary_data = []
    for model_name, results in all_results.items():
        df = pd.DataFrame(results)
        mae_mean, mae_std = df['mae'].mean(), df['mae'].std()
        rmse_mean, rmse_std = df['rmse'].mean(), df['rmse'].std()
        mape_mean, mape_std = df['mape'].mean(), df['mape'].std()
        
        print(f"{model_name:<25} {mae_mean:>6,.1f} ± {mae_std:<8,.1f}  {rmse_mean:>6,.1f} ± {rmse_std:<8,.1f}  {mape_mean:>5.2f} ± {mape_std:<5.2f}")
        summary_data.append({
            'model': model_name,
            'mae_mean': mae_mean,
            'mae_std': mae_std
        })
    
    print("-" * 80)
    
    # Best model
    best = min(summary_data, key=lambda x: x['mae_mean'])
    print(f"\n🏆 Best Model: {best['model']}")
    print(f"   CV MAE: {best['mae_mean']:,.1f} ± {best['mae_std']:,.1f} MW")
    
    print(f"\n💡 Key Takeaways:")
    print(f"   ✓ Cross-validation provides more robust performance estimates")
    print(f"   ✓ Standard deviation shows model stability across time periods")
    print(f"   ✓ Results are more reliable than single train/test split")
    
    print(f"\n📁 Visualizations saved:")
    print(f"   - figures/11_cv_results.png (metrics across folds)")
    print(f"   - figures/12_cv_predictions.png (actual vs predicted)")
    print(f"   - figures/13_cv_timeline.png (predictions over time)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Run cross-validation
    all_results, all_cv_predictions, splits = run_cross_validation(
        data=data,
        n_splits=5,
        test_size=30,
        method='expanding'
    )
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_cv_results(all_results)
    plot_cv_predictions(all_cv_predictions)
    plot_cv_timeline(data, splits, all_cv_predictions)
    
    # Print summary
    print_summary(all_results)
    
    return all_results, all_cv_predictions

if __name__ == "__main__":
    results, predictions = main()