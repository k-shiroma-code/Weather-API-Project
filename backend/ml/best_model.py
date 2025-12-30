"""
================================================================================
CALIFORNIA GRID LOAD FORECASTING - IMPROVED MODELS
================================================================================
Compares: Persistence Baseline, Ridge+Weather, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_data():
    print("\n" + "=" * 80)
    print("🔌 CALIFORNIA GRID LOAD FORECASTING - IMPROVED MODELS")
    print("=" * 80)
    df = pd.read_csv('data/processed/california_data_processed.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    print(f"\n✅ Data loaded: {len(df):,} observations")
    print(f"📅 Date range: {df['Time'].min()} to {df['Time'].max()}")
    return df

def prepare_daily_data(df):
    print(f"\n📊 Aggregating to daily level...")
    df['Date'] = df['Time'].dt.date
    daily_agg = df.groupby('Date').agg({
        'Electric_demand': 'mean', 'Temperature': 'mean', 'Humidity': 'mean',
        'Wind_speed': 'mean', 'GHI': 'mean', 'DNI': 'mean', 'DHI': 'mean',
        'PV_production': 'mean', 'Wind_production': 'mean',
        'Season': 'first', 'Day_of_the_week': 'first'
    }).reset_index()
    daily_agg['Date'] = pd.to_datetime(daily_agg['Date'])
    daily_agg = daily_agg.set_index('Date')
    daily_agg['Temp_squared'] = daily_agg['Temperature'] ** 2
    daily_agg['Is_weekend'] = (daily_agg['Day_of_the_week'] >= 5).astype(int)
    daily_agg['Total_renewable'] = daily_agg['PV_production'] + daily_agg['Wind_production']
    print(f"   Daily observations: {len(daily_agg)}")
    return daily_agg

def create_features(df):
    features = df.copy()
    for lag in [1, 2, 3, 7, 14, 21]:
        features[f'demand_lag_{lag}'] = features['Electric_demand'].shift(lag)
    features['demand_rolling_7d_mean'] = features['Electric_demand'].rolling(7).mean()
    features['demand_rolling_7d_std'] = features['Electric_demand'].rolling(7).std()
    features['demand_rolling_14d_mean'] = features['Electric_demand'].rolling(14).mean()
    features['temp_rolling_3d_mean'] = features['Temperature'].rolling(3).mean()
    features['temp_rolling_7d_mean'] = features['Temperature'].rolling(7).mean()
    features['day_of_year'] = features.index.dayofyear
    features['month'] = features.index.month
    features['week_of_year'] = features.index.isocalendar().week.astype(int)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
    features['week_sin'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
    features['week_cos'] = np.cos(2 * np.pi * features['week_of_year'] / 52)
    return features

def split_data(df, test_days=90):
    print(f"\n📊 Train/Test Split ({test_days} days for testing):")
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    print(f"   Train: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
    print(f"   Test:  {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")
    return train, test

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

def build_persistence_baseline(train, test):
    print(f"\n" + "-" * 60)
    print(f"📈 MODEL 0: Persistence Baseline")
    print("-" * 60)
    predictions = np.full(len(test), train['Electric_demand'].iloc[-1])
    metrics = evaluate(test['Electric_demand'].values, predictions)
    print(f"   MAE: {metrics['mae']:,.2f} MW | RMSE: {metrics['rmse']:,.2f} MW | MAPE: {metrics['mape']:.2f}%")
    return {'name': 'Persistence Baseline', 'predictions': predictions, **metrics}

def build_ridge_model(train, test, feature_cols):
    print(f"\n" + "-" * 60)
    print(f"📈 MODEL 1: Ridge + Weather ({len(feature_cols)} features)")
    print("-" * 60)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    model = Ridge(alpha=1.0)
    model.fit(X_train, train['Electric_demand'])
    predictions = model.predict(X_test)
    metrics = evaluate(test['Electric_demand'].values, predictions)
    print(f"   MAE: {metrics['mae']:,.2f} MW | RMSE: {metrics['rmse']:,.2f} MW | MAPE: {metrics['mape']:.2f}%")
    importance = pd.DataFrame({'feature': feature_cols, 'importance': np.abs(model.coef_)}).sort_values('importance', ascending=False)
    return {'name': 'Ridge + Weather', 'predictions': predictions, 'importance': importance, **metrics}

def build_rf_model(train, test, feature_cols):
    print(f"\n" + "-" * 60)
    print(f"📈 MODEL 2: Random Forest ({len(feature_cols)} features)")
    print("-" * 60)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(train[feature_cols], train['Electric_demand'])
    predictions = model.predict(test[feature_cols])
    metrics = evaluate(test['Electric_demand'].values, predictions)
    print(f"   MAE: {metrics['mae']:,.2f} MW | RMSE: {metrics['rmse']:,.2f} MW | MAPE: {metrics['mape']:.2f}%")
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print(f"\n   🔝 Top 5: {', '.join(importance.head(5)['feature'].tolist())}")
    return {'name': 'Random Forest', 'predictions': predictions, 'importance': importance, 'model': model, **metrics}

def build_gbm_model(train, test, feature_cols):
    print(f"\n" + "-" * 60)
    print(f"📈 MODEL 3: Gradient Boosting ({len(feature_cols)} features)")
    print("-" * 60)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
    model.fit(train[feature_cols], train['Electric_demand'])
    predictions = model.predict(test[feature_cols])
    metrics = evaluate(test['Electric_demand'].values, predictions)
    print(f"   MAE: {metrics['mae']:,.2f} MW | RMSE: {metrics['rmse']:,.2f} MW | MAPE: {metrics['mape']:.2f}%")
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print(f"\n   🔝 Top 5: {', '.join(importance.head(5)['feature'].tolist())}")
    return {'name': 'Gradient Boosting', 'predictions': predictions, 'importance': importance, 'model': model, **metrics}

def plot_comparison(test, results, save_path='figures/08_model_comparison.png'):
    print(f"\n📉 Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'Persistence Baseline': '#95a5a6', 'Ridge + Weather': '#3498db', 'Random Forest': '#27ae60', 'Gradient Boosting': '#e74c3c'}
    
    # Forecasts
    ax = axes[0, 0]
    test_plot = test.iloc[-30:]
    ax.plot(test_plot.index, test_plot['Electric_demand'], label='Actual', color='black', linewidth=2.5)
    for name, res in results.items():
        ax.plot(test_plot.index, res['predictions'][-30:], label=name, color=colors.get(name, 'gray'), linestyle='--', linewidth=1.5)
    ax.set_title('Forecast Comparison (Last 30 Days)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    
    # MAE/RMSE bars
    ax = axes[0, 1]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, [results[m]['mae'] for m in models], width, label='MAE', color='#3498db')
    ax.bar(x + width/2, [results[m]['rmse'] for m in models], width, label='RMSE', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_title('Error Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAPE
    ax = axes[1, 0]
    mape_vals = [results[m]['mape'] for m in models]
    bars = ax.bar(models, mape_vals, color=[colors.get(m, '#95a5a6') for m in models])
    for bar, val in zip(bars, mape_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}%', ha='center', fontweight='bold')
    ax.set_title('MAPE Comparison', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Improvement
    ax = axes[1, 1]
    baseline_mae = results['Persistence Baseline']['mae']
    improvements = {n: ((baseline_mae - r['mae']) / baseline_mae) * 100 for n, r in results.items() if n != 'Persistence Baseline'}
    bars = ax.bar(list(improvements.keys()), list(improvements.values()), color=[colors.get(m, '#95a5a6') for m in improvements.keys()])
    for bar, val in zip(bars, improvements.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_title('MAE Improvement Over Baseline', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

def plot_importance(results, save_path='figures/09_feature_importance.png'):
    print(f"\n📊 Creating feature importance plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, name in zip(axes, ['Random Forest', 'Gradient Boosting']):
        if name in results and 'importance' in results[name]:
            data = results[name]['importance'].head(15).sort_values('importance')
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
            ax.barh(data['feature'], data['importance'], color=colors)
            ax.set_title(f'{name} - Top 15 Features', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

def plot_error_analysis(test, result, save_path='figures/10_error_analysis_improved.png'):
    print(f"\n📈 Creating error analysis...")
    predictions = result['predictions']
    actual = test['Electric_demand'].values
    errors = actual - predictions
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(test.index, errors, color='#e74c3c', linewidth=1)
    axes[0, 0].axhline(y=0, color='black', linestyle='--')
    axes[0, 0].set_title('Error Over Time', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(errors, bins=30, color='#3498db', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Error Distribution', fontweight='bold')
    
    axes[0, 2].scatter(actual, predictions, alpha=0.5, color='#27ae60')
    axes[0, 2].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', linewidth=2)
    axes[0, 2].set_title('Actual vs Predicted', fontweight='bold')
    
    axes[1, 0].scatter(test['Temperature'], errors, alpha=0.5, color='#e74c3c')
    axes[1, 0].axhline(y=0, color='black', linestyle='--')
    axes[1, 0].set_title('Error vs Temperature', fontweight='bold')
    
    error_by_dow = pd.DataFrame({'dow': test['Day_of_the_week'], 'abs_error': np.abs(errors)})
    axes[1, 1].bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], error_by_dow.groupby('dow')['abs_error'].mean().values, color='#9b59b6')
    axes[1, 1].set_title('Error by Day of Week', fontweight='bold')
    
    ape = np.abs(errors / actual) * 100
    axes[1, 2].plot(test.index, ape, color='#f39c12')
    axes[1, 2].axhline(y=np.mean(ape), color='red', linestyle='--', label=f'Mean: {np.mean(ape):.2f}%')
    axes[1, 2].set_title('Percentage Error', fontweight='bold')
    axes[1, 2].legend()
    
    plt.suptitle(f'Error Analysis: {result["name"]}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()

def print_summary(results):
    print("\n" + "=" * 80)
    print("✅ IMPROVED MODELING COMPLETE!")
    print("=" * 80)
    print("\n📊 FINAL RESULTS:")
    print("-" * 70)
    print(f"{'Model':<25} {'MAE (MW)':<15} {'RMSE (MW)':<15} {'MAPE (%)':<10}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<25} {res['mae']:>10,.2f}     {res['rmse']:>10,.2f}     {res['mape']:>6.2f}")
    print("-" * 70)
    
    best = min([(k, v) for k, v in results.items() if k != 'Persistence Baseline'], key=lambda x: x[1]['mae'])
    baseline_mae = results['Persistence Baseline']['mae']
    improvement = ((baseline_mae - best[1]['mae']) / baseline_mae) * 100
    
    print(f"\n🏆 Best Model: {best[0]}")
    print(f"   MAE: {best[1]['mae']:,.2f} MW | Improvement: {improvement:.1f}%")
    print(f"\n💡 Key Insights:")
    print(f"   ✓ Weather + lag features significantly improve forecasts")
    print(f"   ✓ Tree models capture non-linear temperature-demand relationship")
    print(f"   ✓ demand_lag_1 is typically the most important feature")

def main():
    df = load_data()
    daily_df = prepare_daily_data(df)
    
    print(f"\n🔧 Engineering features...")
    daily_features = create_features(daily_df).dropna()
    print(f"   Final: {len(daily_features)} days")
    
    train, test = split_data(daily_features, test_days=90)
    
    weather_features = ['Temperature', 'Humidity', 'Wind_speed', 'Temp_squared', 'Is_weekend', 'day_sin', 'day_cos']
    all_features = [f for f in [
        'Temperature', 'Humidity', 'Wind_speed', 'GHI', 'PV_production', 'Wind_production',
        'Temp_squared', 'Is_weekend', 'Total_renewable',
        'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_7', 'demand_lag_14',
        'demand_rolling_7d_mean', 'demand_rolling_7d_std', 'demand_rolling_14d_mean',
        'temp_rolling_3d_mean', 'temp_rolling_7d_mean', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month'
    ] if f in train.columns]
    
    results = {}
    print("\n" + "=" * 80)
    print("🏃 TRAINING MODELS")
    print("=" * 80)
    
    results['Persistence Baseline'] = build_persistence_baseline(train, test)
    results['Ridge + Weather'] = build_ridge_model(train, test, weather_features)
    results['Random Forest'] = build_rf_model(train, test, all_features)
    results['Gradient Boosting'] = build_gbm_model(train, test, all_features)
    
    print("\n" + "=" * 80)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_comparison(test, results)
    plot_importance(results)
    best_name = min([(k, v) for k, v in results.items() if k != 'Persistence Baseline'], key=lambda x: x[1]['mae'])[0]
    plot_error_analysis(test, results[best_name])
    
    print_summary(results)
    return results

if __name__ == "__main__":
    results = main()