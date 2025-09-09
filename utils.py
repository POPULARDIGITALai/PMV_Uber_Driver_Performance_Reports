import numpy as np
import pandas as pd

def format_currency(amount):
    """Format amount as Indian currency"""
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.1f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.1f}L"
    elif amount >= 1000:
        return f"₹{amount/1000:.1f}K"
    else:
        return f"₹{amount:,.0f}"

def calculate_correlation(x, y):
    """Calculate correlation coefficient between two series"""
    try:
        # Remove any NaN values
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(valid_data) < 2:
            return 0.0
        
        correlation = np.corrcoef(valid_data['x'], valid_data['y'])[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0

def get_scheme_colors():
    """Return color mapping for different schemes"""
    return {
        'Revenue Share': '#1f77b4',    # Blue
        'Rental (All)': '#2ca02c',     # Green
        'Rental 2DP': '#d62728',       # Red
        'Combined': '#ff7f0e'          # Orange
    }

def get_month_colors():
    """Return color mapping for different months"""
    return {
        'June': '#8c564b',     # Brown
        'July': '#e377c2',     # Pink
        'August': '#7f7f7f'    # Gray
    }

def calculate_performance_metrics(data):
    """Calculate comprehensive performance metrics for a dataset"""
    if len(data) == 0:
        return {}
    
    earnings = data['Net Earnings (Toll - Tip)']
    tenure = data['Tenure(Days)']
    
    metrics = {
        'total_records': len(data),
        'avg_earnings': earnings.mean(),
        'median_earnings': earnings.median(),
        'std_earnings': earnings.std(),
        'min_earnings': earnings.min(),
        'max_earnings': earnings.max(),
        'q25_earnings': earnings.quantile(0.25),
        'q75_earnings': earnings.quantile(0.75),
        'avg_tenure': tenure.mean(),
        'median_tenure': tenure.median(),
        'std_tenure': tenure.std(),
        'min_tenure': tenure.min(),
        'max_tenure': tenure.max(),
        'correlation': calculate_correlation(tenure, earnings)
    }
    
    # Calculate additional metrics if available
    if 'Online Hours' in data.columns:
        online_hours = data['Online Hours']
        metrics.update({
            'avg_online_hours': online_hours.mean(),
            'total_online_hours': online_hours.sum(),
            'earnings_per_hour': earnings.sum() / online_hours.sum() if online_hours.sum() > 0 else 0
        })
    
    if 'Trips' in data.columns:
        trips = data['Trips']
        metrics.update({
            'avg_trips': trips.mean(),
            'total_trips': trips.sum(),
            'earnings_per_trip': earnings.sum() / trips.sum() if trips.sum() > 0 else 0
        })
    
    return metrics

def identify_outliers(data, column, method='iqr', factor=1.5):
    """Identify outliers in a dataset column"""
    if column not in data.columns or len(data) == 0:
        return []
    
    series = data[column].dropna()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = data[
            (data[column] < lower_bound) | (data[column] > upper_bound)
        ].index.tolist()
        
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = data[z_scores > factor].index.tolist()
    
    else:
        outliers = []
    
    return outliers

def categorize_performance(earnings, tenure_brackets=None, earnings_brackets=None):
    """Categorize driver performance based on earnings and tenure"""
    if tenure_brackets is None:
        tenure_brackets = [0, 30, 90, 180, 365, float('inf')]
    
    if earnings_brackets is None:
        earnings_brackets = [0, 10000, 25000, 50000, 100000, float('inf')]
    
    # Tenure categories
    tenure_labels = ['New (0-30)', 'Junior (31-90)', 'Mid (91-180)', 'Senior (181-365)', 'Veteran (365+)']
    
    # Earnings categories
    earnings_labels = ['Low (<10K)', 'Below Avg (10K-25K)', 'Average (25K-50K)', 'Good (50K-100K)', 'Excellent (100K+)']
    
    return {
        'tenure_category': pd.cut(tenure_brackets, bins=tenure_brackets, labels=tenure_labels, right=False),
        'earnings_category': pd.cut(earnings_brackets, bins=earnings_brackets, labels=earnings_labels, right=False)
    }

def generate_insights(data, scheme_name):
    """Generate automated insights from the data"""
    insights = []
    
    if len(data) == 0:
        return ["No data available for analysis."]
    
    metrics = calculate_performance_metrics(data)
    
    # Correlation insights
    correlation = metrics['correlation']
    if correlation > 0.5:
        insights.append(f"Strong positive correlation ({correlation:.2f}) between tenure and earnings in {scheme_name}.")
    elif correlation > 0.2:
        insights.append(f"Moderate positive correlation ({correlation:.2f}) between tenure and earnings in {scheme_name}.")
    elif correlation < -0.2:
        insights.append(f"Negative correlation ({correlation:.2f}) between tenure and earnings in {scheme_name}.")
    else:
        insights.append(f"Weak correlation ({correlation:.2f}) between tenure and earnings in {scheme_name}.")
    
    # Earnings distribution insights
    avg_earnings = metrics['avg_earnings']
    median_earnings = metrics['median_earnings']
    
    if avg_earnings > median_earnings * 1.2:
        insights.append(f"Earnings distribution is right-skewed - few high earners pull the average up.")
    elif median_earnings > avg_earnings * 1.2:
        insights.append(f"Earnings distribution is left-skewed - most drivers earn above average.")
    
    # Tenure insights
    avg_tenure = metrics['avg_tenure']
    if avg_tenure < 30:
        insights.append(f"High driver turnover - average tenure is only {avg_tenure:.0f} days.")
    elif avg_tenure > 180:
        insights.append(f"Good driver retention - average tenure is {avg_tenure:.0f} days.")
    
    # Performance insights
    if 'earnings_per_hour' in metrics and metrics['earnings_per_hour'] > 0:
        eph = metrics['earnings_per_hour']
        insights.append(f"Average earnings per hour: ₹{eph:.0f}")
    
    if 'earnings_per_trip' in metrics and metrics['earnings_per_trip'] > 0:
        ept = metrics['earnings_per_trip']
        insights.append(f"Average earnings per trip: ₹{ept:.0f}")
    
    return insights

def export_data_summary(filtered_data, filename="driver_analysis_summary.csv"):
    """Export summary statistics for all schemes"""
    summary_data = []
    
    for scheme, data in filtered_data.items():
        metrics = calculate_performance_metrics(data)
        metrics['scheme'] = scheme
        summary_data.append(metrics)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def validate_data_quality(data, required_columns=None):
    """Validate data quality and return quality score"""
    if required_columns is None:
        required_columns = ['Tenure(Days)', 'Net Earnings (Toll - Tip)']
    
    quality_issues = []
    
    # Check for missing required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        quality_issues.append(f"Missing columns: {missing_cols}")
    
    # Check for missing values
    for col in required_columns:
        if col in data.columns:
            missing_pct = data[col].isnull().mean() * 100
            if missing_pct > 10:
                quality_issues.append(f"{col}: {missing_pct:.1f}% missing values")
    
    # Check for data types
    numeric_cols = ['Tenure(Days)', 'Net Earnings (Toll - Tip)', 'Online Hours', 'Trips']
    for col in numeric_cols:
        if col in data.columns:
            try:
                pd.to_numeric(data[col], errors='raise')
            except:
                quality_issues.append(f"{col}: Contains non-numeric values")
    
    # Check for outliers
    for col in ['Tenure(Days)', 'Net Earnings (Toll - Tip)']:
        if col in data.columns:
            outliers = identify_outliers(data, col)
            outlier_pct = len(outliers) / len(data) * 100
            if outlier_pct > 5:
                quality_issues.append(f"{col}: {outlier_pct:.1f}% potential outliers")
    
    # Calculate quality score (0-100)
    max_issues = 10
    quality_score = max(0, 100 - (len(quality_issues) / max_issues * 100))
    
    return {
        'quality_score': quality_score,
        'issues': quality_issues,
        'is_valid': len(quality_issues) == 0
    }

def get_filter_options(data):
    """Get available filter options from data"""
    options = {}
    
    if 'Month' in data.columns:
        options['months'] = sorted(data['Month'].dropna().unique().tolist())
    
    if 'Scheme' in data.columns:
        options['schemes'] = sorted(data['Scheme'].dropna().unique().tolist())
    
    # Numeric ranges
    if 'Tenure(Days)' in data.columns:
        options['tenure_range'] = [
            int(data['Tenure(Days)'].min()), 
            int(data['Tenure(Days)'].max())
        ]
    
    if 'Net Earnings (Toll - Tip)' in data.columns:
        options['earnings_range'] = [
            int(data['Net Earnings (Toll - Tip)'].min()), 
            int(data['Net Earnings (Toll - Tip)'].max())
        ]
    
    return options