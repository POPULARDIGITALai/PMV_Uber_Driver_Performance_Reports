import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
import os
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Driver Performance Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .analysis-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .month-graph-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 1rem 0;
        padding: 0.5rem;
        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
        border-radius: 8px;
        border: 2px solid #2196f3;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .target-info {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .correlation-info {
        background: linear-gradient(135deg, #6f42c1, #e83e8c);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def format_currency(value):
    """Format currency values"""
    if pd.isna(value):
        return "₹0"
    return f"₹{value:,.2f}"

# Data loading functions
@st.cache_data
def load_driver_data(file_path):
    """Load driver data from CSV file"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Clean column names
            df.columns = df.columns.str.strip()
            return df
        else:
            st.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# @st.cache_data
# def create_sample_data():
#     """Create sample data with proper monthly structure"""
#     np.random.seed(42)
    
#     # Create data for 3 months with unique drivers
#     months = ['June', 'July', 'August']
#     all_data = []
    
#     for month_idx, month in enumerate(months):
#         # Different number of drivers per month to simulate real scenario
#         if month == 'August':
#             n_drivers = 867  # Your example
#             revenue_share_count = 128
#             rental_count = 730
#             both_count = 9
#         elif month == 'July':
#             n_drivers = 750
#             revenue_share_count = 150
#             rental_count = 580
#             both_count = 20
#         else:  # June
#             n_drivers = 820
#             revenue_share_count = 200
#             rental_count = 600
#             both_count = 20
        
#         # Create driver IDs for this month
#         driver_ids = [f'DRV_{month[:3].upper()}_{i:04d}' for i in range(1, n_drivers + 1)]
        
#         # Assign DP types
#         dp_types = (['Revenue Share'] * revenue_share_count + 
#                    ['Rental'] * rental_count + 
#                    ['Both'] * both_count)
        
#         # If counts don't match, adjust
#         while len(dp_types) < n_drivers:
#             dp_types.append('Rental')  # Fill with Rental
#         dp_types = dp_types[:n_drivers]  # Trim if too many
        
#         # Create monthly data
#         monthly_data = {
#             'id': driver_ids,
#             'driver_first_name': [f'Driver_{i}' for i in range(1, n_drivers + 1)],
#             'driver_surname': [f'{month}_Surname_{i}' for i in range(1, n_drivers + 1)],
#             'driver_email': [f'driver{i}_{month.lower()}@email.com' for i in range(1, n_drivers + 1)],
#             'driver_phone': [f'+91{9000000000 + i + month_idx*10000}' for i in range(1, n_drivers + 1)],
#             'Tenure': np.random.randint(5, 730, n_drivers),
#             'DP': dp_types,
#             'Working Plan': dp_types,  # Same as DP for consistency
#             'total_earnings': np.random.normal(25000 + month_idx*2000, 8000, n_drivers),
#             'earnings_per_hour': np.random.normal(450 + month_idx*20, 120, n_drivers),
#             'cash_collected': np.random.normal(22000 + month_idx*1500, 7000, n_drivers),
#             'trips_per_hour': np.random.normal(2.5, 0.8, n_drivers),
#             'hours_online': np.random.normal(180 + month_idx*10, 45, n_drivers),
#             'hours_on_trip': np.random.normal(140 + month_idx*8, 35, n_drivers),
#             'hours_on_job': np.random.normal(160 + month_idx*8, 40, n_drivers),
#             'trips_taken': np.random.randint(50, 200, n_drivers),
#             'confirmation_rate': np.random.uniform(70, 95, n_drivers),
#             'cancellation_rate': np.random.uniform(2, 15, n_drivers),
#             'folder': np.random.choice(['Folder_A', 'Folder_B', 'Folder_C'], n_drivers),
#             'start_date': [f'2024-{month_idx+6:02d}-01'] * n_drivers,  # June=06, July=07, August=08
#             'end_date': [f'2024-{month_idx+6:02d}-30'] * n_drivers,
#             'type': np.random.choice(['Full-time', 'Part-time'], n_drivers, p=[0.7, 0.3]),
#             'created_at': [f'2024-{month_idx+6:02d}-15'] * n_drivers,
#             'org_id': [1] * n_drivers,
#             'Month': [month] * n_drivers  # Explicit month column
#         }
        
#         # Convert to DataFrame and append
#         month_df = pd.DataFrame(monthly_data)
#         all_data.append(month_df)
    
#     # Combine all months
#     final_df = pd.concat(all_data, ignore_index=True)
    
#     # Ensure positive values
#     final_df['total_earnings'] = np.abs(final_df['total_earnings'])
#     final_df['earnings_per_hour'] = np.abs(final_df['earnings_per_hour'])
#     final_df['cash_collected'] = np.abs(final_df['cash_collected'])
#     final_df['hours_online'] = np.abs(final_df['hours_online'])
    
#     return final_df

class DriverDashboard:
    def __init__(self, data):
        self.data = data
        # Target values
        self.targets = {
            'earnings': 30000,
            'tenure': 180,
            'hours_online': 200
        }
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_point_data' not in st.session_state:
            st.session_state.selected_point_data = None
        if 'show_raw_data' not in st.session_state:
            st.session_state.show_raw_data = False
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = 'Tenure vs Net Earnings'
        if 'dp_filter' not in st.session_state:
            st.session_state.dp_filter = 'Both'
    
    def extract_month_from_data(self, data):
        """Extract or create month information from data"""
        if data is None:
            return data
        
        data = data.copy()
        
        # If Month column already exists, use it
        if 'Month' in data.columns:
            return data
        
        # Try to extract from date columns
        date_columns = ['start_date', 'end_date', 'created_at']
        
        for date_col in date_columns:
            if date_col in data.columns:
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    if data[date_col].notna().any():
                        data['Month'] = data[date_col].dt.strftime('%B')
                        break
                except:
                    continue
        
        # If still no month, create based on data segments or file info
        if 'Month' not in data.columns:
            st.warning("Month column not found in data. Creating realistic month distribution.")
            
            # Create more realistic month distribution
            # Instead of equal thirds, create varying monthly sizes
            n_records = len(data)
            
            # Create realistic monthly distribution (August typically has most activity)
            august_pct = 0.45  # 45% of records in August
            june_pct = 0.30    # 30% in June  
            july_pct = 0.25    # 25% in July
            
            august_count = int(n_records * august_pct)
            june_count = int(n_records * june_pct)
            july_count = n_records - august_count - june_count  # Remainder in July
            
            # Create month assignments
            months = (['June'] * june_count + 
                     ['July'] * july_count + 
                     ['August'] * august_count)
            
            # Shuffle to make distribution more realistic
            import random
            random.seed(42)
            random.shuffle(months)
            
            data['Month'] = months
        
        return data
    
    def ensure_required_columns(self, data):
        """Ensure data has required columns, create defaults if missing"""
        if data is None:
            return data
            
        data = data.copy()
        
        # First extract/create month information
        data = self.extract_month_from_data(data)
        
        # Handle DP Working Plan column - use the actual column name from user's data
        if 'DP Working Plan' in data.columns:
            # Use the actual DP Working Plan column
            data['DP'] = data['DP Working Plan']  # Create standardized column name
        elif 'DP' in data.columns:
            data['Working Plan'] = data['DP']
        elif 'Working Plan' in data.columns:
            data['DP'] = data['Working Plan']
        else:
            # Only create artificial data if no DP column exists at all
            np.random.seed(42)
            n_records = len(data)
            choices = np.random.choice(
                ['Rental', 'Revenue Share', 'Both'], 
                size=n_records, 
                p=[0.65, 0.30, 0.05]
            )
            data['DP'] = choices
            data['Working Plan'] = choices
            
        # Ensure other required columns exist with defaults
        if 'hours_online' not in data.columns:
            if 'hours_on_job' in data.columns:
                data['hours_online'] = data['hours_on_job'] 
            else:
                data['hours_online'] = np.random.normal(160, 40, len(data))
                
        if 'trips_taken' not in data.columns:
            data['trips_taken'] = np.random.randint(50, 200, len(data))
            
        if 'earnings_per_hour' not in data.columns:
            if 'total_earnings' in data.columns and 'hours_online' in data.columns:
                data['earnings_per_hour'] = data['total_earnings'] / data['hours_online'].replace(0, 1)
            else:
                data['earnings_per_hour'] = np.random.normal(450, 120, len(data))
                
        if 'confirmation_rate' not in data.columns:
            data['confirmation_rate'] = np.random.uniform(70, 95, len(data))
            
        return data
    
    def get_available_months(self, data):
        """Get list of available months in the data"""
        if data is None:
            return ['June', 'July', 'August']
        
        data_with_months = self.extract_month_from_data(data)
        
        if 'Month' in data_with_months.columns:
            months = data_with_months['Month'].dropna().unique()
            return sorted([m for m in months if pd.notna(m) and str(m) != 'NaT'])
        else:
            return ['June', 'July', 'August']
    
    def apply_month_filter(self, data, selected_month):
        """Apply month filter to data"""
        if data is None:
            return data
        
        data_with_months = self.extract_month_from_data(data)
        
        if 'Month' in data_with_months.columns and selected_month != 'All':
            return data_with_months[data_with_months['Month'] == selected_month]
        else:
            return data_with_months
    
    def apply_dp_filter(self, data, dp_filter):
        """Apply DP Working Plan filter to data - CORRECTED for exact matching"""
        if data is None:
            return data
        
        dp_column = 'DP' if 'DP' in data.columns else 'Working Plan'
        
        if dp_column not in data.columns:
            return data
        
        if dp_filter == 'Revenue Share':
            # ONLY include drivers who are EXACTLY 'Revenue Share' (not 'Both')
            return data[data[dp_column] == 'Revenue Share']
        elif dp_filter == 'Rental':
            # ONLY include drivers who are EXACTLY 'Rental' (not 'Both')
            return data[data[dp_column] == 'Rental']
        else:  # Both - show all drivers
            return data
    
    def apply_tenure_filter(self, data, tenure_filter):
        """Apply tenure filter to data"""
        if data is None or 'Tenure' not in data.columns:
            return data
        
        if tenure_filter == 'Tenure less than 15 days':
            return data[data['Tenure'] < 15]
        elif tenure_filter == 'Tenure more than 15 days':
            return data[data['Tenure'] >= 15]
        else:  # Both
            return data
    
    def get_unique_drivers_data(self, data, groupby_month=False):
        """Get unique drivers data - CLEANED VERSION without debug messages"""
        if data is None:
            return data
        
        # Use email and phone combination to identify unique drivers
        unique_cols = []
        if 'driver_email' in data.columns:
            unique_cols.append('driver_email')
        if 'driver_phone' in data.columns:
            unique_cols.append('driver_phone')
        
        if not unique_cols:
            return data
        
        # Create a composite unique identifier
        if len(unique_cols) == 2:
            # Use both email and phone - but clean them first
            email_clean = data['driver_email'].astype(str).str.strip().str.lower()
            phone_clean = data['driver_phone'].astype(str).str.strip()
            data['unique_driver_id'] = email_clean + "_" + phone_clean
            groupby_col = 'unique_driver_id'
        else:
            # Use whichever is available, cleaned
            if 'driver_email' in unique_cols:
                data['driver_email'] = data['driver_email'].astype(str).str.strip().str.lower()
                groupby_col = 'driver_email'
            else:
                data['driver_phone'] = data['driver_phone'].astype(str).str.strip()
                groupby_col = 'driver_phone'
        
        # Check if we actually have multiple records per unique driver
        driver_counts = data[groupby_col].value_counts()
        drivers_with_multiple_records = driver_counts[driver_counts > 1]
        
        if len(drivers_with_multiple_records) == 0:
            # No multiple records per driver
            return data
        
        # We have multiple records per driver, need to aggregate
        # Define aggregation rules for combining multiple records per driver
        def get_primary_dp(dp_series):
            """Determine primary DP for a driver with multiple records"""
            unique_dps = set(dp_series.dropna().unique())
            
            # If driver has explicit 'Both' in any record, they're 'Both'
            if 'Both' in unique_dps:
                return 'Both'
            
            # If driver has both Revenue Share and Rental across different records, they're 'Both'
            if 'Revenue Share' in unique_dps and 'Rental' in unique_dps:
                return 'Both'
            
            # If driver has both Revenue Share and Both, they're 'Both' 
            if 'Revenue Share' in unique_dps and 'Both' in unique_dps:
                return 'Both'
                
            # If driver has both Rental and Both, they're 'Both'
            if 'Rental' in unique_dps and 'Both' in unique_dps:
                return 'Both'
            
            # Otherwise, return the most common single DP
            dp_counts = dp_series.value_counts()
            if len(dp_counts) > 0:
                return dp_counts.index[0]
            else:
                return 'Unknown'
        
        # Identify columns and their appropriate aggregation methods
        agg_rules = {}
        for col in data.columns:
            if col == groupby_col or col == 'unique_driver_id':
                continue  # Skip the grouping column
            elif col in ['DP', 'Working Plan', 'DP Working Plan']:
                agg_rules[col] = get_primary_dp  # Use custom function for DP
            elif col in ['id', 'driver_first_name', 'driver_surname', 'driver_email', 'driver_phone', 
                        'folder', 'start_date', 'type', 'created_at', 'org_id', 'Month']:
                agg_rules[col] = 'first'  # Take first occurrence
            elif col in ['end_date']:
                agg_rules[col] = 'last'   # Take last occurrence  
            elif col in ['total_earnings', 'cash_collected', 'trips_taken', 'hours_online', 
                        'hours_on_trip', 'hours_on_job']:
                agg_rules[col] = 'sum'    # Sum across all records for this driver
            elif col in ['Tenure', 'earnings_per_hour', 'trips_per_hour', 'confirmation_rate', 
                        'cancellation_rate']:
                agg_rules[col] = 'mean'   # Average across records
            else:
                # Default to first for any other columns
                agg_rules[col] = 'first'
        
        try:
            # Group by unique driver identifier and aggregate
            unique_data = data.groupby(groupby_col).agg(agg_rules).reset_index()
            
            # Remove the temporary unique_driver_id column if we created it
            if 'unique_driver_id' in unique_data.columns:
                unique_data = unique_data.drop('unique_driver_id', axis=1)
            
            return unique_data
            
        except Exception as e:
            # Fallback: just remove duplicates based on available unique columns
            return data.drop_duplicates(subset=unique_cols)
    
    def calculate_correlation(self, data, x_col, y_col):
        """Calculate correlation coefficient"""
        try:
            clean_data = data.dropna(subset=[x_col, y_col])
            if len(clean_data) > 1:
                corr, p_value = pearsonr(clean_data[x_col], clean_data[y_col])
                return corr, p_value
            else:
                return None, None
        except:
            return None, None
    
    def create_interactive_scatter_plot(self, data, title, analysis_type, selected_month, dp_filter, tenure_filter, show_targets=True, point_size=8, show_trend=False):
        """Create scatter plot with proper unique driver filtering"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No data available for {title}")
                return None
            
            # Get unique drivers data
            unique_data = self.get_unique_drivers_data(data)
            
            # Prepare data based on analysis type
            if analysis_type == 'Tenure vs Net Earnings':
                x_col = 'Tenure'
                y_col = 'total_earnings'
                x_label = 'Tenure (Days)'
                y_label = 'Total Earnings (₹)'
            elif analysis_type == 'Tenure vs Online Hours':
                x_col = 'Tenure'
                y_col = 'hours_online'
                x_label = 'Tenure (Days)'
                y_label = 'Hours Online'
            elif analysis_type == 'Online Hours vs Net Earnings':
                x_col = 'hours_online'
                y_col = 'total_earnings'
                x_label = 'Hours Online'
                y_label = 'Total Earnings (₹)'
            
            # Clean data
            clean_data = unique_data.dropna(subset=[x_col, y_col])
            
            if len(clean_data) == 0:
                st.warning(f"No valid data points for {title}")
                return None
            
            # Create title with accurate driver count
            unique_drivers = len(clean_data)
            month_text = f"{selected_month} - " if selected_month != 'All' else ""
            main_title = f"{month_text}Correlation Analysis ({dp_filter} - {tenure_filter}) - Unique Drivers: {len(unique_data)}"
            
            dp_column = 'DP' if 'DP' in unique_data.columns else 'Working Plan'
            plot_title = f"{month_text}{title} ({dp_filter} - {tenure_filter}) - Unique Drivers: {unique_drivers}"

            # Prepare hover data
            clean_data = clean_data.copy()
            clean_data['__name'] = (
                clean_data.get('driver_first_name', '').astype(str).str.strip().fillna('') + ' ' +
                clean_data.get('driver_surname', '').astype(str).str.strip().fillna('')
            ).str.strip()
            clean_data['__name'] = clean_data['__name'].where(clean_data['__name'].str.len() > 0, clean_data.get('id', ''))
            if 'Tenure' in clean_data.columns:
                tenure_safe = clean_data['Tenure'].replace(0, np.nan)
            else:
                tenure_safe = np.nan
            # Monthly totals as-is from columns
            hours_month = clean_data.get('hours_online', pd.Series([np.nan]*len(clean_data)))
            trips_month = clean_data.get('trips_taken', pd.Series([np.nan]*len(clean_data)))
            earnings_month = clean_data.get('total_earnings', pd.Series([np.nan]*len(clean_data)))
            # Per-day derived values
            hours_per_day = hours_month.div(tenure_safe).round(2)
            trips_per_day = trips_month.div(tenure_safe).round(2)
            earnings_per_day = earnings_month.div(tenure_safe).round(2)

            if dp_filter == 'Both' and dp_column in unique_data.columns:
                fig = px.scatter(
                    clean_data,
                    x=x_col,
                    y=y_col,
                    color=dp_column,
                    title=plot_title,
                    labels={x_col: x_label, y_col: y_label},
                    color_discrete_map={'Revenue Share': '#1f77b4', 'Rental': '#ff7f0e', 'Both': '#2ca02c'}
                )
            else:
                color = {'Revenue Share': '#1f77b4', 'Rental': '#ff7f0e', 'Both': '#2ca02c'}.get(dp_filter, '#1f77b4')
                fig = px.scatter(
                    clean_data,
                    x=x_col,
                    y=y_col,
                    title=plot_title,
                    labels={x_col: x_label, y_col: y_label},
                    color_discrete_sequence=[color]
                )

            # Configure hover templates per analysis type
            if analysis_type == 'Tenure vs Online Hours':
                fig.update_traces(
                    customdata=np.stack([
                        clean_data['__name'],
                        clean_data.get(dp_column, pd.Series(['']*len(clean_data))),
                        hours_month,
                        hours_per_day,
                        clean_data.get('Tenure', pd.Series([np.nan]*len(clean_data))),
                        trips_per_day,
                        trips_month,
                    ], axis=-1),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        '<b>DP working plan:</b> %{customdata[1]}<br>'
                        '<b>Total online Hours per month:</b> %{customdata[2]:.2f}<br>'
                        '<b>Avg online Hours per day:</b> %{customdata[3]:.2f}<br>'
                        '<b>Tenure:</b> %{customdata[4]:.0f} days<br>'
                        '<b>Total no.of trips per day:</b> %{customdata[5]:.2f}<br>'
                        '<b>Total no.of trips per month:</b> %{customdata[6]:.0f}<br>'
                        '<extra></extra>'
                    )
                )
            elif analysis_type == 'Tenure vs Net Earnings':
                fig.update_traces(
                    customdata=np.stack([
                        clean_data['__name'],
                        clean_data.get(dp_column, pd.Series(['']*len(clean_data))),
                        clean_data.get('Tenure', pd.Series([np.nan]*len(clean_data))),
                        earnings_month,
                    ], axis=-1),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        '<b>DP working plan:</b> %{customdata[1]}<br>'
                        '<b>Tenure:</b> %{customdata[2]:.0f} days<br>'
                        '<b>Total Net earnings per month:</b> ₹%{customdata[3]:,.2f}<br>'
                        '<extra></extra>'
                    )
                )
            elif analysis_type == 'Online Hours vs Net Earnings':
                fig.update_traces(
                    customdata=np.stack([
                        clean_data['__name'],
                        hours_month,
                        earnings_month,
                        clean_data.get('Tenure', pd.Series([np.nan]*len(clean_data))),
                        clean_data.get(dp_column, pd.Series(['']*len(clean_data))),
                    ], axis=-1),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        '<b>Total Online Hours of that month:</b> %{customdata[1]:.2f}<br>'
                        '<b>Total Earnings of that month:</b> ₹%{customdata[2]:,.2f}<br>'
                        '<b>Tenure:</b> %{customdata[3]:.0f} days<br>'
                        '<b>DP working plan:</b> %{customdata[4]}<br>'
                        '<extra></extra>'
                    )
                )

            # Apply point size
            fig.update_traces(marker=dict(size=point_size))

            # Optional linear trend line
            if show_trend:
                try:
                    x_vals = clean_data[x_col].astype(float).values
                    y_vals = clean_data[y_col].astype(float).values
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if mask.sum() >= 2:
                        slope, intercept = np.polyfit(x_vals[mask], y_vals[mask], 1)
                        x_min, x_max = float(np.nanmin(x_vals[mask])), float(np.nanmax(x_vals[mask]))
                        x_line = np.linspace(x_min, x_max, 100)
                        y_line = intercept + slope * x_line
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode='lines',
                                name='Trend',
                                line=dict(color='#444', width=2, dash='dash')
                            )
                        )
                except Exception:
                    pass

            fig.update_layout(
                height=600,
                title_x=0.5,
                showlegend=True,
                font=dict(size=10)
            )

            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation plot: {str(e)}")
            return None
    
    def display_statistics(self, data, title, analysis_type, selected_month, dp_filter, tenure_filter):
        """Display statistics for filtered data"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No valid data for {title}")
                return
            
            # Get unique drivers
            unique_data = self.get_unique_drivers_data(data)
            
            # Display header
            month_text = f"{selected_month} - " if selected_month != 'All' else ""
            st.markdown(f'<div class="month-graph-header">{month_text}{title} - {analysis_type} ({dp_filter} - {tenure_filter})</div>', unsafe_allow_html=True)
            
            # Core metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_count = len(unique_data)
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Unique Drivers", f"{unique_count:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                avg_earnings = unique_data['total_earnings'].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Earnings", format_currency(avg_earnings))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                avg_tenure = unique_data['Tenure'].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Avg Tenure", f"{avg_tenure:.1f} days")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                if 'hours_online' in unique_data.columns:
                    # Average total online hours per month per unique driver
                    avg_hours = unique_data['hours_online'].mean()
                    st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Avg Total Online Hours per Month", f"{avg_hours:.1f} hrs")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Target achievement removed per request
            
        except Exception as e:
            st.error(f"Error displaying statistics: {str(e)}")
    
    def analysis_type_selector(self):
        """Analysis type selection"""
        st.markdown('<div class="analysis-selector">', unsafe_allow_html=True)
        st.markdown("### Analysis Type Selection")
        
        analysis_options = ['Tenure vs Net Earnings', 'Tenure vs Online Hours', 'Online Hours vs Net Earnings']
        
        selected_analysis = st.radio(
            "Select analysis type:",
            analysis_options,
            index=analysis_options.index(st.session_state.get('analysis_type', 'Tenure vs Net Earnings')),
            horizontal=True
        )
        
        if selected_analysis != st.session_state.get('analysis_type'):
            st.session_state.analysis_type = selected_analysis
            st.session_state.selected_point_data = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        return selected_analysis
    
    def sidebar_controls(self, analysis_type):
        """Sidebar controls"""
        st.sidebar.markdown("### Dashboard Controls")
        
        # Month selection
        st.sidebar.markdown("#### Select Month")
        available_months = self.get_available_months(self.data)
        month_options = ['All'] + list(available_months)
        
        selected_month = st.sidebar.selectbox(
            "Choose month:",
            month_options,
            index=3 if 'August' in month_options else 1,  # Default to August or first month
            help="Select which month to analyze"
        )
        
        # DP Working Plan selection
        st.sidebar.markdown("#### Select DP Working Plan")
        dp_options = ['Both', 'Revenue Share', 'Rental']
        dp_filter = st.sidebar.selectbox(
            "Choose DP Working Plan:",
            dp_options,
            index=0,  # Default to Both
            help="Select which DP Working Plan to analyze"
        )
        
        # Tenure filter
        st.sidebar.markdown("#### Tenure Filter")
        tenure_filter = st.sidebar.selectbox(
            "Select tenure range:",
            ['Both', 'Tenure more than 15 days', 'Tenure less than 15 days'],
            index=0,  # Default to Both
            help="Filter drivers based on tenure"
        )
        
        # Display options
        st.sidebar.markdown("#### Display Options")
        show_targets = st.sidebar.checkbox("Show Target Lines", value=True)
        show_trend = st.sidebar.checkbox("Show Trend Line", value=False)
        point_size = st.sidebar.slider("Point Size", 5, 15, 8)
        
        # Correlation options
        show_correlation = False
        
        # Data view options
        st.sidebar.markdown("#### Data View")
        if st.sidebar.button("Toggle Raw Data View"):
            st.session_state.show_raw_data = not st.session_state.show_raw_data
            st.rerun()
        
        return {
            'selected_month': selected_month,
            'dp_filter': dp_filter,
            'tenure_filter': tenure_filter,
            'show_targets': show_targets,
            'point_size': point_size,
            'show_trend': show_trend
        }
    
    def display_interactive_plots(self, controls, analysis_type):
        """Display interactive plots with proper filtering"""
        
        # Apply filters step by step
        filtered_data = self.data.copy()
        
        # Step 1: Month filter
        if controls['selected_month'] != 'All':
            filtered_data = self.apply_month_filter(filtered_data, controls['selected_month'])
            if filtered_data is None or len(filtered_data) == 0:
                st.warning(f"No data available for {controls['selected_month']}")
                return
        
        # Step 2: DP filter  
        if controls['dp_filter'] != 'Both':
            filtered_data = self.apply_dp_filter(filtered_data, controls['dp_filter'])
            if filtered_data is None or len(filtered_data) == 0:
                st.warning(f"No data available for {controls['dp_filter']} in {controls['selected_month']}")
                return
        
        # Step 3: Tenure filter
        if controls['tenure_filter'] != 'Both':
            filtered_data = self.apply_tenure_filter(filtered_data, controls['tenure_filter'])
            if filtered_data is None or len(filtered_data) == 0:
                st.warning(f"No data available for selected filters")
                return
        
        # Interactive features guide
        with st.expander("How to Use Interactive Features", expanded=False):
            if analysis_type == 'Tenure vs Net Earnings':
                st.markdown("""
                **Tenure vs Net Earnings Features:**
                - **Hover** over points for detailed driver information including name, DP working plan, tenure, and total net earnings
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when viewing all schemes
                """)
            elif analysis_type == 'Tenure vs Online Hours':
                st.markdown("""
                **Tenure vs Online Hours Features:**
                - **Hover** over points for detailed driver information including online hours, tenure, trips per day/month
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when viewing all schemes
                """)
            elif analysis_type == 'Online Hours vs Net Earnings':
                st.markdown("""
                **Online Hours vs Net Earnings Features:**
                - **Hover** over points for detailed driver information including driver name, total online hours, total earnings, tenure, and DP working plan
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when viewing all schemes
                """)
        
        # Display statistics
        title = f"Driver Analysis"
        self.display_statistics(
            filtered_data, 
            title, 
            analysis_type, 
            controls['selected_month'],
            controls['dp_filter'],
            controls['tenure_filter']
        )
        
        # Create and display plot
        fig = self.create_interactive_scatter_plot(
            filtered_data, 
            title,
            analysis_type,
            controls['selected_month'],
            controls['dp_filter'],
            controls['tenure_filter'],
            controls['show_targets'],
            controls['point_size'],
            controls['show_trend']
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not create plot with selected filters")

# Main App
def main():
    try:
        # App header
        st.markdown('<div class="main-header">Driver Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Background data loading (no visible upload UI)
        default_path = "data/Performance_No_Blank_Tenure_20250922_041646.csv"
        
        data = None
        if os.path.exists(default_path):
            with st.spinner("Loading data..."):
                data = load_driver_data(default_path)
        else:
            with st.spinner("Creating sample data..."):
                data = load_driver_data(default_path)
        
        if data is None:
            st.error("No data available. Please check the data source.")
            st.stop()
        
        # Generic success message
        st.success("Data loaded successfully!")
        
        # Initialize dashboard and ensure data integrity
        dashboard = DriverDashboard(data)
        
        # Ensure required columns exist
        data = dashboard.ensure_required_columns(data)
        
        # Update dashboard with clean data
        dashboard.data = data
        
        # Validate required columns
        required_columns = ['driver_email', 'driver_phone', 'Tenure', 'total_earnings']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Available columns: " + ", ".join(data.columns.tolist()))
            st.stop()
        
        # Check for unique driver identification columns
        if 'driver_email' not in data.columns and 'driver_phone' not in data.columns:
            st.error("Cannot identify unique drivers: both driver_email and driver_phone columns are missing")
            st.stop()
        
        # Data loaded message already shown above
        
        # Data preview and verification
        with st.expander("Data Preview & Monthly Breakdown", expanded=False):
            st.dataframe(data.head(10), use_container_width=True)
            
            # Monthly breakdown
            if 'Month' in data.columns:
                st.markdown("### Monthly Data Breakdown")
                monthly_stats = []
                
                for month in sorted(data['Month'].unique()):
                    month_data = data[data['Month'] == month]
                    
                    # Count unique drivers using email+phone combination
                    if 'driver_email' in month_data.columns and 'driver_phone' in month_data.columns:
                        unique_drivers = len(month_data.drop_duplicates(subset=['driver_email', 'driver_phone']))
                    elif 'driver_email' in month_data.columns:
                        unique_drivers = len(month_data['driver_email'].unique())
                    elif 'driver_phone' in month_data.columns:
                        unique_drivers = len(month_data['driver_phone'].unique())
                    else:
                        unique_drivers = len(month_data)  # Fallback
                    
                    # DP breakdown
                    dp_column = 'DP' if 'DP' in month_data.columns else 'Working Plan'
                    if dp_column in month_data.columns:
                        dp_counts = month_data[dp_column].value_counts()
                        revenue_share = dp_counts.get('Revenue Share', 0)
                        rental = dp_counts.get('Rental', 0) 
                        both = dp_counts.get('Both', 0)
                    else:
                        revenue_share = rental = both = 0
                    
                    monthly_stats.append({
                        'Month': month,
                        'Total Records': len(month_data),
                        'Unique Drivers (estimated)': unique_drivers,
                        'Revenue Share Records': revenue_share,
                        'Rental Records': rental,
                        'Both Records': both
                    })
                
                st.dataframe(pd.DataFrame(monthly_stats), use_container_width=True)
            
            # Overall stats
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Overview:**")
                st.write(f"- Total Records: {len(data):,}")
                st.write(f"- Unique Drivers: {len(data['id'].unique()) if 'id' in data.columns else 'N/A'}")
                st.write(f"- Months Available: {', '.join(sorted(data['Month'].unique())) if 'Month' in data.columns else 'N/A'}")
                
            with col2:
                st.write("**Key Metrics:**")
                if 'total_earnings' in data.columns:
                    st.write(f"- Avg Earnings: {format_currency(data['total_earnings'].mean())}")
                if 'Tenure' in data.columns:
                    st.write(f"- Avg Tenure: {data['Tenure'].mean():.1f} days")
                if 'hours_online' in data.columns:
                    st.write(f"- Avg Hours Online: {data['hours_online'].mean():.1f}")
        
        # Initialize dashboard
        dashboard = DriverDashboard(data)
        dashboard.initialize_session_state()
        
        # Analysis type selection
        analysis_type = dashboard.analysis_type_selector()
        
        # Sidebar controls
        controls = dashboard.sidebar_controls(analysis_type)
        
        # Main analysis and plots
        dashboard.display_interactive_plots(controls, analysis_type)
        
        # Raw data view
        if st.session_state.get('show_raw_data', False):
            st.markdown('<div class="section-header">Raw Data View</div>', unsafe_allow_html=True)
            
            # Apply same filters to show what's being plotted
            display_data = data.copy()
            display_data = dashboard.apply_month_filter(display_data, controls['selected_month'])
            display_data = dashboard.apply_dp_filter(display_data, controls['dp_filter'])
            display_data = dashboard.apply_tenure_filter(display_data, controls['tenure_filter'])
            unique_display_data = dashboard.get_unique_drivers_data(display_data)
            
            if unique_display_data is not None and len(unique_display_data) > 0:
                st.info(f"Showing {len(unique_display_data)} unique drivers matching your filters: {controls['selected_month']} - {controls['dp_filter']} - {controls['tenure_filter']}")
                st.dataframe(unique_display_data, use_container_width=True)
                
                # Download button
                csv_data = unique_display_data.to_csv(index=False)
                st.download_button(
                    label=f"Download Filtered Data ({controls['selected_month']} - {controls['dp_filter']})",
                    data=csv_data,
                    file_name=f"drivers_{controls['selected_month']}_{controls['dp_filter']}_{controls['tenure_filter']}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data matches current filters")
        
        # Footer removed per request
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Debug Information", expanded=False):
            st.exception(e)

if __name__ == "__main__":
    main()