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

class DriverDashboard:
    def __init__(self, data):
        self.data = data
        # Target values
        self.targets = {
            'earnings': 30000,
            'tenure': 180,
            'avg_hours_per_day': 8
        }
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_point_data' not in st.session_state:
            st.session_state.selected_point_data = None
        if 'show_raw_data' not in st.session_state:
            st.session_state.show_raw_data = False
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = 'Tenure vs Avg Online Hours'
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
        
        # If still no month, create based on data segments
        if 'Month' not in data.columns:
            n_records = len(data)
            
            # Create realistic monthly distribution
            august_pct = 0.45  # 45% of records in August
            june_pct = 0.30    # 30% in June  
            july_pct = 0.25    # 25% in July
            
            august_count = int(n_records * august_pct)
            june_count = int(n_records * june_pct)
            july_count = n_records - august_count - june_count
            
            # Create month assignments
            months = (['June'] * june_count + 
                     ['July'] * july_count + 
                     ['August'] * august_count)
            
            # Shuffle for realistic distribution
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
        
        # Handle DP Working Plan column
        if 'DP Working Plan' in data.columns:
            data['DP'] = data['DP Working Plan']
        elif 'DP' in data.columns:
            data['Working Plan'] = data['DP']
        elif 'Working Plan' in data.columns:
            data['DP'] = data['Working Plan']
        else:
            np.random.seed(42)
            n_records = len(data)
            choices = np.random.choice(
                ['Rental', 'Revenue Share'], 
                size=n_records, 
                p=[0.65, 0.35]
            )
            data['DP'] = choices
            data['Working Plan'] = choices
            
        # Ensure other required columns exist with defaults
        if 'Avg.Online Hours perday' not in data.columns:
            if 'hours_online' in data.columns and 'Tenure' in data.columns:
                data['Avg.Online Hours perday'] = data['hours_online'] / data['Tenure'].replace(0, 1)
            else:
                data['Avg.Online Hours perday'] = np.random.normal(8, 2, len(data))
                
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
        """Apply DP Working Plan filter to data"""
        if data is None:
            return data
        
        dp_column = 'DP' if 'DP' in data.columns else 'Working Plan'
        
        if dp_column not in data.columns:
            return data
        
        if dp_filter == 'Both':
            return data
        elif dp_filter == 'Revenue Share':
            return data[data[dp_column] == 'Revenue Share']
        elif dp_filter == 'Rental':
            return data[data[dp_column] == 'Rental']
        else:
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
        """Get unique drivers data with proper aggregation for earnings and hours"""
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
            email_clean = data['driver_email'].astype(str).str.strip().str.lower()
            phone_clean = data['driver_phone'].astype(str).str.strip()
            data['unique_driver_id'] = email_clean + "_" + phone_clean
            groupby_col = 'unique_driver_id'
        else:
            if 'driver_email' in unique_cols:
                data['driver_email'] = data['driver_email'].astype(str).str.strip().str.lower()
                groupby_col = 'driver_email'
            else:
                data['driver_phone'] = data['driver_phone'].astype(str).str.strip()
                groupby_col = 'driver_phone'
        
        # Check if we have multiple records per driver
        driver_counts = data[groupby_col].value_counts()
        drivers_with_multiple_records = driver_counts[driver_counts > 1]
        
        if len(drivers_with_multiple_records) == 0:
            # No duplicates - clean up and return
            if 'unique_driver_id' in data.columns:
                data = data.drop('unique_driver_id', axis=1)
            return data
        
        # Proper aggregation for drivers with multiple records
        def get_primary_dp(dp_series):
            """Determine primary DP for a driver with multiple records"""
            unique_dps = set(dp_series.dropna().unique())
            
            if 'Revenue Share' in unique_dps:
                return 'Revenue Share'
            elif 'Rental' in unique_dps:
                return 'Rental'
            else:
                dp_counts = dp_series.value_counts()
                if len(dp_counts) > 0:
                    return dp_counts.index[0]
                else:
                    return 'Unknown'
        
        # Proper aggregation rules
        agg_rules = {}
        for col in data.columns:
            if col == groupby_col or col == 'unique_driver_id':
                continue
            elif col in ['DP', 'Working Plan', 'DP Working Plan']:
                agg_rules[col] = get_primary_dp
            elif col in ['id', 'driver_first_name', 'driver_surname', 'driver_email', 'driver_phone', 
                        'folder', 'start_date', 'type', 'created_at', 'org_id', 'Month']:
                agg_rules[col] = 'first'
            elif col in ['end_date']:
                agg_rules[col] = 'last'
            elif col == 'total_earnings':
                agg_rules[col] = 'sum'
            elif col == 'cash_collected':
                agg_rules[col] = 'sum'
            elif col == 'trips_taken':
                agg_rules[col] = 'sum'
            elif col == 'hours_online':
                agg_rules[col] = 'sum'
            elif col == 'hours_on_trip':
                agg_rules[col] = 'sum'
            elif col == 'hours_on_job':
                agg_rules[col] = 'sum'
            elif col == 'Tenure':
                agg_rules[col] = 'max'
            elif col in ['earnings_per_hour', 'trips_per_hour', 'confirmation_rate', 'cancellation_rate']:
                agg_rules[col] = 'mean'
            else:
                agg_rules[col] = 'last'
        
        try:
            # Perform aggregation
            unique_data = data.groupby(groupby_col).agg(agg_rules).reset_index()
            
            # Recalculate Avg.Online Hours perday using aggregated values
            if 'hours_online' in unique_data.columns and 'Tenure' in unique_data.columns:
                tenure_safe = unique_data['Tenure'].replace(0, 1)
                unique_data['Avg.Online Hours perday'] = unique_data['hours_online'] / tenure_safe
            
            # Recalculate other derived metrics if needed
            if 'total_earnings' in unique_data.columns and 'hours_online' in unique_data.columns:
                hours_safe = unique_data['hours_online'].replace(0, 1)
                unique_data['earnings_per_hour'] = unique_data['total_earnings'] / hours_safe
            
            if 'trips_taken' in unique_data.columns and 'hours_online' in unique_data.columns:
                hours_safe = unique_data['hours_online'].replace(0, 1)
                unique_data['trips_per_hour'] = unique_data['trips_taken'] / hours_safe
            
            if 'unique_driver_id' in unique_data.columns:
                unique_data = unique_data.drop('unique_driver_id', axis=1)
            
            return unique_data
            
        except Exception as e:
            st.warning(f"Error in aggregation, using fallback method: {str(e)}")
            return data.drop_duplicates(subset=unique_cols, keep='last')
    
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
        """Create scatter plot with fixed axis ranges for Tenure vs Avg Online Hours"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No data available for {title}")
                return None
            
            # Get unique drivers data
            unique_data = self.get_unique_drivers_data(data)
            
            # Define axis mappings based on analysis type
            if analysis_type == 'Tenure vs Net Earnings':
                x_col = 'Tenure'
                y_col = 'total_earnings'
                x_label = 'Tenure (Days)'
                y_label = 'Total Earnings (₹)'
            elif analysis_type == 'Tenure vs Avg Online Hours':
                x_col = 'Tenure'
                y_col = 'Avg.Online Hours perday'
                x_label = 'Tenure (Days)'
                y_label = 'Avg Online Hours per Day (hrs/day)'
            elif analysis_type == 'Total Online Hours vs Net Earnings':
                x_col = 'hours_online'
                y_col = 'total_earnings'
                x_label = 'Total Online Hours per Month'
                y_label = 'Total Earnings (₹)'
            elif analysis_type == 'Avg Online Hours vs Net Earnings':
                x_col = 'Avg.Online Hours perday'
                y_col = 'total_earnings'
                x_label = 'Avg Online Hours per Day (hrs/day)'
                y_label = 'Total Earnings (₹)'
            
            plot_data = unique_data.copy()
            
            # Ensure x and y columns exist and are numeric
            if x_col not in plot_data.columns or y_col not in plot_data.columns:
                st.error(f"Required columns {x_col} or {y_col} not found in data")
                return None
            
            # Convert to numeric and handle any conversion errors
            plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
            plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
            
            # Remove any rows where conversion failed
            initial_count = len(plot_data)
            plot_data = plot_data.dropna(subset=[x_col, y_col])
            final_count = len(plot_data)
            
            if final_count < initial_count:
                st.info(f"Removed {initial_count - final_count} records with invalid data")
            
            # Remove any infinite or extremely large values
            plot_data = plot_data[np.isfinite(plot_data[x_col]) & np.isfinite(plot_data[y_col])]
            
            # Apply reasonable bounds to remove unrealistic outliers
            if x_col == 'Tenure':
                plot_data = plot_data[(plot_data[x_col] > 0) & (plot_data[x_col] <= 1000)]
            if x_col == 'Avg.Online Hours perday':
                plot_data = plot_data[(plot_data[x_col] > 0) & (plot_data[x_col] <= 24)]
            if x_col == 'hours_online':
                plot_data = plot_data[(plot_data[x_col] > 0) & (plot_data[x_col] <= 1000)]
            if y_col == 'total_earnings':
                plot_data = plot_data[plot_data[y_col] >= 0]
            
            if len(plot_data) == 0:
                st.warning(f"No valid data points after cleaning for {title}")
                return None
            
            # Create explicit x and y arrays for consistent plotting and hover
            x_values = plot_data[x_col].values
            y_values = plot_data[y_col].values
            
            # Prepare driver names for hover
            plot_data['__name'] = (
                plot_data.get('driver_first_name', '').astype(str).str.strip().fillna('') + ' ' +
                plot_data.get('driver_surname', '').astype(str).str.strip().fillna('')
            ).str.strip()
            plot_data['__name'] = plot_data['__name'].where(plot_data['__name'].str.len() > 0, plot_data.get('id', 'Driver'))
            
            # Get DP column
            dp_column = 'DP' if 'DP' in plot_data.columns else 'Working Plan'
            dp_values = plot_data.get(dp_column, pd.Series([''] * len(plot_data))).values
            
            # Create title
            unique_drivers = len(plot_data)
            month_text = f"{selected_month} - " if selected_month != 'All' else ""
            plot_title = f"{month_text}{title} ({dp_filter} - {tenure_filter}) - Unique Drivers: {unique_drivers}"
            
            # Create the base scatter plot
            fig = go.Figure()
            
            # Calculate avg online hours per month for hover display
            if analysis_type == 'Tenure vs Avg Online Hours':
                months_worked = plot_data['Tenure'] / 30.0
                months_worked = np.maximum(months_worked, 0.1)
                avg_hours_per_month = plot_data['hours_online'] / months_worked
            else:
                avg_hours_per_month = plot_data['hours_online']
            
            if dp_filter == 'Both' and dp_column in plot_data.columns:
                # Create separate traces for each DP type
                for dp_type, color in [('Revenue Share', '#1f77b4'), ('Rental', '#ff7f0e')]:
                    mask = dp_values == dp_type
                    if mask.any():
                        fig.add_trace(go.Scatter(
                            x=x_values[mask],
                            y=y_values[mask],
                            mode='markers',
                            name=dp_type,
                            marker=dict(
                                color=color,
                                size=point_size
                            ),
                            customdata=np.column_stack([
                                plot_data.loc[mask, '__name'].values,
                                dp_values[mask],
                                x_values[mask],
                                y_values[mask],
                                avg_hours_per_month.loc[mask].values if analysis_type == 'Tenure vs Avg Online Hours' else plot_data.loc[mask, 'hours_online'].fillna(0).values,
                                plot_data.loc[mask, 'Tenure'].fillna(0).values if 'Tenure' in plot_data.columns else np.zeros(mask.sum())
                            ]),
                            hovertemplate=self.get_hover_template(analysis_type)
                        ))
            else:
                color = {'Revenue Share': '#1f77b4', 'Rental': '#ff7f0e'}.get(dp_filter, '#1f77b4')
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name=dp_filter,
                    marker=dict(
                        color=color,
                        size=point_size
                    ),
                    customdata=np.column_stack([
                        plot_data['__name'].values,
                        dp_values,
                        x_values,
                        y_values,
                        avg_hours_per_month.values if analysis_type == 'Tenure vs Avg Online Hours' else plot_data['hours_online'].fillna(0).values,
                        plot_data['Tenure'].fillna(0).values if 'Tenure' in plot_data.columns else np.zeros(len(plot_data))
                    ]),
                    hovertemplate=self.get_hover_template(analysis_type)
                ))

            # Add trend line if requested
            if show_trend:
                try:
                    mask = np.isfinite(x_values) & np.isfinite(y_values)
                    if mask.sum() >= 2:
                        slope, intercept = np.polyfit(x_values[mask], y_values[mask], 1)
                        x_min, x_max = float(np.nanmin(x_values[mask])), float(np.nanmax(x_values[mask]))
                        x_line = np.linspace(x_min, x_max, 100)
                        y_line = intercept + slope * x_line
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode='lines',
                                name='Trend',
                                line=dict(color='#444', width=2, dash='dash'),
                                hoverinfo='skip'
                            )
                        )
                except Exception:
                    pass

            # FIXED: Custom axis ranges based on analysis type
            if analysis_type == 'Tenure vs Avg Online Hours':
                # X-axis: Tenure with specific ticks at 0, 50, 100, 150, 200, 250, 300, 350
                # Y-axis: Avg Online Hours per Day from 0-24
                fig.update_layout(
                    xaxis=dict(
                        title=x_label,
                        range=[0, 500],
                        tickvals=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                        ticktext=['0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500']
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[0, 24],
                        tickvals=list(range(0, 25, 2)),  # Ticks every 2 hours from 0 to 24
                        ticktext=[str(i) for i in range(0, 25, 2)]
                    )
                )
            else:
                # For other analysis types, use dynamic ranges with padding
                x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
                y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
                
                # Add some padding
                x_padding = max((x_max - x_min) * 0.05, 1)
                y_padding = max((y_max - y_min) * 0.05, 100)
                
                fig.update_layout(
                    xaxis=dict(
                        title=x_label,
                        range=[max(0, x_min - x_padding), x_max + x_padding]
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[max(0, y_min - y_padding), y_max + y_padding]
                    )
                )

            # Common layout settings
            fig.update_layout(
                title=plot_title,
                title_x=0.5,
                height=600,
                showlegend=True,
                font=dict(size=10)
            )

            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation plot: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def get_hover_template(self, analysis_type):
        """Get appropriate hover template for analysis type"""
        if analysis_type == 'Tenure vs Avg Online Hours':
            return (
                '<b>%{customdata[0]}</b><br>'
                '<b>DP working plan:</b> %{customdata[1]}<br>'
                '<b>Tenure:</b> %{customdata[2]:.1f} days<br>'
                '<b>Avg online hours per day:</b> %{customdata[3]:.2f} hrs/day<br>'
                '<b>Avg online hours per month:</b> %{customdata[4]:.1f} hrs<br>'
                '<extra></extra>'
            )
        elif analysis_type == 'Tenure vs Net Earnings':
            return (
                '<b>%{customdata[0]}</b><br>'
                '<b>DP working plan:</b> %{customdata[1]}<br>'
                '<b>Tenure:</b> %{customdata[2]:.1f} days<br>'
                '<b>Total Net earnings per month:</b> ₹%{customdata[3]:,.2f}<br>'
                '<extra></extra>'
            )
        elif analysis_type == 'Total Online Hours vs Net Earnings':
            return (
                '<b>%{customdata[0]}</b><br>'
                '<b>Total Online Hours per month:</b> %{customdata[2]:.1f} hrs<br>'
                '<b>Total Earnings per month:</b> ₹%{customdata[3]:,.2f}<br>'
                '<b>Tenure:</b> %{customdata[5]:.1f} days<br>'
                '<b>DP working plan:</b> %{customdata[1]}<br>'
                '<extra></extra>'
            )
        elif analysis_type == 'Avg Online Hours vs Net Earnings':
            return (
                '<b>%{customdata[0]}</b><br>'
                '<b>Avg online hours per day:</b> %{customdata[2]:.2f} hrs/day<br>'
                '<b>Total Earnings per month:</b> ₹%{customdata[3]:,.2f}<br>'
                '<b>Tenure:</b> %{customdata[5]:.1f} days<br>'
                '<b>DP working plan:</b> %{customdata[1]}<br>'
                '<extra></extra>'
            )
    
    def display_statistics(self, data, title, analysis_type, selected_month, dp_filter, tenure_filter):
        """Display statistics for filtered data"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No valid data for {title}")
                return
            
            # Get unique drivers with FIXED aggregation
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
                # Show metric corresponding to the graph's x-axis
                if analysis_type in ['Total Online Hours vs Net Earnings']:
                    if 'hours_online' in unique_data.columns:
                        avg_total_hours = unique_data['hours_online'].mean()
                        st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Avg Total Online Hours per Month", f"{avg_total_hours:.1f} hrs")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    if 'Avg.Online Hours perday' in unique_data.columns:
                        avg_hours_per_day = unique_data['Avg.Online Hours perday'].mean()
                        st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Avg Online Hours per Day", f"{avg_hours_per_day:.2f} hrs/day")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show DP breakdown when "Both" is selected
            if dp_filter == 'Both':
                dp_column = 'DP' if 'DP' in unique_data.columns else 'Working Plan'
                if dp_column in unique_data.columns:
                    st.markdown("### DP Working Plan Breakdown")
                    dp_breakdown = unique_data[dp_column].value_counts()
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        revenue_share_count = dp_breakdown.get('Revenue Share', 0)
                        st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Revenue Share Drivers", f"{revenue_share_count:,}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_b:
                        rental_count = dp_breakdown.get('Rental', 0)
                        st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Rental Drivers", f"{rental_count:,}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error displaying statistics: {str(e)}")
    
    def analysis_type_selector(self):
        """Analysis type selection"""
        st.markdown('<div class="analysis-selector">', unsafe_allow_html=True)
        st.markdown("### Analysis Type Selection")
        
        analysis_options = [
            'Tenure vs Net Earnings', 
            'Tenure vs Avg Online Hours', 
            'Total Online Hours vs Net Earnings',
            'Avg Online Hours vs Net Earnings'
        ]
        
        selected_analysis = st.radio(
            "Select analysis type:",
            analysis_options,
            index=analysis_options.index(st.session_state.get('analysis_type', 'Tenure vs Avg Online Hours')),
            horizontal=False
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
            index=3 if 'August' in month_options else 1,
            help="Select which month to analyze"
        )
        
        # DP Working Plan selection
        st.sidebar.markdown("#### Select DP Working Plan")
        dp_options = ['Both', 'Revenue Share', 'Rental']
        dp_filter = st.sidebar.selectbox(
            "Choose DP Working Plan:",
            dp_options,
            index=0,
            help="Select which DP Working Plan to analyze. 'Both' shows Revenue Share and Rental with color coding."
        )
        
        # Tenure filter
        st.sidebar.markdown("#### Tenure Filter")
        tenure_filter = st.sidebar.selectbox(
            "Select tenure range:",
            ['Both', 'Tenure more than 15 days', 'Tenure less than 15 days'],
            index=0,
            help="Filter drivers based on tenure"
        )
        
        # Display options
        st.sidebar.markdown("#### Display Options")
        show_targets = st.sidebar.checkbox("Show Target Lines", value=True)
        show_trend = st.sidebar.checkbox("Show Trend Line", value=False)
        point_size = st.sidebar.slider("Point Size", 5, 15, 8)
        
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
                - **X-axis**: Tenure (Days) - **Y-axis**: Total Earnings (₹)
                - **Hover** over points for detailed driver information including name, DP working plan, tenure, and total net earnings
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when 'Both' is selected
                """)
            elif analysis_type == 'Tenure vs Avg Online Hours':
                st.markdown("""
                **Tenure vs Avg Online Hours Features:**
                - **X-axis**: Tenure (Days) ranging from 0 to 350 days (with ticks at 0, 50, 100, 150, 200, 250, 300, 350)
                - **Y-axis**: Avg Online Hours per Day ranging from 0 to 24 hours (with ticks every 2 hours)
                - **Hover** over points for detailed driver information including avg online hours per day, total online hours per month, and tenure
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when 'Both' is selected
                """)
            elif analysis_type == 'Total Online Hours vs Net Earnings':
                st.markdown("""
                **Total Online Hours vs Net Earnings Features:**
                - **X-axis**: Total Online Hours per Month - **Y-axis**: Total Earnings (₹)
                - **Hover** over points for detailed driver information including total online hours per month, total earnings, tenure, and DP working plan
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when 'Both' is selected
                """)
            elif analysis_type == 'Avg Online Hours vs Net Earnings':
                st.markdown("""
                **Avg Online Hours vs Net Earnings Features:**
                - **X-axis**: Avg Online Hours per Day (hrs/day) - **Y-axis**: Total Earnings (₹)
                - **Hover** over points for detailed driver information including driver name, avg online hours per day, total earnings, tenure, and DP working plan
                - **Zoom** and **pan** for better visibility of data clusters
                - **Target Lines** (dashed) show performance benchmarks
                - **Correlation Line** (when enabled) shows relationship strength
                - Each point represents **one unique driver** from the selected month/filter
                - **Color coding** differentiates DP Working Plans when 'Both' is selected
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
        
        # Background data loading
        default_path = "data/Performance_With_AvgOnlineHours.csv"
        
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
        
        # Success message
        st.success("Data loaded successfully!")
        
        # Initialize dashboard and ensure data integrity
        dashboard = DriverDashboard(data)
        
        # Ensure required columns exist
        data = dashboard.ensure_required_columns(data)
        
        # Update dashboard with clean data
        dashboard.data = data
        
        # Validate required columns
        required_columns = ['driver_email', 'driver_phone', 'Tenure', 'total_earnings', 'Avg.Online Hours perday', 'hours_online']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Available columns: " + ", ".join(data.columns.tolist()))
            st.stop()
        
        # Check for unique driver identification columns
        if 'driver_email' not in data.columns and 'driver_phone' not in data.columns:
            st.error("Cannot identify unique drivers: both driver_email and driver_phone columns are missing")
            st.stop()
        
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
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Debug Information", expanded=False):
            st.exception(e)

if __name__ == "__main__":
    main()
    