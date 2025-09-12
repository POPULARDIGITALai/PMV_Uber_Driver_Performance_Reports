import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
import os

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
    .plotly-graph-title {
        text-align: center !important;
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
def load_driver_data():
    """Load driver data from Excel file"""
    # Try multiple possible file paths
    possible_paths = [
        os.path.join('app_v2', 'data', 'DriverMIS_Data_22Aug.xlsx'),
        os.path.join('data', 'DriverMIS_Data_22Aug.xlsx'),
        'DriverMIS_Data_22Aug.xlsx',
        os.path.join('..', 'data', 'DriverMIS_Data_22Aug.xlsx')
    ]
    
    data_dict = {}
    file_found = False
    
    for file_path in possible_paths:
        try:
            if os.path.exists(file_path):
                # Load Revenue Share data
                try:
                    revenue_df = pd.read_excel(file_path, sheet_name='Revenue Share')
                    data_dict['Revenue Share'] = revenue_df
                except Exception as e:
                    pass
                
                # Load Rental data
                try:
                    rental_df = pd.read_excel(file_path, sheet_name='Rental')
                    data_dict['Rental'] = rental_df
                except Exception as e:
                    pass
                
                file_found = True
                break
                
        except Exception as e:
            continue
    
    if not file_found or not data_dict:
        # Create sample data if file not found
        data_dict = create_sample_data()
    
    return data_dict

@st.cache_data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Sample Revenue Share data
    revenue_data = {
        'PMV ID': [f'REV_{i:04d}' for i in range(1, 501)],
        'Driver Id': [f'REV_{i:04d}' for i in range(1, 501)],
        'Name': [f'Driver_{i}' for i in range(1, 501)],
        'Tenure(Days)': np.random.randint(5, 730, 500),  # Include some < 15 days
        'Net Earnings (Toll - Tip)': np.random.normal(25000, 8000, 500),
        'Online Hours': np.random.normal(56, 14, 500),  # Weekly hours
        'Trips': np.random.randint(15, 45, 500),
        'Month': np.random.choice(['June', 'July', 'August'], 500),
        'Acceptance Rate': np.random.uniform(70, 95, 500),
        'VBI': np.random.uniform(3.5, 4.8, 500),
        'Scheme': 'Revenue Share'
    }
    
    # Sample Rental data
    rental_data = {
        'PMV ID': [f'REN_{i:04d}' for i in range(1, 351)],
        'Driver Id': [f'REN_{i:04d}' for i in range(1, 351)],
        'Name': [f'Driver_{i+500}' for i in range(1, 351)],
        'Tenure(Days)': np.random.randint(5, 600, 350),  # Include some < 15 days
        'Net Earnings (Toll - Tip)': np.random.normal(22000, 6000, 350),
        'Online Hours': np.random.normal(52.5, 12.6, 350),  # Weekly hours
        'Trips': np.random.randint(12, 40, 350),
        'Month': np.random.choice(['June', 'July', 'August'], 350),
        'Acceptance Rate': np.random.uniform(65, 90, 350),
        'VBI': np.random.uniform(3.2, 4.5, 350),
        'Scheme': 'Rental'
    }
    
    return {
        'Revenue Share': pd.DataFrame(revenue_data),
        'Rental': pd.DataFrame(rental_data)
    }

class DriverDashboard:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Target values
        self.targets = {
            'revenue': 30000,
            'tenure': 180,
            'hours_per_day': 10
        }
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_point_data' not in st.session_state:
            st.session_state.selected_point_data = None
        if 'show_raw_data' not in st.session_state:
            st.session_state.show_raw_data = False
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = 'Tenure vs Net Earnings'
    
    def get_pmv_id_column(self, data):
        """Get the correct PMV ID column name from the data"""
        possible_names = ['PMV ID', 'PMV_ID', 'pmv_id', 'PMV Id', 'PMVID']
        for col in possible_names:
            if col in data.columns:
                return col
        # Fallback to Driver Id if PMV ID not found
        driver_id_names = ['Driver Id', 'Driver UUID', 'driver_id', 'driver_uuid', 'Driver_Id', 'DriverId']
        for col in driver_id_names:
            if col in data.columns:
                return col
        return None
    
    def normalize_month_data(self, df):
        """Normalize month data to handle different formats"""
        if 'Month' not in df.columns:
            return df
        
        df = df.copy()
        
        # Convert datetime to month names if needed
        if pd.api.types.is_datetime64_any_dtype(df['Month']):
            df['Month'] = df['Month'].dt.strftime('%B')
        else:
            df['Month'] = df['Month'].astype(str)
        
        # Clean up any NaN or invalid entries
        df = df[df['Month'].notna() & (df['Month'] != 'nan') & (df['Month'] != 'NaT')]
        
        return df
    
    def get_aggregated_stats(self, data, pmv_id_col):
        """Get aggregated statistics for unique PMV IDs"""
        if data is None or len(data) == 0:
            return None
        
        # Group by PMV ID and calculate SUMS for performance metrics and MEAN for driver attributes
        agg_dict = {
            'Net Earnings (Toll - Tip)': 'sum',
            'Trips': 'sum',
            'Online Hours': 'sum',  # This is weekly hours, we'll convert to daily later
            'Tenure(Days)': 'mean',
            'Acceptance Rate': 'mean',
            'VBI': 'mean',
        }
        
        # Only include columns that exist in the data
        final_agg_dict = {}
        for col, func in agg_dict.items():
            if col in data.columns:
                final_agg_dict[col] = func
        
        # Group by PMV ID and aggregate
        aggregated_data = data.groupby(pmv_id_col).agg(final_agg_dict).reset_index()
        
        # Add back other identifying information
        id_columns = ['Month', 'Scheme', 'Name']
        for col in id_columns:
            if col in data.columns:
                first_occurrence = data.groupby(pmv_id_col)[col].first().reset_index()
                aggregated_data = aggregated_data.merge(first_occurrence, on=pmv_id_col)
        
        # Calculate Online Hours per Day (assuming Online Hours is weekly data)
        if 'Online Hours' in aggregated_data.columns:
            aggregated_data['Online Hours Per Day'] = aggregated_data['Online Hours'] / 7
        
        return aggregated_data
    
    def apply_tenure_filter(self, data, tenure_filter):
        """Apply tenure filter to data"""
        if data is None or 'Tenure(Days)' not in data.columns:
            return data
        
        if tenure_filter == 'Tenure less than 15 days':
            return data[data['Tenure(Days)'] < 15]
        elif tenure_filter == 'Tenure more than 15 days':
            return data[data['Tenure(Days)'] >= 15]
        else:  # Both
            return data
    
    def combine_schemes_data(self, selected_schemes, selected_month, tenure_filter):
        """Combine data from selected schemes"""
        combined_data = []
        
        for scheme in selected_schemes:
            if scheme in self.data_dict:
                df = self.data_dict[scheme].copy()
                df = self.normalize_month_data(df)
                
                if 'Month' in df.columns:
                    df = df[df['Month'] == selected_month]
                
                # Apply tenure filter
                df = self.apply_tenure_filter(df, tenure_filter)
                
                if len(df) > 0:
                    df['Scheme'] = scheme  # Ensure scheme column exists
                    combined_data.append(df)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return None
    
    def create_interactive_scatter_plot(self, data, title, analysis_type, month, tenure_filter, show_targets=True, point_size=8):
        """Create an interactive scatter plot with target lines"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No data available for {title}")
                return None
            
            # Get PMV ID column
            pmv_id_col = self.get_pmv_id_column(data)
            if not pmv_id_col:
                st.error(f"PMV ID column not found in {title}")
                return None
            
            # Get aggregated data for unique PMV IDs
            aggregated_data = self.get_aggregated_stats(data, pmv_id_col)
            if aggregated_data is None or len(aggregated_data) == 0:
                st.warning(f"No valid data points for {title}")
                return None
            
            # Prepare data based on analysis type
            if analysis_type == 'Tenure vs Net Earnings':
                x_col = 'Tenure(Days)'
                y_col = 'Net Earnings (Toll - Tip)'
                x_label = 'Tenure (Days)'
                y_label = 'Net Earnings (₹)'
                clean_data = aggregated_data.dropna(subset=[x_col, y_col])
            else:  # Online Hours Per Day vs Net Earnings
                x_col = 'Online Hours Per Day'
                y_col = 'Net Earnings (Toll - Tip)'
                x_label = 'Online Hours Per Day'
                y_label = 'Net Earnings (₹)'
                clean_data = aggregated_data.dropna(subset=[x_col, y_col])
            
            if len(clean_data) == 0:
                st.warning(f"No valid data points for {title}")
                return None
            
            # Create the scatter plot
            unique_drivers = len(clean_data)
            plot_title = f'{month} - {analysis_type} ({tenure_filter}) - Unique PMV IDs: {unique_drivers}'
            
            # Create comprehensive hover data based on requirements
            hover_template = "<b>%{customdata[0]}</b><br>"  # Driver Name
            hover_template += "PMV ID: %{customdata[1]}<br>"
            hover_template += "Total Earnings: ₹%{customdata[2]:,.2f}<br>"
            hover_template += "Total Trips: %{customdata[3]:,.0f}<br>"
            hover_template += "Month: %{customdata[4]}<br>"
            hover_template += "Scheme: %{customdata[5]}<br>"
            
            if analysis_type == 'Tenure vs Net Earnings':
                hover_template += "Total Tenure Days: %{customdata[6]:.1f}<br>"
            else:  # Online Hours Per Day vs Net Earnings
                hover_template += "Total Online Hours: %{customdata[6]:.1f} hrs<br>"
                hover_template += "Online Hours per Day: %{customdata[7]:.1f} hrs<br>"
            
            hover_template += "<extra></extra>"  # Remove trace box
            
            # Prepare custom data for hover
            if analysis_type == 'Tenure vs Net Earnings':
                customdata = np.column_stack([
                    clean_data.get('Name', ['N/A'] * len(clean_data)),
                    clean_data[pmv_id_col],
                    clean_data['Net Earnings (Toll - Tip)'],
                    clean_data.get('Trips', [0] * len(clean_data)),
                    clean_data.get('Month', ['N/A'] * len(clean_data)),
                    clean_data.get('Scheme', ['N/A'] * len(clean_data)),
                    clean_data['Tenure(Days)']
                ])
            else:  # Online Hours Per Day vs Net Earnings
                customdata = np.column_stack([
                    clean_data.get('Name', ['N/A'] * len(clean_data)),
                    clean_data[pmv_id_col],
                    clean_data['Net Earnings (Toll - Tip)'],
                    clean_data.get('Trips', [0] * len(clean_data)),
                    clean_data.get('Month', ['N/A'] * len(clean_data)),
                    clean_data.get('Scheme', ['N/A'] * len(clean_data)),
                    clean_data.get('Online Hours', [0] * len(clean_data)),
                    clean_data.get('Online Hours Per Day', [0] * len(clean_data))
                ])
            
            # Create different colors for schemes if multiple schemes
            if 'Both' in title or len(clean_data['Scheme'].unique()) > 1:
                fig = px.scatter(
                    clean_data,
                    x=x_col,
                    y=y_col,
                    color='Scheme',
                    title=plot_title,
                    labels={x_col: x_label, y_col: y_label},
                    color_discrete_map={
                        'Revenue Share': '#1f77b4',
                        'Rental': '#ff7f0e'
                    }
                )
            else:
                # Single scheme color
                scheme_colors = {
                    'Revenue Share': '#1f77b4',
                    'Rental': '#ff7f0e'
                }
                scheme = clean_data['Scheme'].iloc[0] if 'Scheme' in clean_data.columns else 'Unknown'
                color = scheme_colors.get(scheme, '#1f77b4')
                
                fig = px.scatter(
                    clean_data,
                    x=x_col,
                    y=y_col,
                    title=plot_title,
                    labels={x_col: x_label, y_col: y_label},
                    color_discrete_sequence=[color]
                )
            
            # Update traces with custom hover template and data
            fig.update_traces(
                customdata=customdata,
                hovertemplate=hover_template,
                marker=dict(size=point_size, opacity=0.7, line=dict(width=1, color='white'))
            )
            
            # Add target lines
            if show_targets:
                if analysis_type == 'Tenure vs Net Earnings':
                    # Horizontal line at 30K revenue
                    fig.add_hline(
                        y=self.targets['revenue'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Revenue Target: ₹{self.targets['revenue']:,}",
                        annotation_position="bottom right"
                    )
                    # Vertical line at 180 days tenure
                    fig.add_vline(
                        x=self.targets['tenure'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Tenure Target: {self.targets['tenure']} days",
                        annotation_position="top left"
                    )
                else:  # Online Hours Per Day vs Net Earnings
                    # Horizontal line at 30K revenue
                    fig.add_hline(
                        y=self.targets['revenue'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Revenue Target: ₹{self.targets['revenue']:,}",
                        annotation_position="bottom right"
                    )
                    # Vertical line at 10 hours/day
                    fig.add_vline(
                        x=self.targets['hours_per_day'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Hours/Day Target: {self.targets['hours_per_day']} hrs",
                        annotation_position="top left"
                    )
            
            # Update layout with centered title
            fig.update_layout(
                height=600,
                hovermode='closest',
                title_x=0.5,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot for {title}: {str(e)}")
            return None
    
    def analysis_type_selector(self):
        """Analysis type selection section"""
        st.markdown('<div class="analysis-selector">', unsafe_allow_html=True)
        st.markdown("### 🎯 Choose Analysis Type")
        
        analysis_options = ['Tenure vs Net Earnings', 'Online Hours Per Day vs Net Earnings']
        
        selected_analysis = st.radio(
            "Select the type of analysis:",
            analysis_options,
            index=analysis_options.index(st.session_state.get('analysis_type', 'Tenure vs Net Earnings')),
            horizontal=True,
            key="analysis_type_radio"
        )
        
        # Update session state
        if selected_analysis != st.session_state.get('analysis_type'):
            st.session_state.analysis_type = selected_analysis
            st.session_state.selected_point_data = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        return selected_analysis
    
    def sidebar_controls(self):
        """Sidebar controls"""
        st.sidebar.markdown("### 🎛️ Dashboard Controls")
        
        # Month selection dropdown
        available_months = set()
        for scheme, df in self.data_dict.items():
            if df is not None and 'Month' in df.columns:
                normalized_df = self.normalize_month_data(df)
                months = normalized_df['Month'].unique()
                available_months.update([m for m in months if pd.notna(m) and m != 'nan'])
        
        if not available_months:
            available_months = ['June', 'July', 'August']
        
        month_options = sorted(list(available_months))
        selected_month = st.sidebar.selectbox(
            "📅 Select Month",
            month_options,
            index=0,
            help="Choose a month to display"
        )
        
        # Scheme selection - Radio buttons for Revenue Share, Rental, Both
        st.sidebar.markdown("### 🎯 Select Scheme")
        scheme_option = st.sidebar.radio(
            "Choose scheme(s):",
            ['Revenue Share', 'Rental', 'Both'],
            index=0,
            help="Select which scheme(s) to analyze"
        )
        
        # Tenure filter - Dropdown for tenure filtering
        st.sidebar.markdown("### 📊 Tenure Filter")
        tenure_filter = st.sidebar.selectbox(
            "Select tenure range:",
            ['Both', 'Tenure more than 15 days', 'Tenure less than 15 days'],
            index=1,  # Default to 'Tenure more than 15 days'
            help="Filter drivers based on tenure"
        )
        
        # Display options
        st.sidebar.markdown("### 🎨 Display Options")
        show_targets = st.sidebar.checkbox("🎯 Show Target Lines", value=True)
        point_size = st.sidebar.slider("🔘 Point Size", 5, 15, 8)
        
        # Data view options
        st.sidebar.markdown("### 📋 Data View")
        if st.sidebar.button("🔄 Toggle Raw Data View"):
            st.session_state.show_raw_data = not st.session_state.show_raw_data
            st.rerun()
        
        return {
            'selected_month': selected_month,
            'scheme_option': scheme_option,
            'tenure_filter': tenure_filter,
            'show_targets': show_targets,
            'point_size': point_size,
        }
    
    def display_statistics(self, data, title, analysis_type, month, tenure_filter):
        """Display statistics for a dataset"""
        try:
            if data is None or len(data) == 0:
                st.warning(f"No valid data for statistics in {title} for {month}")
                return
            
            # Get PMV ID column
            pmv_id_col = self.get_pmv_id_column(data)
            if not pmv_id_col:
                st.error(f"PMV ID column not found in {title}")
                return
            
            # Get aggregated data for unique PMV IDs
            aggregated_data = self.get_aggregated_stats(data, pmv_id_col)
            if aggregated_data is None or len(aggregated_data) == 0:
                st.warning(f"No valid data for statistics in {title} for {month}")
                return
            
            # Clean data based on analysis type
            if analysis_type == 'Tenure vs Net Earnings':
                clean_data = aggregated_data.dropna(subset=['Tenure(Days)', 'Net Earnings (Toll - Tip)'])
            else:
                clean_data = aggregated_data.dropna(subset=['Online Hours Per Day', 'Net Earnings (Toll - Tip)'])
            
            # Display month-graph header
            st.markdown(f'<div class="month-graph-header">{month} - {title} - {analysis_type} ({tenure_filter})</div>', unsafe_allow_html=True)
            
            # Create metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_drivers = len(clean_data)
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🚗 Unique PMV IDs", f"{unique_drivers:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                avg_earnings = clean_data['Net Earnings (Toll - Tip)'].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("💰 Avg Earnings", format_currency(avg_earnings))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                avg_tenure = clean_data['Tenure(Days)'].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} days")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                if 'Online Hours Per Day' in clean_data.columns:
                    avg_hours_per_day = clean_data['Online Hours Per Day'].mean()
                    st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("⏰ Avg Hours/Day", f"{avg_hours_per_day:.1f} hrs")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Target achievement analysis
            st.markdown("### 🎯 Target Achievement Analysis")
            target_col1, target_col2, target_col3 = st.columns(3)
            
            with target_col1:
                revenue_achievers = len(clean_data[clean_data['Net Earnings (Toll - Tip)'] >= self.targets['revenue']])
                revenue_pct = (revenue_achievers / len(clean_data) * 100) if len(clean_data) > 0 else 0
                st.metric("💰 Revenue Target Achievers", f"{revenue_achievers} ({revenue_pct:.1f}%)")
            
            with target_col2:
                tenure_achievers = len(clean_data[clean_data['Tenure(Days)'] >= self.targets['tenure']])
                tenure_pct = (tenure_achievers / len(clean_data) * 100) if len(clean_data) > 0 else 0
                st.metric("📅 Tenure Target Achievers", f"{tenure_achievers} ({tenure_pct:.1f}%)")
            
            with target_col3:
                if 'Online Hours Per Day' in clean_data.columns:
                    hours_achievers = len(clean_data[clean_data['Online Hours Per Day'] >= self.targets['hours_per_day']])
                    hours_pct = (hours_achievers / len(clean_data) * 100) if len(clean_data) > 0 else 0
                    st.metric("⏰ Hours/Day Target Achievers", f"{hours_achievers} ({hours_pct:.1f}%)")
            
        except Exception as e:
            st.error(f"Error displaying statistics for {title}: {str(e)}")
    
    def display_interactive_plots(self, controls, analysis_type):
        """Display interactive plots"""
        
        # Determine which schemes to include
        if controls['scheme_option'] == 'Both':
            selected_schemes = ['Revenue Share', 'Rental']
            title = 'Both Schemes'
        else:
            selected_schemes = [controls['scheme_option']]
            title = controls['scheme_option']
        
        # Combine data from selected schemes
        combined_data = self.combine_schemes_data(
            selected_schemes, 
            controls['selected_month'], 
            controls['tenure_filter']
        )
        
        if combined_data is None or len(combined_data) == 0:
            st.warning(f"⚠️ No data available for {title} in {controls['selected_month']} with {controls['tenure_filter']}.")
            return
        
        # Interactive features guide
        with st.expander("💡 How to Use", expanded=False):
            st.markdown(f"""
            **Interactive Features for {analysis_type}:**
            * **Hover** over points to see detailed driver information
            * **Zoom** and **pan** on charts for better visibility
            * **Target Lines** show performance benchmarks:
              - Red dashed line: Revenue target (₹{self.targets['revenue']:,})
              - Green dashed line: {'Tenure target (' + str(self.targets['tenure']) + ' days)' if analysis_type == 'Tenure vs Net Earnings' else 'Hours/Day target (' + str(self.targets['hours_per_day']) + ' hrs)'}
            * **Each point represents a unique PMV ID** (driver)
            * **Values are SUMMED** for each driver across all weeks in the month
            """)
        
        # Display statistics
        self.display_statistics(
            combined_data, 
            title, 
            analysis_type, 
            controls['selected_month'],
            controls['tenure_filter']
        )
        
        # Create and display plot
        fig = self.create_interactive_scatter_plot(
            combined_data, 
            title,
            analysis_type,
            controls['selected_month'],
            controls['tenure_filter'],
            controls['show_targets'],
            controls['point_size']
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not create plot for {title} in {controls['selected_month']}")

# Main App
def main():
    try:
        # App header
        st.markdown('<div class="main-header">🚗 Driver Performance Dashboard </div>', unsafe_allow_html=True)
        
        # Load data silently without showing status
        with st.spinner("Loading data..."):
            data_dict = load_driver_data()
        
        if not data_dict:
            st.error("❌ No data available. Please check the file path.")
            st.stop()
        
        # Show data summary
        st.success(f"✅ Data loaded successfully! Found {len(data_dict)} datasets.")
        
        # Initialize dashboard
        dashboard = DriverDashboard(data_dict)
        dashboard.initialize_session_state()
        
        # Analysis type selection
        analysis_type = dashboard.analysis_type_selector()
        
        # Sidebar controls
        controls = dashboard.sidebar_controls()
        
        # Main plots
        dashboard.display_interactive_plots(controls, analysis_type)
        
        # Raw data view
        if st.session_state.get('show_raw_data', False):
            st.markdown('<div class="section-header">📋 Raw Data (Monthly Totals by PMV ID)</div>', unsafe_allow_html=True)
            
            st.info(f"📊 Currently viewing: **{controls['scheme_option']}** data for **{controls['selected_month']}** - **{analysis_type}** ({controls['tenure_filter']})")
            st.info("🚗 **Note:** Data shows monthly totals by unique PMV ID (Trips, Earnings, Hours are SUMMED). Online Hours Per Day calculated from weekly hours ÷ 7")
            
            # Determine which schemes to include
            if controls['scheme_option'] == 'Both':
                selected_schemes = ['Revenue Share', 'Rental']
            else:
                selected_schemes = [controls['scheme_option']]
            
            # Combine and display data
            combined_data = dashboard.combine_schemes_data(
                selected_schemes, 
                controls['selected_month'], 
                controls['tenure_filter']
            )
            
            if combined_data is not None and len(combined_data) > 0:
                # Get aggregated data
                pmv_id_col = dashboard.get_pmv_id_column(combined_data)
                if pmv_id_col:
                    aggregated_data = dashboard.get_aggregated_stats(combined_data, pmv_id_col)
                    if aggregated_data is not None:
                        # Reorder columns for better display
                        display_columns = [pmv_id_col, 'Name', 'Scheme', 'Month', 'Tenure(Days)', 
                                         'Net Earnings (Toll - Tip)', 'Online Hours', 'Online Hours Per Day', 
                                         'Trips', 'Acceptance Rate', 'VBI']
                        available_display_columns = [col for col in display_columns if col in aggregated_data.columns]
                        
                        st.dataframe(aggregated_data[available_display_columns], use_container_width=True)
                        
                        # Download button
                        csv_data = aggregated_data.to_csv(index=False)
                        file_suffix = f"{controls['scheme_option'].lower().replace(' ', '_')}_{controls['selected_month'].lower()}_{controls['tenure_filter'].lower().replace(' ', '_')}"
                        st.download_button(
                            label=f"📥 Download {controls['scheme_option']} - {controls['selected_month']} - {controls['tenure_filter']} Data",
                            data=csv_data,
                            file_name=f"{file_suffix}_monthly_totals.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Could not aggregate data by PMV ID")
                else:
                    st.error("PMV ID column not found")
            else:
                st.info(f"No data available for {controls['scheme_option']} in {controls['selected_month']} with {controls['tenure_filter']}")
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please check the error details and data format.")

if __name__ == "__main__":
    main()