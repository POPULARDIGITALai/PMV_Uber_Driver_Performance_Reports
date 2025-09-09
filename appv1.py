import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings

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
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def format_currency(value):
    """Format currency values"""
    if pd.isna(value):
        return "₹0"
    return f"₹{value:,.2f}"

# Data loading functionus
@st.cache_data
def load_driver_data():
    """Load driver data from Excel file"""
    file_path = r"data\DriverMIS_Data_22Aug.xlsx"
    
    data_dict = {}
    
    try:
        # Load Revenue Share data
        try:
            revenue_df = pd.read_excel(file_path, sheet_name='Revenue Share')
            data_dict['Revenue Share'] = revenue_df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Revenue Share sheet not found: {str(e)}")
        
        # Load Rental data
        try:
            rental_df = pd.read_excel(file_path, sheet_name='Rental')
            data_dict['Rental'] = rental_df
        except Exception as e:
            st.sidebar.warning(f"⚠️ Rental sheet not found: {str(e)}")
        
        if not data_dict:
            # Create sample data if file not found
            st.sidebar.error("❌ Excel file not found. Using sample data...")
            data_dict = create_sample_data()
        
        return data_dict
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")
        st.sidebar.info("📝 Using sample data instead...")
        return create_sample_data()

@st.cache_data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Sample Revenue Share data
    revenue_data = {
        'PMV ID': [f'REV_{i:04d}' for i in range(1, 501)],
        'Driver Id': [f'REV_{i:04d}' for i in range(1, 501)],  # Keep for compatibility
        'Name': [f'Driver_{i}' for i in range(1, 501)],
        'Tenure(Days)': np.random.randint(30, 730, 500),
        'Net Earnings (Toll - Tip)': np.random.normal(25000, 8000, 500),
        'Online Hours': np.random.normal(8, 2, 500),
        'Trips': np.random.randint(15, 45, 500),
        'Month': np.random.choice(['June', 'July', 'August'], 500),
        'Acceptance Rate': np.random.uniform(70, 95, 500),
        'VBI': np.random.uniform(3.5, 4.8, 500),
        'Scheme': 'Revenue Share'
    }
    
    # Sample Rental data
    rental_data = {
        'PMV ID': [f'REN_{i:04d}' for i in range(1, 351)],
        'Driver Id': [f'REN_{i:04d}' for i in range(1, 351)],  # Keep for compatibility
        'Name': [f'Driver_{i+500}' for i in range(1, 351)],
        'Tenure(Days)': np.random.randint(45, 600, 350),
        'Net Earnings (Toll - Tip)': np.random.normal(22000, 6000, 350),
        'Online Hours': np.random.normal(7.5, 1.8, 350),
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
    
    def get_y_axis_info(self, analysis_type):
        """Get Y-axis column name and formatting based on analysis type"""
        if analysis_type == 'Tenure vs Net Earnings':
            return {
                'column': 'Net Earnings (Toll - Tip)',
                'label': 'Net Earnings (₹)',
                'format': ':,.2f',
                'metric_name': 'Avg Earnings',
                'format_func': format_currency,
                'display_name': 'Total Online Earnings'
            }
        elif analysis_type == 'Tenure vs Online Hours':
            return {
                'column': 'Online Hours',
                'label': 'Online Hours',
                'format': ':.2f',
                'metric_name': 'Avg Hours',
                'format_func': lambda x: f"{x:.2f} hrs",
                'display_name': 'Total Online Hours'
            }
    
    def normalize_month_data(self, df):
        """Normalize month data to handle different formats"""
        if 'Month' not in df.columns:
            return df
        
        df = df.copy()
        
        # Convert datetime to month names if needed
        if pd.api.types.is_datetime64_any_dtype(df['Month']):
            df['Month'] = df['Month'].dt.strftime('%B')  # Convert to full month name
        else:
            # Convert to string if not already
            df['Month'] = df['Month'].astype(str)
        
        # Clean up any NaN or invalid entries
        df = df[df['Month'].notna() & (df['Month'] != 'nan') & (df['Month'] != 'NaT')]
        
        return df
    
    def get_aggregated_stats(self, data, pmv_id_col):
        """Get aggregated statistics for unique PMV IDs - SUM values for the same driver in the same month"""
        if data is None or len(data) == 0:
            return None
        
        # Group by PMV ID and calculate SUMS for performance metrics and MEAN for driver attributes
        agg_dict = {
            # SUM these performance metrics (what the driver achieved in the month)
            'Net Earnings (Toll - Tip)': 'sum',  # Total earnings in the month
            'Trips': 'sum',  # Total trips in the month
            'Online Hours': 'sum',  # Total hours in the month
            
            # MEAN/FIRST for driver attributes (these should be consistent)
            'Tenure(Days)': 'mean',  # Average tenure (should be similar across weeks)
            'Acceptance Rate': 'mean',  # Average acceptance rate
            'VBI': 'mean',  # Average VBI
        }
        
        # Only include columns that exist in the data
        final_agg_dict = {}
        for col, func in agg_dict.items():
            if col in data.columns:
                final_agg_dict[col] = func
        
        # Group by PMV ID and aggregate
        aggregated_data = data.groupby(pmv_id_col).agg(final_agg_dict).reset_index()
        
        # Add back other identifying information (take first occurrence)
        id_columns = ['Month', 'Scheme', 'Name']
        for col in id_columns:
            if col in data.columns:
                first_occurrence = data.groupby(pmv_id_col)[col].first().reset_index()
                aggregated_data = aggregated_data.merge(first_occurrence, on=pmv_id_col)
        
        return aggregated_data
    
    def create_interactive_scatter_plot(self, data, title, scheme_color, analysis_type, month, show_trend=True, point_size=8):
        """Create an interactive scatter plot"""
        try:
            y_axis_info = self.get_y_axis_info(analysis_type)
            y_column = y_axis_info['column']
            y_label = y_axis_info['label']
            y_format = y_axis_info['format']
            
            # Get PMV ID column
            pmv_id_col = self.get_pmv_id_column(data)
            if not pmv_id_col:
                st.error(f"PMV ID column not found in {title}")
                return None
            
            # Ensure required columns exist
            if 'Tenure(Days)' not in data.columns or y_column not in data.columns:
                st.error(f"Required columns missing in {title}. Need 'Tenure(Days)' and '{y_column}'")
                st.write(f"Available columns: {list(data.columns)}")
                return None
            
            # Get aggregated data for unique PMV IDs
            aggregated_data = self.get_aggregated_stats(data, pmv_id_col)
            if aggregated_data is None or len(aggregated_data) == 0:
                st.warning(f"No valid data points for {title}")
                return None
            
            # Clean data
            clean_data = aggregated_data.dropna(subset=['Tenure(Days)', y_column])
            if len(clean_data) == 0:
                st.warning(f"No valid data points for {title}")
                return None
            
            # Create a copy for display with renamed columns for hover
            display_data = clean_data.copy()
            
            # Rename columns for display in hover
            column_mapping = {}
            if 'Net Earnings (Toll - Tip)' in display_data.columns:
                display_data['Total Online Earnings'] = display_data['Net Earnings (Toll - Tip)']
                column_mapping['Net Earnings (Toll - Tip)'] = 'Total Online Earnings'
            
            if 'Online Hours' in display_data.columns:
                display_data['Total Online Hours'] = display_data['Online Hours']
                column_mapping['Online Hours'] = 'Total Online Hours'
            
            if 'Trips' in display_data.columns:
                display_data['Total Trips'] = display_data['Trips']
                column_mapping['Trips'] = 'Total Trips'
            
            # Determine which column to use for Y-axis display
            if analysis_type == 'Tenure vs Net Earnings':
                y_display_column = 'Total Online Earnings'
                cross_ref_column = 'Total Online Hours'
            else:  # Tenure vs Online Hours
                y_display_column = 'Total Online Hours'
                cross_ref_column = 'Total Online Earnings'
            
            # Prepare hover data with display names
            hover_data = {
                y_display_column: y_format,
                'Tenure(Days)': True,
                pmv_id_col: True
            }
            
            # Add optional columns if they exist
            if 'Month' in display_data.columns:
                hover_data['Month'] = True
            if 'Name' in display_data.columns:
                hover_data['Name'] = True
            if 'Total Trips' in display_data.columns:
                hover_data['Total Trips'] = ':.0f'  # Show total trips as whole number
            
            # Add cross-reference metric
            if cross_ref_column in display_data.columns:
                cross_ref_format = ':.2f' if 'Hours' in cross_ref_column else ':,.2f'
                hover_data[cross_ref_column] = cross_ref_format
            
            # Create scatter plot with month in title
            plot_title = f'{month} - {analysis_type} (Unique PMV IDs: {len(clean_data)})'
            
            fig = px.scatter(
                display_data, 
                x='Tenure(Days)', 
                y=y_display_column,
                title=plot_title,
                labels={
                    'Tenure(Days)': 'Tenure (Days)',
                    y_display_column: y_label
                },
                color_discrete_sequence=[scheme_color],
                hover_data=hover_data
            )
            
            # Add trend line using original clean_data (not display_data)
            if show_trend and len(clean_data) > 1:
                x = clean_data['Tenure(Days)'].values
                y = clean_data[y_column].values
                
                # Calculate trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=p(x),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=True
                ))
            
            # Update layout
            fig.update_layout(
                height=500,
                hovermode='closest',
                title_x=0.5,
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            fig.update_traces(
                marker=dict(size=point_size, opacity=0.7, line=dict(width=1, color='white')),
                selector=dict(mode='markers')
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot for {title}: {str(e)}")
            return None
    
    def analysis_type_selector(self):
        """Analysis type selection section"""
        st.markdown('<div class="analysis-selector">', unsafe_allow_html=True)
        st.markdown("### 🎯 Choose Analysis Type")
        
        analysis_options = ['Tenure vs Net Earnings', 'Tenure vs Online Hours']
        
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
        
        # Month selection dropdown - get available months from data
        available_months = set()
        for scheme, df in self.data_dict.items():
            if df is not None and 'Month' in df.columns:
                normalized_df = self.normalize_month_data(df)
                months = normalized_df['Month'].unique()
                available_months.update([m for m in months if pd.notna(m) and m != 'nan'])
        
        if not available_months:
            available_months = ['June', 'July', 'August']  # Default fallback
        
        month_options = sorted(list(available_months))
        selected_month = st.sidebar.selectbox(
            "📅 Select Month",
            month_options,
            index=0,
            help="Choose a month to display"
        )
        
        # Scheme selection dropdown
        scheme_options = list(self.data_dict.keys())
        selected_scheme = st.sidebar.selectbox(
            "🎯 Select Scheme",
            scheme_options,
            index=0,
            help="Choose a scheme to analyze"
        )
        
        # Display options
        st.sidebar.markdown("### 🎨 Display Options")
        show_trend_line = st.sidebar.checkbox("📈 Show Trend Line", value=True)
        point_size = st.sidebar.slider("🔘 Point Size", 5, 15, 8)
        
        # Data view options
        st.sidebar.markdown("### 📋 Data View")
        if st.sidebar.button("🔄 Toggle Raw Data View"):
            st.session_state.show_raw_data = not st.session_state.show_raw_data
            st.rerun()
        
        return {
            'selected_month': selected_month,
            'selected_scheme': selected_scheme,
            'show_trend_line': show_trend_line,
            'point_size': point_size,
        }
    
    def get_filtered_data(self, selected_month, selected_scheme):
        """Get filtered data for specific month and scheme"""
        if selected_scheme not in self.data_dict:
            return None
        
        df = self.data_dict[selected_scheme].copy()
        
        if df is None or len(df) == 0:
            return None
        
        # Normalize month data
        df = self.normalize_month_data(df)
        
        if 'Month' in df.columns:
            # Filter by selected month
            filtered_data = df[df['Month'] == selected_month]
            
            if len(filtered_data) == 0:
                # If no exact match, try the first available month
                available_months = df['Month'].unique()
                if len(available_months) > 0:
                    fallback_month = available_months[0]
                    st.warning(f"No data found for '{selected_month}'. Showing data for '{fallback_month}' instead.")
                    return df[df['Month'] == fallback_month]
                else:
                    return None
            
            return filtered_data
        else:
            # Return all data if no month column
            return df if len(df) > 0 else None
    
    def display_statistics(self, data, title, analysis_type, month):
        """Display statistics for a dataset with proper PMV ID counting"""
        try:
            y_axis_info = self.get_y_axis_info(analysis_type)
            y_column = y_axis_info['column']
            
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
            
            clean_data = aggregated_data.dropna(subset=['Tenure(Days)', y_column])
            
            # Display month-graph header
            st.markdown(f'<div class="month-graph-header">{month} - {title} - {analysis_type}</div>', unsafe_allow_html=True)
            
            # Create metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_drivers = len(clean_data)
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🚗 Unique PMV IDs", f"{unique_drivers:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                avg_y = clean_data[y_column].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric(f"💰 {y_axis_info['metric_name']}", y_axis_info['format_func'](avg_y))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                avg_tenure = clean_data['Tenure(Days)'].mean()
                st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} days")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                # Additional metric based on analysis type
                if analysis_type == 'Tenure vs Net Earnings' and 'Online Hours' in clean_data.columns:
                    avg_hours = clean_data['Online Hours'].mean()
                    st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("⏰ Avg Online Hours", f"{avg_hours:.1f} hrs")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif analysis_type == 'Tenure vs Online Hours' and 'Net Earnings (Toll - Tip)' in clean_data.columns:
                    avg_earnings = clean_data['Net Earnings (Toll - Tip)'].mean()
                    st.markdown(f'<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("💰 Avg Earnings", format_currency(avg_earnings))
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional summary statistics
            with st.expander(f"📊 Detailed Statistics for {title} - {month}", expanded=False):
                summary_stats = clean_data[['Tenure(Days)', y_column]].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
                # Show aggregation verification for the first PMV ID
                st.markdown("#### 🔍 Aggregation Verification (SUM Method)")
                st.markdown("*Showing how data is SUMMED for the first PMV ID in this month:*")
                
                # Get original data for verification
                first_pmv = clean_data[pmv_id_col].iloc[0]
                # Filter original data by both PMV ID and Month
                original_records = data[
                    (data[pmv_id_col] == first_pmv) & 
                    (data['Month'] == month)
                ]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📋 Original Weekly Records for {first_pmv} in {month}:**")
                    display_cols = [pmv_id_col, 'Month', 'Trips', 'Net Earnings (Toll - Tip)', 'Online Hours', 'Tenure(Days)']
                    # Only add Week Begin if it exists in the data
                    if 'Week Begin' in original_records.columns:
                        display_cols.insert(2, 'Week Begin')
                    available_cols = [col for col in display_cols if col in original_records.columns]
                    st.dataframe(original_records[available_cols], use_container_width=True)
                
                with col2:
                    st.markdown(f"**📊 Monthly Total for {first_pmv} in {month}:**")
                    aggregated_result = clean_data[clean_data[pmv_id_col] == first_pmv]
                    available_cols = [col for col in display_cols if col in aggregated_result.columns]
                    st.dataframe(aggregated_result[available_cols], use_container_width=True)
                    
                    # Show the math for SUMMED values
                    st.markdown("**🔢 Monthly Totals Calculation:**")
                    
                    if 'Trips' in original_records.columns:
                        total_trips = original_records['Trips'].sum()
                        trips_breakdown = ' + '.join(map(str, original_records['Trips'].values))
                        st.write(f"✅ **Total Trips**: {trips_breakdown} = **{total_trips}**")
                    
                    if 'Net Earnings (Toll - Tip)' in original_records.columns:
                        total_earnings = original_records['Net Earnings (Toll - Tip)'].sum()
                        earnings_breakdown = ' + '.join([f'₹{x:,.2f}' for x in original_records['Net Earnings (Toll - Tip)'].values])
                        st.write(f"✅ **Total Earnings**: {earnings_breakdown} = **₹{total_earnings:,.2f}**")
                    
                    if 'Online Hours' in original_records.columns:
                        total_hours = original_records['Online Hours'].sum()
                        hours_breakdown = ' + '.join([f'{x:.2f}' for x in original_records['Online Hours'].values])
                        st.write(f"✅ **Total Hours**: {hours_breakdown} = **{total_hours:.2f} hrs**")
                    
                    if 'Tenure(Days)' in original_records.columns:
                        avg_tenure = original_records['Tenure(Days)'].mean()
                        tenure_breakdown = ' + '.join([f'{x:.1f}' for x in original_records['Tenure(Days)'].values])
                        st.write(f"📅 **Avg Tenure**: ({tenure_breakdown}) ÷ {len(original_records)} = **{avg_tenure:.1f} days**")
                
                # Show PMV ID distribution
                st.markdown("#### 📋 PMV ID Sample (Monthly Totals)")
                sample_pmvs = clean_data[[pmv_id_col, 'Tenure(Days)', y_column]].head(10)
                st.dataframe(sample_pmvs, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying statistics for {title}: {str(e)}")
    
    def display_interactive_plots(self, controls, analysis_type):
        """Display interactive plots"""
        
        filtered_data = self.get_filtered_data(controls['selected_month'], controls['selected_scheme'])
        
        if filtered_data is None or len(filtered_data) == 0:
            st.warning(f"⚠️ No data available for {controls['selected_scheme']} in {controls['selected_month']}.")
            return
        
        # Color scheme
        scheme_colors = {
            'Revenue Share': '#1f77b4',
            'Rental': '#ff7f0e'
        }
        
        # Interactive features guide
        with st.expander("💡 How to Use", expanded=False):
            st.markdown(f"""
            **Interactive Features for {analysis_type}:**
            * **Hover** over points to see detailed driver information
            * **Zoom** and **pan** on charts for better visibility
            * **Use dropdowns** in sidebar to select month and scheme
            * **Switch analysis type** using radio buttons above
            * **Each point represents a unique PMV ID** (driver)
            * **Values are SUMMED** for each driver across all weeks in the month
            * **Performance metrics** (Trips, Earnings, Hours) are totaled per month
            * **Driver attributes** (Tenure, Acceptance Rate, VBI) are averaged
            """)
        
        # Display statistics
        self.display_statistics(
            filtered_data, 
            controls['selected_scheme'], 
            analysis_type, 
            controls['selected_month']
        )
        
        # Create and display plot
        fig = self.create_interactive_scatter_plot(
            filtered_data, 
            controls['selected_scheme'],
            scheme_colors.get(controls['selected_scheme'], '#1f77b4'),
            analysis_type,
            controls['selected_month'],
            controls['show_trend_line'],
            controls['point_size']
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not create plot for {controls['selected_scheme']} in {controls['selected_month']}")

# Main App
def main():
    try:
        # App header
        st.markdown('<div class="main-header">🚗 Driver Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Load data
        st.sidebar.markdown("### 📊 Data Loading Status")
        
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
            
            st.info(f"📊 Currently viewing: **{controls['selected_scheme']}** data for **{controls['selected_month']}** - **{analysis_type}**")
            st.info("🚗 **Note:** Data shows monthly totals by unique PMV ID (Trips, Earnings, Hours are SUMMED)")
            
            filtered_data = dashboard.get_filtered_data(controls['selected_month'], controls['selected_scheme'])
            
            if filtered_data is not None and len(filtered_data) > 0:
                # Get aggregated data
                pmv_id_col = dashboard.get_pmv_id_column(filtered_data)
                if pmv_id_col:
                    aggregated_data = dashboard.get_aggregated_stats(filtered_data, pmv_id_col)
                    if aggregated_data is not None:
                        st.dataframe(aggregated_data, use_container_width=True)
                        
                        # Download button
                        csv_data = aggregated_data.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {controls['selected_scheme']} - {controls['selected_month']} Monthly Totals",
                            data=csv_data,
                            file_name=f"{controls['selected_scheme'].lower().replace(' ', '_')}_{controls['selected_month'].lower()}_monthly_totals.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Could not aggregate data by PMV ID")
                else:
                    st.error("PMV ID column not found")
            else:
                st.info(f"No data available for {controls['selected_scheme']} in {controls['selected_month']}")
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please check the error details and data format.")

if __name__ == "__main__":
    main()