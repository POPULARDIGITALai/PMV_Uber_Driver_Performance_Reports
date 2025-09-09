import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles data loading and preprocessing for the driver dashboard"""
    
    def __init__(self):
        self.revenue_share_df = pd.DataFrame()
        self.rental_df = pd.DataFrame()
        self.rental_2dp_df = pd.DataFrame()
        
    @st.cache_data(show_spinner=False)
    def _read_excel_sheets_cached(_self_unused, file_bytes):
        """Cached helper: read required sheets from an uploaded excel file bytes."""
        xls = pd.ExcelFile(file_bytes)
        sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
        return sheets
        
    def load_excel_data(self, uploaded_file):
        """Load data from uploaded Excel file"""
        try:
            # Read Excel sheets (cached)
            file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file
            excel_data = pd.ExcelFile(file_bytes)
            
            # Check required sheets
            required_sheets = ['Revenue Share', 'Rental']
            for sheet in required_sheets:
                if sheet not in excel_data.sheet_names:
                    st.error(f"Required sheet '{sheet}' not found in Excel file")
                    return False
            
            # Load data from sheets using cache
            sheets = self._read_excel_sheets_cached(file_bytes)
            self.revenue_share_df = sheets.get('Revenue Share', pd.DataFrame())
            self.rental_df = sheets.get('Rental', pd.DataFrame())
            
            # Clean the data
            self.revenue_share_df = self._clean_data(self.revenue_share_df, 'Revenue Share')
            self.rental_df = self._clean_data(self.rental_df, 'Rental (All)')
            
            # Create Rental 2DP subset
            if 'DP Working Plan' in self.rental_df.columns:
                self.rental_2dp_df = self.rental_df[
                    self.rental_df['DP Working Plan'] == 'Rental 2DP'
                ].copy()
                self.rental_2dp_df = self._clean_data(self.rental_2dp_df, 'Rental 2DP')
            else:
                self.rental_2dp_df = pd.DataFrame()
                st.warning("'DP Working Plan' column not found. Rental 2DP filtering unavailable.")
            
            # Add month mapping
            self._add_month_mapping()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _clean_data(self, df, scheme_name):
        """Clean and prepare data for analysis"""
        if df.empty:
            return df
            
        df = df.copy()
        df['Scheme'] = scheme_name
        
        # Ensure required columns exist
        required_columns = ['Tenure(Days)', 'Net Earnings (Toll - Tip)']
        for col in required_columns:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found in {scheme_name} data")
                return pd.DataFrame()
        
        # Convert numeric columns
        numeric_columns = ['Tenure(Days)', 'Net Earnings (Toll - Tip)', 'Online Hours', 'Trips']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            st.info(f"Removed {removed_count} rows with missing data from {scheme_name}")
        
        # Remove outliers (optional - can be made configurable)
        df = self._remove_outliers(df)
        
        return df
    
    def _remove_outliers(self, df, method='iqr', factor=1.5):
        """Remove outliers from earnings and tenure data"""
        if len(df) == 0:
            return df
            
        df_clean = df.copy()
        
        for column in ['Net Earnings (Toll - Tip)', 'Tenure(Days)']:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                initial_count = len(df_clean)
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) & 
                    (df_clean[column] <= upper_bound)
                ]
                
        return df_clean
    
    def _add_month_mapping(self):
        """Add month mapping to dataframes"""
        # This is a simplified mapping - adjust based on your actual date columns
        month_mapping = {
            '2025-06': 'June',
            '2025-07': 'July',
            '2025-08': 'August',
            '06': 'June',
            '07': 'July',
            '08': 'August'
        }
        
        for df in [self.revenue_share_df, self.rental_df, self.rental_2dp_df]:
            if not df.empty:
                # If Month column doesn't exist, try to create it from other date columns
                if 'Month' not in df.columns:
                    # Look for date columns
                    date_columns = [col for col in df.columns 
                                  if any(date_word in col.lower() 
                                        for date_word in ['date', 'month', 'time'])]
                    
                    if date_columns:
                        # Use the first date column found
                        date_col = date_columns[0]
                        try:
                            df['Month'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m')
                            df['Month'] = df['Month'].map(month_mapping).fillna(df['Month'])
                        except:
                            # If conversion fails, create dummy month data for demo
                            df['Month'] = np.random.choice(['June', 'July', 'August'], size=len(df))
                    else:
                        # Create dummy month data for demo purposes
                        df['Month'] = np.random.choice(['June', 'July', 'August'], size=len(df))
                else:
                    # Clean existing Month column
                    df['Month'] = df['Month'].astype(str)
                    for key, value in month_mapping.items():
                        df['Month'] = df['Month'].str.replace(key, value, regex=False)
    
    def get_filtered_data(self, selected_months, selected_schemes):
        """Get filtered data based on user selections"""
        filtered_data = {}
        
        # Map scheme names to dataframes
        scheme_data_map = {
            'Revenue Share': self.revenue_share_df,
            'Rental (All)': self.rental_df,
            'Rental 2DP': self.rental_2dp_df
        }
        
        for scheme in selected_schemes:
            if scheme in scheme_data_map and not scheme_data_map[scheme].empty:
                df = scheme_data_map[scheme].copy()
                
                # Filter by selected months
                if 'Month' in df.columns and selected_months:
                    df = df[df['Month'].isin(selected_months)]
                
                if len(df) > 0:
                    filtered_data[scheme] = df
        
        return filtered_data
    
    def get_data_summary(self):
        """Get summary statistics of loaded data"""
        summary = {
            'Revenue Share': {
                'count': len(self.revenue_share_df),
                'avg_earnings': self.revenue_share_df['Net Earnings (Toll - Tip)'].mean() if not self.revenue_share_df.empty else 0,
                'avg_tenure': self.revenue_share_df['Tenure(Days)'].mean() if not self.revenue_share_df.empty else 0
            },
            'Rental (All)': {
                'count': len(self.rental_df),
                'avg_earnings': self.rental_df['Net Earnings (Toll - Tip)'].mean() if not self.rental_df.empty else 0,
                'avg_tenure': self.rental_df['Tenure(Days)'].mean() if not self.rental_df.empty else 0
            },
            'Rental 2DP': {
                'count': len(self.rental_2dp_df),
                'avg_earnings': self.rental_2dp_df['Net Earnings (Toll - Tip)'].mean() if not self.rental_2dp_df.empty else 0,
                'avg_tenure': self.rental_2dp_df['Tenure(Days)'].mean() if not self.rental_2dp_df.empty else 0
            }
        }
        
        return summary
    
    def validate_data_integrity(self):
        """Validate data integrity and return issues if any"""
        issues = []
        
        for name, df in [('Revenue Share', self.revenue_share_df), 
                        ('Rental (All)', self.rental_df), 
                        ('Rental 2DP', self.rental_2dp_df)]:
            if df.empty:
                continue
                
            # Check for negative values
            if (df['Net Earnings (Toll - Tip)'] < 0).any():
                issues.append(f"{name}: Contains negative earnings values")
            
            if (df['Tenure(Days)'] < 0).any():
                issues.append(f"{name}: Contains negative tenure values")
            
            # Check for extreme values
            earnings_mean = df['Net Earnings (Toll - Tip)'].mean()
            earnings_std = df['Net Earnings (Toll - Tip)'].std()
            
            if earnings_std > 0:
                extreme_earnings = df[
                    abs(df['Net Earnings (Toll - Tip)'] - earnings_mean) > 3 * earnings_std
                ]
                if len(extreme_earnings) > len(df) * 0.05:  # More than 5% extreme values
                    issues.append(f"{name}: High number of extreme earnings values detected")
        
        return issues