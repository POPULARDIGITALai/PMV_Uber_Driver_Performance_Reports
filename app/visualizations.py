import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from utils import get_scheme_colors, calculate_correlation

class ScatterPlotGenerator:
    """Generates interactive scatter plots for driver performance analysis"""
    
    def __init__(self):
        self.scheme_colors = get_scheme_colors()
        
    def create_month_scatter_plot(self, month_data, month, controls):
        """Create scatter plot for a specific month with multiple schemes"""
        fig = go.Figure()
        
        for scheme, data in month_data:
            if len(data) == 0:
                continue
                
            # Add hover template with driver details
            hover_template = (
                "<b>%{customdata[0]}</b><br>" +
                "Scheme: " + scheme + "<br>" +
                "Tenure: %{x} days<br>" +
                "Earnings: ₹%{y:,.0f}<br>" +
                "Month: " + month + "<br>" +
                "Online Hours: %{customdata[1]}<br>" +
                "Trips: %{customdata[2]}<br>" +
                "<extra></extra>"
            )
            
            # Prepare custom data for hover
            customdata = []
            for _, row in data.iterrows():
                driver_name = row.get('Name', 'Unknown')
                online_hours = row.get('Online Hours', 0)
                trips = row.get('Trips', 0)
                customdata.append([driver_name, online_hours, trips])
            
            fig.add_trace(go.Scattergl(
                x=data['Tenure(Days)'],
                y=data['Net Earnings (Toll - Tip)'],
                mode='markers',
                name=f'{scheme} ({len(data)})',
                marker=dict(
                    color=self.scheme_colors.get(scheme, '#1f77b4'),
                    size=controls['point_size'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                customdata=customdata,
                hovertemplate=hover_template
            ))
            
            # Add trend line if requested
            if controls['show_trend_line'] and len(data) > 1:
                z = np.polyfit(data['Tenure(Days)'], data['Net Earnings (Toll - Tip)'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data['Tenure(Days)'].min(), data['Tenure(Days)'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scattergl(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name=f'{scheme} Trend',
                    line=dict(
                        color=self.scheme_colors.get(scheme, '#1f77b4'),
                        width=2,
                        dash='dash'
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f'{month} Performance Analysis - Tenure vs Net Earnings',
            xaxis_title='Tenure (Days)',
            yaxis_title='Net Earnings (₹)',
            template='plotly_white',
            width=1200,
            height=600,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add correlation annotation if requested
        if controls['show_correlation'] and len(month_data) > 0:
            correlations = []
            for scheme, data in month_data:
                if len(data) > 1:
                    corr = calculate_correlation(data['Tenure(Days)'], data['Net Earnings (Toll - Tip)'])
                    correlations.append(f'{scheme}: {corr:.3f}')
            
            if correlations:
                fig.add_annotation(
                    text='<br>'.join(['Correlations:'] + correlations),
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    showarrow=False,
                    align='left',
                    font=dict(size=12)
                )
        
        return fig
    
    def create_scheme_scatter_plot(self, data, scheme, controls):
        """Create scatter plot for a specific scheme with month-wise coloring"""
        fig = go.Figure()
        
        if 'Month' in data.columns:
            # Color by month
            months = data['Month'].unique()
            colors = px.colors.qualitative.Set1[:len(months)]
            
            for i, month in enumerate(months):
                month_data = data[data['Month'] == month]
                if len(month_data) == 0:
                    continue
                    
                # Prepare hover template
                hover_template = (
                    "<b>%{customdata[0]}</b><br>" +
                    f"Scheme: {scheme}<br>" +
                    "Month: " + month + "<br>" +
                    "Tenure: %{x} days<br>" +
                    "Earnings: ₹%{y:,.0f}<br>" +
                    "Online Hours: %{customdata[1]}<br>" +
                    "Trips: %{customdata[2]}<br>" +
                    "<extra></extra>"
                )
                
                # Prepare custom data for hover
                customdata = []
                for _, row in month_data.iterrows():
                    driver_name = row.get('Name', 'Unknown')
                    online_hours = row.get('Online Hours', 0)
                    trips = row.get('Trips', 0)
                    customdata.append([driver_name, online_hours, trips])
                
                fig.add_trace(go.Scattergl(
                    x=month_data['Tenure(Days)'],
                    y=month_data['Net Earnings (Toll - Tip)'],
                    mode='markers',
                    name=f'{month} ({len(month_data)})',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=controls['point_size'],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    customdata=customdata,
                    hovertemplate=hover_template
                ))
        else:
            # Single color if no month data
            hover_template = (
                "<b>%{customdata[0]}</b><br>" +
                f"Scheme: {scheme}<br>" +
                "Tenure: %{x} days<br>" +
                "Earnings: ₹%{y:,.0f}<br>" +
                "Online Hours: %{customdata[1]}<br>" +
                "Trips: %{customdata[2]}<br>" +
                "<extra></extra>"
            )
            
            customdata = []
            for _, row in data.iterrows():
                driver_name = row.get('Name', 'Unknown')
                online_hours = row.get('Online Hours', 0)
                trips = row.get('Trips', 0)
                customdata.append([driver_name, online_hours, trips])
            
            fig.add_trace(go.Scattergl(
                x=data['Tenure(Days)'],
                y=data['Net Earnings (Toll - Tip)'],
                mode='markers',
                name=f'{scheme} ({len(data)})',
                marker=dict(
                    color=self.scheme_colors.get(scheme, '#1f77b4'),
                    size=controls['point_size'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                customdata=customdata,
                hovertemplate=hover_template
            ))
        
        # Add trend line if requested
        if controls['show_trend_line'] and len(data) > 1:
            z = np.polyfit(data['Tenure(Days)'], data['Net Earnings (Toll - Tip)'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data['Tenure(Days)'].min(), data['Tenure(Days)'].max(), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scattergl(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Overall Trend',
                line=dict(
                    color='red',
                    width=3,
                    dash='dash'
                ),
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{scheme} Performance Analysis - Tenure vs Net Earnings',
            xaxis_title='Tenure (Days)',
            yaxis_title='Net Earnings (₹)',
            template='plotly_white',
            width=1200,
            height=600,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add correlation and statistics
        if controls['show_correlation'] and len(data) > 1:
            correlation = calculate_correlation(data['Tenure(Days)'], data['Net Earnings (Toll - Tip)'])
            avg_earnings = data['Net Earnings (Toll - Tip)'].mean()
            avg_tenure = data['Tenure(Days)'].mean()
            
            stats_text = (
                f'Statistics:<br>'
                f'Records: {len(data):,}<br>'
                f'Avg Earnings: ₹{avg_earnings:,.0f}<br>'
                f'Avg Tenure: {avg_tenure:.1f} days<br>'
                f'Correlation: {correlation:.3f}'
            )
            
            fig.add_annotation(
                text=stats_text,
                x=0.98, y=0.02,
                xref='paper', yref='paper',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1,
                showarrow=False,
                align='left',
                font=dict(size=12),
                xanchor='right',
                yanchor='bottom'
            )
        
        return fig
    
    def create_combined_scatter_plot(self, filtered_data, controls):
        """Create combined scatter plot with all schemes"""
        fig = go.Figure()
        
        for scheme, data in filtered_data.items():
            if len(data) == 0:
                continue
                
            hover_template = (
                "<b>%{customdata[0]}</b><br>" +
                f"Scheme: {scheme}<br>" +
                "Tenure: %{x} days<br>" +
                "Earnings: ₹%{y:,.0f}<br>" +
                "Month: %{customdata[1]}<br>" +
                "Online Hours: %{customdata[2]}<br>" +
                "Trips: %{customdata[3]}<br>" +
                "<extra></extra>"
            )
            
            # Prepare custom data
            customdata = []
            for _, row in data.iterrows():
                driver_name = row.get('Name', 'Unknown')
                month = row.get('Month', 'Unknown')
                online_hours = row.get('Online Hours', 0)
                trips = row.get('Trips', 0)
                customdata.append([driver_name, month, online_hours, trips])
            
            fig.add_trace(go.Scattergl(
                x=data['Tenure(Days)'],
                y=data['Net Earnings (Toll - Tip)'],
                mode='markers',
                name=f'{scheme} ({len(data)})',
                marker=dict(
                    color=self.scheme_colors.get(scheme, '#1f77b4'),
                    size=controls['point_size'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                customdata=customdata,
                hovertemplate=hover_template
            ))
            
            # Add trend line for each scheme if requested
            if controls['show_trend_line'] and len(data) > 1:
                z = np.polyfit(data['Tenure(Days)'], data['Net Earnings (Toll - Tip)'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data['Tenure(Days)'].min(), data['Tenure(Days)'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scattergl(
                    x=x_trend,
                    y=y_trend,
                    mode='lines',
                    name=f'{scheme} Trend',
                    line=dict(
                        color=self.scheme_colors.get(scheme, '#1f77b4'),
                        width=2,
                        dash='dash'
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title='Combined Schemes Performance Analysis - Tenure vs Net Earnings',
            xaxis_title='Tenure (Days)',
            yaxis_title='Net Earnings (₹)',
            template='plotly_white',
            width=1400,
            height=700,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add overall statistics if requested
        if controls['show_correlation'] and filtered_data:
            all_data = pd.concat(filtered_data.values())
            if len(all_data) > 1:
                overall_correlation = calculate_correlation(
                    all_data['Tenure(Days)'], 
                    all_data['Net Earnings (Toll - Tip)']
                )
                total_records = len(all_data)
                avg_earnings = all_data['Net Earnings (Toll - Tip)'].mean()
                
                stats_text = (
                    f'Overall Statistics:<br>'
                    f'Total Records: {total_records:,}<br>'
                    f'Avg Earnings: ₹{avg_earnings:,.0f}<br>'
                    f'Overall Correlation: {overall_correlation:.3f}'
                )
                
                fig.add_annotation(
                    text=stats_text,
                    x=0.98, y=0.02,
                    xref='paper', yref='paper',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    showarrow=False,
                    align='left',
                    font=dict(size=12),
                    xanchor='right',
                    yanchor='bottom'
                )
        
        return fig
    
    def create_tenure_vs_online_hours_plot(self, data, scheme, controls):
        """Create scatter plot for Tenure vs Online Hours"""
        if 'Online Hours' not in data.columns:
            return None
            
        fig = go.Figure()
        
        hover_template = (
            "<b>%{customdata[0]}</b><br>" +
            f"Scheme: {scheme}<br>" +
            "Tenure: %{x} days<br>" +
            "Online Hours: %{y}<br>" +
            "Earnings: ₹%{customdata[1]:,.0f}<br>" +
            "Month: %{customdata[2]}<br>" +
            "<extra></extra>"
        )
        
        customdata = []
        for _, row in data.iterrows():
            driver_name = row.get('Name', 'Unknown')
            earnings = row.get('Net Earnings (Toll - Tip)', 0)
            month = row.get('Month', 'Unknown')
            customdata.append([driver_name, earnings, month])
        
        fig.add_trace(go.Scattergl(
            x=data['Tenure(Days)'],
            y=data['Online Hours'],
            mode='markers',
            name=f'{scheme}',
            marker=dict(
                color=data['Net Earnings (Toll - Tip)'],
                colorscale='Viridis',
                size=controls['point_size'],
                opacity=0.7,
                line=dict(width=1, color='white'),
                colorbar=dict(title="Net Earnings (₹)")
            ),
            customdata=customdata,
            hovertemplate=hover_template
        ))
        
        # Add trend line if requested
        if controls['show_trend_line'] and len(data) > 1:
            z = np.polyfit(data['Tenure(Days)'], data['Online Hours'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data['Tenure(Days)'].min(), data['Tenure(Days)'].max(), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scattergl(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash'),
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'{scheme} - Tenure vs Online Hours (Color: Earnings)',
            xaxis_title='Tenure (Days)',
            yaxis_title='Total Online Hours',
            template='plotly_white',
            width=1200,
            height=600,
            hovermode='closest'
        )
        
        return fig