#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# In[2]:


# Set page configuration
st.set_page_config(
    page_title="African Fiscal Sustainability Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# In[3]:


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("10Alytics Hackathon- Fiscal Data (1).csv", encoding='latin1')

    # Convert Time to datetime and extract features
    df['Time'] = pd.to_datetime(df['Time'])
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Quarter'] = df['Time'].dt.quarter

    # Handle missing values for Amount, Country, and Indicator to ensure correct types
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Country'] = df['Country'].fillna('Unknown').astype(str)
    df['Indicator'] = df['Indicator'].fillna('Unknown').astype(str) # Add this line to handle Indicator column

    return df


# In[4]:


# Advanced analytical functions
def detect_anomalies(df):
    """Detect fiscal anomalies using Isolation Forest"""
    if len(df) < 10:
        return df.assign(Anomaly=0, Anomaly_Score=0)

    features = df[['Amount']].dropna()
    if len(features) < 10:
        return df.assign(Anomaly=0, Anomaly_Score=0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(features_scaled)
    scores = iso_forest.decision_function(features_scaled)

    result = df.copy()
    result['Anomaly'] = 0
    result['Anomaly_Score'] = 0

    result.loc[features.index, 'Anomaly'] = anomalies
    result.loc[features.index, 'Anomaly_Score'] = scores

    return result


# In[5]:


def calculate_fiscal_sustainability(df):
    """Calculate fiscal sustainability metrics"""
    sustainability_metrics = []

    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Time')

        if len(country_data) < 3:
            continue

        # Trend analysis
        amounts = country_data['Amount'].values
        time_index = np.arange(len(amounts))

        # Linear trend
        trend_coef = np.polyfit(time_index, amounts, 1)[0]

        # Volatility (standard deviation)
        volatility = np.std(amounts)

        # Deficit persistence (percentage of periods with deficit)
        deficit_percentage = (amounts < 0).mean() * 100

        # Recent trend (last 3 years vs previous 3 years)
        if len(amounts) >= 6:
            recent_avg = np.mean(amounts[-3:])
            previous_avg = np.mean(amounts[-6:-3])
            trend_direction = "Improving" if recent_avg > previous_avg else "Worsening"
        else:
            trend_direction = "Insufficient Data"

        sustainability_metrics.append({
            'Country': country,
            'Trend_Coefficient': trend_coef,
            'Volatility': volatility,
            'Deficit_Percentage': deficit_percentage,
            'Trend_Direction': trend_direction,
            'Data_Points': len(country_data)
        })

    return pd.DataFrame(sustainability_metrics)


# In[6]:


def predict_fiscal_stress(df, horizon=3):
    """Simple predictive model for fiscal stress"""
    temp_predictions = []

    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Time')

        if len(country_data) < 5:
            continue

        amounts = country_data['Amount'].values

        # Ensure enough data points for moving average prediction
        if len(amounts) < 3:
            continue
        last_values = amounts[-3:]  # Last 3 periods
        predicted = np.mean(last_values)  # Simple average prediction

        # Ensure enough data points for polyfit
        if len(amounts) < 2:
            continue
        # Risk assessment based on trends
        trend = np.polyfit(np.arange(len(amounts)), amounts, 1)[0]
        volatility = np.std(amounts)

        # Risk score (higher = more risky)
        risk_score = abs(trend) * volatility * (1 if trend < 0 else 0.5)

        temp_predictions.append({
            'Country': country,
            'Predicted_Deficit': predicted,
            'Risk_Score': risk_score,
            'Confidence': min(95, len(country_data) * 5)  # Confidence based on data points
        })

    if not temp_predictions:
        # Return an empty DataFrame with the expected columns if no predictions were made
        return pd.DataFrame(columns=['Country', 'Predicted_Deficit', 'Risk_Score', 'Risk_Level', 'Confidence'])

    predictions_df = pd.DataFrame(temp_predictions)

    # Calculate Risk_Level after all risk scores are gathered
    if not predictions_df.empty:
        q75 = predictions_df['Risk_Score'].quantile(0.75)
        q25 = predictions_df['Risk_Score'].quantile(0.25)

        def assign_risk_level(score):
            if score > q75:
                return 'High'
            elif score > q25:
                return 'Medium'
            else:
                return 'Low'

        predictions_df['Risk_Level'] = predictions_df['Risk_Score'].apply(assign_risk_level)
    else:
        predictions_df['Risk_Level'] = 'Low' # Default or handle as appropriate for empty scores

    return predictions_df


# In[7]:


def main():
    st.title("🌍 African Fiscal Sustainability & Risk Analytics Dashboard")
    st.markdown("""
    ### **Primary Objective**: Identify patterns, detect risks, and develop strategies for sustainable financial governance
    *Analyzing long-term trends, drivers of fiscal imbalance, and emerging risks to support evidence-based policymaking*
    """)

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Dashboard Filters")

    countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        countries,
        default=["Egypt", "Nigeria", "South Africa", "Ghana", "Ethiopia"]
    )

    indicators = sorted(df['Indicator'].unique())
    selected_indicator = st.sidebar.selectbox(
        "Select Fiscal Indicator",
        indicators
    )

    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider(
        "Analysis Period",
        min_year, max_year,
        (max(min_year, max_year-10), max_year)  # Default last 10 years
    )

    # Filter data
    filtered_df = df[
        (df['Country'].isin(selected_countries)) &
        (df['Indicator'] == selected_indicator) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ]

    # Enhanced analytics
    sustainability_df = calculate_fiscal_sustainability(filtered_df)
    predictions_df = predict_fiscal_stress(filtered_df)
    anomaly_df = detect_anomalies(filtered_df)

    # Key Metrics aligned with objectives
    st.header("📊 Executive Summary - Fiscal Health Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_risk_count = len(predictions_df[predictions_df['Risk_Level'] == 'High'])
        st.metric("High Risk Countries", high_risk_count,
                 delta=f"{high_risk_count} requiring immediate attention")

    with col2:
        avg_deficit_pct = sustainability_df['Deficit_Percentage'].mean()
        st.metric("Average Deficit Frequency", f"{avg_deficit_pct:.1f}%")

    with col3:
        anomalies_count = len(anomaly_df[anomaly_df['Anomaly'] == -1])
        st.metric("Fiscal Anomalies Detected", anomalies_count)

    with col4:
        worsening_trends = len(sustainability_df[sustainability_df['Trend_Direction'] == 'Worsening'])
        st.metric("Countries with Worsening Trends", worsening_trends)

    # MAIN ANALYSIS TABS ALIGNED WITH OBJECTIVES
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Pattern Identification",
        "🚨 Risk Detection",
        "📈 Trend Analysis",
        "🔮 Predictive Analytics",
        "💡 Policy Solutions",
        "📋 Data Diagnostics"
    ])

    with tab1:
        st.header("🎯 Pattern Identification & Fiscal Drivers")

        col1, col2 = st.columns(2)

        with col1:
            # Fiscal behavior clustering
            st.subheader("Country Fiscal Behavior Patterns")
            fig_patterns = px.scatter(
                sustainability_df,
                x='Trend_Coefficient',
                y='Volatility',
                size='Deficit_Percentage',
                color='Trend_Direction',
                hover_name='Country',
                title="Fiscal Behavior Clustering: Trends vs Volatility",
                labels={'Trend_Coefficient': 'Trend (Negative = Worsening)', 'Volatility': 'Fiscal Volatility'}
            )
            st.plotly_chart(fig_patterns, use_container_width=True)

        with col2:
            # Deficit persistence analysis
            st.subheader("Deficit Persistence Patterns")
            deficit_persistence = sustainability_df.sort_values('Deficit_Percentage', ascending=False)
            fig_persistence = px.bar(
                deficit_persistence.head(10),
                x='Country',
                y='Deficit_Percentage',
                color='Deficit_Percentage',
                title="Countries with Highest Deficit Frequency",
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_persistence, use_container_width=True)

        # Cross-country correlation analysis
        st.subheader("Cross-Country Fiscal Correlation Matrix")
        pivot_corr = filtered_df.pivot_table(
            values='Amount', index='Time', columns='Country', aggfunc='sum'
        ).corr()

        fig_corr = px.imshow(
            pivot_corr,
            title="Fiscal Policy Correlation Between Countries",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        st.header("🚨 Risk Detection & Anomaly Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Risk classification
            st.subheader("Fiscal Stress Risk Classification")
            risk_summary = predictions_df.groupby('Risk_Level').size().reset_index(name='Count')
            fig_risk = px.pie(
                risk_summary,
                values='Count',
                names='Risk_Level',
                title="Country Distribution by Risk Level",
                color='Risk_Level',
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            # Anomaly detection results
            st.subheader("Detected Fiscal Anomalies")
            anomalies = anomaly_df[anomaly_df['Anomaly'] == -1]
            if not anomalies.empty:
                fig_anomalies = px.scatter(
                    anomalies,
                    x='Time',
                    y='Amount',
                    color='Country',
                    size=np.abs(anomalies['Amount']),
                    title="Detected Fiscal Anomalies Over Time",
                    hover_data=['Anomaly_Score']
                )
                st.plotly_chart(fig_anomalies, use_container_width=True)
            else:
                st.info("No significant anomalies detected in current selection")

        # Detailed risk factors
        st.subheader("Risk Factor Analysis")
        risk_factors = predictions_df.merge(sustainability_df, on='Country', how='left')

        # Filter out rows where 'Risk_Score' is NaN before plotting
        risk_factors_plot = risk_factors.dropna(subset=['Risk_Score'])

        if not risk_factors_plot.empty:
            fig_risk_factors = px.scatter(
                risk_factors_plot,
                x='Volatility',
                y='Trend_Coefficient',
                size='Risk_Score',
                color='Risk_Level',
                hover_name='Country',
                title="Risk Factors: Volatility vs Trend Direction",
                labels={'Trend_Coefficient': 'Trend Coefficient', 'Volatility': 'Fiscal Volatility'}
            )
            st.plotly_chart(fig_risk_factors, use_container_width=True)
        else:
            st.info("Insufficient data to plot Risk Factors after cleaning.")

    with tab3:
        st.header("📈 Long-term Trend Analysis & Drivers")

        # Multi-country trend analysis
        st.subheader("Comparative Long-term Fiscal Trends")
        fig_trends = px.line(
            filtered_df,
            x='Time',
            y='Amount',
            color='Country',
            title="Long-term Fiscal Trends (1990-2024)",
            hover_data=['Currency', 'Frequency'],
            line_shape='spline'
        )
        fig_trends.update_layout(height=500)
        st.plotly_chart(fig_trends, use_container_width=True)

        # Trend decomposition by decade
        st.subheader("Decadal Trend Analysis")
        filtered_df['Decade'] = (filtered_df['Year'] // 10) * 10
        decade_trends = filtered_df.groupby(['Country', 'Decade'])['Amount'].mean().reset_index()

        fig_decade = px.line(
            decade_trends,
            x='Decade',
            y='Amount',
            color='Country',
            title="Fiscal Performance by Decade",
            markers=True
        )
        st.plotly_chart(fig_decade, use_container_width=True)

        # Structural break analysis
        st.subheader("Structural Break Identification")
        st.info("""
        **Structural Break Indicators**:
        - Sudden changes in trend direction
        - Significant volatility shifts
        - Persistent deficit/surplus regime changes
        """)

    with tab4:
        st.header("🔮 Predictive Analytics & Early Warning System")

        col1, col2 = st.columns(2)

        with col1:
            # Fiscal stress predictions
            st.subheader("Fiscal Stress Forecast (Next 3 Years)")
            if not predictions_df.empty:
                fig_forecast = px.bar(
                    predictions_df.sort_values('Predicted_Deficit'),
                    x='Country',
                    y='Predicted_Deficit',
                    color='Risk_Level',
                    title="Predicted Fiscal Position",
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

        with col2:
            # Confidence in predictions
            st.subheader("Prediction Confidence Levels")
            if not predictions_df.empty:
                fig_confidence = px.scatter(
                    predictions_df,
                    x='Risk_Score',
                    y='Confidence',
                    size=np.abs(predictions_df['Predicted_Deficit']), # Changed this line to use absolute values
                    color='Risk_Level',
                    hover_name='Country',
                    title="Prediction Confidence vs Risk Score",
                    labels={'Risk_Score': 'Risk Score', 'Confidence': 'Confidence Level (%)'}
                )
                st.plotly_chart(fig_confidence, use_container_width=True)

        # Early warning indicators
        st.subheader("Early Warning System Dashboard")
        warning_indicators = []

        for _, country_data in sustainability_df.iterrows():
            warnings = []
            if country_data['Trend_Coefficient'] < -0.1:
                warnings.append("Worsening trend")
            if country_data['Volatility'] > sustainability_df['Volatility'].quantile(0.75):
                warnings.append("High volatility")
            if country_data['Deficit_Percentage'] > 80:
                warnings.append("Persistent deficits")

            warning_indicators.append({
                'Country': country_data['Country'],
                'Warning_Count': len(warnings),
                'Warnings': ', '.join(warnings) if warnings else 'Stable',
                'Status': 'High Alert' if len(warnings) >= 2 else 'Watch' if len(warnings) == 1 else 'Stable'
            })

        warning_df = pd.DataFrame(warning_indicators)
        st.dataframe(warning_df, use_container_width=True)

    with tab5:
        st.header("💡 Evidence-Based Policy Solutions")

        st.subheader("Tailored Policy Recommendations")

        # Generate recommendations based on risk profile
        recommendations = []
        for _, country in sustainability_df.iterrows():
            risk_profile = predictions_df[predictions_df['Country'] == country['Country']]
            risk_level = risk_profile['Risk_Level'].iloc[0] if not risk_profile.empty else 'Unknown'

            if risk_level == 'High':
                recs = [
                    "Immediate expenditure review and prioritization",
                    "Revenue mobilization reforms",
                    "Debt sustainability analysis",
                    "IMF/World Bank consultation recommended"
                ]
            elif risk_level == 'Medium':
                recs = [
                    "Medium-term fiscal framework development",
                    "Contingency planning for revenue shortfalls",
                    "Public investment efficiency review",
                    "Strengthen fiscal reporting systems"
                ]
            else:
                recs = [
                    "Maintain fiscal discipline",
                    "Continue current policy framework",
                    "Build fiscal buffers for future shocks",
                    "Monitor external vulnerabilities"
                ]

            recommendations.append({
                'Country': country['Country'],
                'Risk_Level': risk_level,
                'Primary_Recommendation': recs[0],
                'Secondary_Measures': '; '.join(recs[1:3]),
                'Implementation_Timeframe': 'Immediate' if risk_level == 'High' else 'Medium-term'
            })

        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)

        # Policy impact simulation
        st.subheader("Policy Impact Simulation")
        st.info("""
        **Available Policy Levers**:
        - Revenue enhancement measures
        - Expenditure rationalization
        - Debt restructuring options
        - Economic growth stimulation
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            revenue_impact = st.slider("Revenue Improvement (%)", 0, 20, 5)
        with col2:
            spending_cut = st.slider("Spending Reduction (%)", 0, 15, 3)
        with col3:
            growth_boost = st.slider("Growth Impact (%)", 0, 10, 2)

        # Simulate policy impact
        simulated_improvement = (revenue_impact + spending_cut) * 0.8 + growth_boost * 0.5
        st.metric("Estimated Fiscal Improvement", f"{simulated_improvement:.1f}%")

    with tab6:
        st.header("📋 Data Quality & Methodology")

        col1, col2 = st.columns(2)

        with col1:
            # Data coverage analysis
            st.subheader("Data Coverage Assessment")
            coverage = filtered_df.groupby('Country').agg({
                'Time': ['min', 'max', 'count'],
                'Amount': ['mean', 'std']
            }).round(2)

            coverage.columns = ['Start_Year', 'End_Year', 'Data_Points', 'Mean_Amount', 'Std_Dev']
            coverage['Coverage_Score'] = (coverage['Data_Points'] / coverage['Data_Points'].max() * 100).round(1)

            st.dataframe(coverage, use_container_width=True)

        with col2:
            # Methodology
            st.subheader("Analytical Methodology")
            st.markdown("""
            **Risk Assessment Framework**:
            - Trend analysis: Linear regression coefficients
            - Volatility: Standard deviation of fiscal balances
            - Deficit persistence: Frequency of deficit occurrences
            - Anomaly detection: Isolation Forest algorithm

            **Predictive Model**:
            - Moving average forecasting
            - Risk scoring based on multiple factors
            - Confidence intervals based on data quality
            """)

        # Data source reliability
        st.subheader("Data Source Reliability")
        source_reliability = df['Source'].value_counts().reset_index()
        source_reliability.columns = ['Source', 'Records']
        source_reliability['Reliability_Score'] = (source_reliability['Records'] / source_reliability['Records'].max() * 100).round(1)

        fig_sources = px.bar(
            source_reliability,
            x='Source',
            y='Reliability_Score',
            title="Data Source Reliability Assessment",
            color='Reliability_Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_sources, use_container_width=True)

    # Footer with key insights
    st.markdown("---")
    st.header("🎯 Key Insights for Policymakers")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.subheader("🚨 Critical Risks Identified")
        high_risk_countries = predictions_df[predictions_df['Risk_Level'] == 'High']['Country'].tolist()
        if high_risk_countries:
            st.error(f"**Immediate Attention Required**: {', '.join(high_risk_countries)}")
        else:
            st.success("No countries currently at high risk level")

    with insights_col2:
        st.subheader("📈 Success Stories")
        improving_countries = sustainability_df[sustainability_df['Trend_Direction'] == 'Improving']['Country'].tolist()
        if improving_countries:
            st.success(f"**Positive Trends**: {', '.join(improving_countries)}")
        else:
            st.warning("Limited positive trends identified in current selection")

    # Export and reporting
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export & Reporting")

    if not filtered_df.empty:
        # Create comprehensive report
        report_data = {
            'Executive_Summary': f"Analysis of {len(selected_countries)} countries from {year_range[0]} to {year_range[1]}",
            'High_Risk_Countries': high_risk_countries,
            'Key_Findings': sustainability_df.to_dict('records'),
            'Policy_Recommendations': recommendations
        }

        st.sidebar.download_button(
            label="📄 Download Analysis Report",
            data=str(report_data),
            file_name=f"fiscal_sustainability_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()


# In[ ]:




