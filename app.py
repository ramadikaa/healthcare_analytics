import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# LOAD AND PROCESS DATA (CACHED)
# ============================================
@st.cache_data
def load_data():
    """Load data dari CSV dan lakukan preprocessing"""
    df = pd.read_csv('data/healthcare_dataset.csv')
    
    # Convert date columns to datetime
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    
    # Calculate Length of Stay (LOS)
    df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    
    # Create age groups
    df['Age Group'] = pd.cut(df['Age'], 
                              bins=[0, 18, 35, 50, 65, 100],
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    
    return df


# Load data (CACHED)
df = load_data()


# ============================================
# CACHE EXPENSIVE AGGREGATIONS (OPTIMIZATION)
# ============================================
@st.cache_data
def get_condition_counts(data):
    """Get medical condition counts"""
    result = data['Medical Condition'].value_counts().reset_index()
    result.columns = ['Medical Condition', 'Count']
    return result


@st.cache_data
def get_hospital_billing(data):
    """Get average billing by hospital"""
    result = data.groupby('Hospital')['Billing Amount'].mean().reset_index()
    return result.sort_values('Billing Amount', ascending=False)


@st.cache_data
def get_hospital_stats(data):
    """Get comprehensive hospital statistics"""
    stats = data.groupby('Hospital').agg({
        'Name': 'count',
        'Billing Amount': ['mean', 'sum'],
        'LOS': 'mean',
        'Age': 'mean'
    }).round(2)
    stats.columns = ['Patient Count', 'Avg Billing', 'Total Billing', 'Avg LOS', 'Avg Age']
    return stats.sort_values('Patient Count', ascending=False)


@st.cache_data
def get_doctor_stats(data):
    """Get doctor performance statistics"""
    stats = data.groupby('Doctor').agg({
        'Name': 'count',
        'Billing Amount': 'mean',
        'LOS': 'mean'
    }).round(2)
    stats.columns = ['Patient Count', 'Avg Billing', 'Avg LOS']
    return stats.sort_values('Patient Count', ascending=False).head(15)


@st.cache_data
def get_age_group_stats(data):
    """Get age group statistics"""
    stats = data.groupby('Age Group').agg({
        'Name': 'count',
        'Billing Amount': 'mean',
        'LOS': 'mean'
    }).round(2)
    stats.columns = ['Patient Count', 'Avg Billing', 'Avg LOS']
    return stats


@st.cache_data
def get_condition_billing(data):
    """Get average billing by medical condition"""
    result = data.groupby('Medical Condition')['Billing Amount'].mean().reset_index()
    return result.sort_values('Billing Amount', ascending=False)


@st.cache_data
def get_condition_los(data):
    """Get average LOS by medical condition"""
    result = data.groupby('Medical Condition')['LOS'].mean().reset_index()
    return result.sort_values('LOS', ascending=False)


@st.cache_data
def get_insurance_counts(data):
    """Get insurance provider counts"""
    result = data['Insurance Provider'].value_counts().head(8).reset_index()
    result.columns = ['Insurance Provider', 'Count']
    return result


@st.cache_data
def get_daily_admissions(data):
    """Get daily admissions trend"""
    result = data.groupby(data['Date of Admission'].dt.date).size().reset_index(name='Count')
    result.columns = ['Date', 'Admissions']
    return result


# ============================================
# SIDEBAR - FILTERS
# ============================================
st.sidebar.header("🔍 Filters")


# Hospital Filter
all_hospitals = ['All Hospitals'] + sorted(df['Hospital'].unique().tolist())
selected_hospitals = st.sidebar.multiselect(
    "Select Hospital(s):",
    options=all_hospitals,
    default=['All Hospitals']
)


# Medical Condition Filter
all_conditions = ['All Conditions'] + sorted(df['Medical Condition'].unique().tolist())
selected_conditions = st.sidebar.multiselect(
    "Select Medical Condition(s):",
    options=all_conditions,
    default=['All Conditions']
)


# Insurance Provider Filter
all_insurance = ['All Insurance'] + sorted(df['Insurance Provider'].unique().tolist())
selected_insurance = st.sidebar.multiselect(
    "Select Insurance Provider(s):",
    options=all_insurance,
    default=['All Insurance']
)


# Admission Type Filter
all_admission = ['All Types'] + sorted(df['Admission Type'].unique().tolist())
selected_admission = st.sidebar.multiselect(
    "Select Admission Type(s):",
    options=all_admission,
    default=['All Types']
)


# Gender Filter
all_gender = ['All Gender'] + sorted(df['Gender'].unique().tolist())
selected_gender = st.sidebar.multiselect(
    "Select Gender(s):",
    options=all_gender,
    default=['All Gender']
)


# ============================================
# APPLY FILTERS
# ============================================
filtered_df = df.copy()


# Filter Hospital
if 'All Hospitals' not in selected_hospitals:
    filtered_df = filtered_df[filtered_df['Hospital'].isin(selected_hospitals)]


# Filter Medical Condition
if 'All Conditions' not in selected_conditions:
    filtered_df = filtered_df[filtered_df['Medical Condition'].isin(selected_conditions)]


# Filter Insurance
if 'All Insurance' not in selected_insurance:
    filtered_df = filtered_df[filtered_df['Insurance Provider'].isin(selected_insurance)]


# Filter Admission Type
if 'All Types' not in selected_admission:
    filtered_df = filtered_df[filtered_df['Admission Type'].isin(selected_admission)]


# Filter Gender
if 'All Gender' not in selected_gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]


# ============================================
# MAIN CONTENT - HEADER
# ============================================
st.title("🏥 Hospital Operational Analytics Dashboard")
st.markdown("Analitik Deskriptif Operasional Rumah Sakit")


# Display data range info
st.info(f"📊 Total Patients: {len(filtered_df):,} | Data Range: {df['Date of Admission'].min().date()} to {df['Discharge Date'].max().date()}")


# ============================================
# TABS NAVIGATION
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🔍 Detailed Analysis", 
    "🏥 Hospital Comparison",
    "🧪 Medical Details",
    "📋 Data Explorer"
])


# ============================================
# TAB 1: OVERVIEW
# ============================================
with tab1:
    st.subheader("📈 Key Performance Indicators (KPI)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(filtered_df)
        st.metric(
            label="Total Patients",
            value=f"{total_patients:,}",
            delta=f"{(total_patients/len(df)*100):.1f}% of total"
        )
    
    with col2:
        avg_billing = filtered_df['Billing Amount'].mean()
        st.metric(
            label="Avg Billing Amount",
            value=f"${avg_billing:,.2f}",
            delta=f"Min: ${filtered_df['Billing Amount'].min():,.0f}"
        )
    
    with col3:
        avg_los = filtered_df['LOS'].mean()
        st.metric(
            label="Avg Length of Stay",
            value=f"{avg_los:.1f} days",
            delta=f"Max: {filtered_df['LOS'].max()} days"
        )
    
    with col4:
        total_hospitals = filtered_df['Hospital'].nunique()
        st.metric(
            label="Total Hospitals",
            value=f"{total_hospitals}",
            delta=f"of {df['Hospital'].nunique()} hospitals"
        )
    
    # ============================================
    # ENHANCEMENT 1: QUICK INSIGHTS
    # ============================================
    st.divider()
    st.subheader("💡 Quick Insights")
    
    col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)
    
    with col_insight1:
        male_count = (filtered_df['Gender'] == 'Male').sum()
        st.metric("Male Patients", f"{male_count:,}", f"{(male_count/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "N/A")
    
    with col_insight2:
        female_count = (filtered_df['Gender'] == 'Female').sum()
        st.metric("Female Patients", f"{female_count:,}", f"{(female_count/len(filtered_df)*100):.1f}%" if len(filtered_df) > 0 else "N/A")
    
    with col_insight3:
        avg_age = filtered_df['Age'].mean()
        st.metric("Avg Patient Age", f"{avg_age:.1f} years", f"Range: {filtered_df['Age'].min()}-{filtered_df['Age'].max()}")
    
    with col_insight4:
        max_billing = filtered_df['Billing Amount'].max()
        st.metric("Max Billing Amount", f"${max_billing:,.2f}", "Peak charge")
    
    st.divider()
    st.subheader("📊 Analysis Overview")
    
    col_a, col_b = st.columns(2)
    
    # VISUALIZATION 1: Distribution of Medical Conditions
    with col_a:
        with st.spinner('Loading condition distribution...'):
            condition_counts = get_condition_counts(filtered_df)
            
            fig_condition = px.pie(
                condition_counts,
                values='Count',
                names='Medical Condition',
                title='Distribution of Medical Conditions',
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_condition.update_traces(textposition='auto', textinfo='percent+label')
            
            st.plotly_chart(fig_condition, use_container_width=True)
    
    # VISUALIZATION 2: Average Billing by Hospital
    with col_b:
        with st.spinner('Loading hospital billing data...'):
            hospital_billing = get_hospital_billing(filtered_df)
            
            fig_hospital = px.bar(
                hospital_billing,
                x='Hospital',
                y='Billing Amount',
                title='Average Billing Amount by Hospital',
                color='Billing Amount',
                color_continuous_scale='Viridis'
            )
            fig_hospital.update_layout(
                xaxis_tickangle=-45,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_hospital, use_container_width=True)
    
    col_c, col_d = st.columns(2)
    
    # VISUALIZATION 3: Age Distribution
    with col_c:
        with st.spinner('Loading age distribution...'):
            fig_age = px.histogram(
                filtered_df,
                x='Age',
                nbins=20,
                title='Distribution of Patient Age',
                color_discrete_sequence=['#636EFA'],
                marginal='box'
            )
            fig_age.update_layout(hovermode='x unified')
            
            st.plotly_chart(fig_age, use_container_width=True)
    
    # VISUALIZATION 4: Admission Type Distribution
    with col_d:
        with st.spinner('Loading admission type data...'):
            admission_counts = filtered_df['Admission Type'].value_counts().reset_index()
            admission_counts.columns = ['Admission Type', 'Count']
            
            fig_admission = px.bar(
                admission_counts,
                x='Admission Type',
                y='Count',
                title='Distribution of Admission Type',
                color='Admission Type',
                color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA']
            )
            fig_admission.update_layout(hovermode='x unified')
            
            st.plotly_chart(fig_admission, use_container_width=True)


# ============================================
# TAB 2: DETAILED ANALYSIS
# ============================================
with tab2:
    st.subheader("💡 Detailed Insights")
    
    col_e, col_f = st.columns(2)
    
    # VISUALIZATION 5: Top Insurance Providers
    with col_e:
        with st.spinner('Loading insurance data...'):
            insurance_counts = get_insurance_counts(filtered_df)
            
            fig_insurance = px.bar(
                insurance_counts,
                y='Insurance Provider',
                x='Count',
                title='Top 8 Insurance Providers by Patient Count',
                color='Count',
                color_continuous_scale='Blues',
                orientation='h'
            )
            
            st.plotly_chart(fig_insurance, use_container_width=True)
    
    # VISUALIZATION 6: Billing by Admission Type
    with col_f:
        with st.spinner('Loading billing distribution...'):
            fig_billing_admission = px.box(
                filtered_df,
                x='Admission Type',
                y='Billing Amount',
                title='Billing Amount Distribution by Admission Type',
                color='Admission Type',
                color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA']
            )
            
            st.plotly_chart(fig_billing_admission, use_container_width=True)
    
    col_g, col_h = st.columns(2)
    
    # VISUALIZATION 7: Average Billing by Medical Condition
    with col_g:
        with st.spinner('Loading condition billing data...'):
            condition_billing = get_condition_billing(filtered_df)
            
            fig_condition_billing = px.bar(
                condition_billing,
                x='Medical Condition',
                y='Billing Amount',
                title='Average Billing Amount by Medical Condition',
                color='Billing Amount',
                color_continuous_scale='Reds'
            )
            fig_condition_billing.update_layout(xaxis_tickangle=-45)
            
            st.plotly_chart(fig_condition_billing, use_container_width=True)
    
    # VISUALIZATION 8: Average LOS by Medical Condition
    with col_h:
        with st.spinner('Loading LOS data...'):
            condition_los = get_condition_los(filtered_df)
            
            fig_condition_los = px.bar(
                condition_los,
                x='Medical Condition',
                y='LOS',
                title='Average Length of Stay by Medical Condition',
                color='LOS',
                color_continuous_scale='Greens'
            )
            fig_condition_los.update_layout(xaxis_tickangle=-45)
            
            st.plotly_chart(fig_condition_los, use_container_width=True)
    
    st.divider()
    
    # CORRELATION HEATMAP
    st.subheader("📊 Correlation Analysis")
    
    with st.spinner('Loading correlation matrix...'):
        numeric_cols = filtered_df[['Age', 'Billing Amount', 'LOS']].copy()
        correlation_matrix = numeric_cols.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(title='Correlation Matrix - Numeric Variables', height=400)
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # SCATTER PLOT
    st.subheader("🔍 Relationship Analysis")
    
    col_scatter1, col_scatter2 = st.columns(2)
    
    with col_scatter1:
        with st.spinner('Loading Age vs Billing scatter plot...'):
            fig_scatter1 = px.scatter(
                filtered_df,
                x='Age',
                y='Billing Amount',
                title='Age vs Billing Amount',
                color='Medical Condition',
                hover_data=['Hospital', 'Admission Type'],
                size_max=8
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)

    with col_scatter2:
        with st.spinner('Loading LOS vs Billing scatter plot...'):
            fig_scatter2 = px.scatter(
                filtered_df,
                x='LOS',
                y='Billing Amount',
                title='Length of Stay vs Billing Amount',
                color='Admission Type',
                hover_data=['Hospital', 'Medical Condition'],
                size_max=8
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)


# ============================================
# TAB 3: HOSPITAL COMPARISON
# ============================================
with tab3:
    st.subheader("🏥 Hospital Performance Statistics")
    
    with st.spinner('Loading hospital statistics...'):
        hospital_stats = get_hospital_stats(filtered_df)
        st.dataframe(hospital_stats, use_container_width=True)
    
    # TIME SERIES ANALYSIS
    st.divider()
    st.subheader("📈 Time Series Trends")
    
    with st.spinner('Loading admissions trend...'):
        daily_admissions = get_daily_admissions(filtered_df)
        
        fig_timeseries = px.line(
            daily_admissions,
            x='Date',
            y='Admissions',
            title='Daily Admissions Trend',
            markers=True
        )
        fig_timeseries.update_layout(hovermode='x unified')
        
        st.plotly_chart(fig_timeseries, use_container_width=True)
    
    st.divider()
    st.subheader("👨‍⚕️ Doctor Performance Statistics")
    
    with st.spinner('Loading doctor statistics...'):
        doctor_stats = get_doctor_stats(filtered_df)
        st.dataframe(doctor_stats, use_container_width=True)


# ============================================
# TAB 4: MEDICAL DETAILS
# ============================================
with tab4:
    st.subheader("💊 Medication Analysis")
    
    col_med1, col_med2 = st.columns(2)
    
    with col_med1:
        with st.spinner('Loading medication data...'):
            medication_counts = filtered_df['Medication'].value_counts().head(10).reset_index()
            medication_counts.columns = ['Medication', 'Count']
            
            fig_med = px.bar(
                medication_counts,
                x='Count',
                y='Medication',
                orientation='h',
                title='Top 10 Medications Used',
                color='Count',
                color_continuous_scale='Purples'
            )
            
            st.plotly_chart(fig_med, use_container_width=True)
    
    with col_med2:
        with st.spinner('Loading test results...'):
            test_results_counts = filtered_df['Test Results'].value_counts().reset_index()
            test_results_counts.columns = ['Test Results', 'Count']
            
            fig_test = px.pie(
                test_results_counts,
                values='Count',
                names='Test Results',
                title='Test Results Distribution'
            )
            fig_test.update_traces(textposition='auto', textinfo='percent+label')
            
            st.plotly_chart(fig_test, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Detailed Statistics by Medical Condition")
    
    with st.spinner('Loading detailed statistics...'):
        summary_stats = filtered_df.groupby('Medical Condition').agg({
            'Name': 'count',
            'Age': ['mean', 'min', 'max'],
            'Billing Amount': ['mean', 'min', 'max'],
            'LOS': ['mean', 'min', 'max']
        }).round(2)
        
        summary_stats.columns = ['Patient Count', 'Avg Age', 'Min Age', 'Max Age', 
                                 'Avg Billing', 'Min Billing', 'Max Billing', 
                                 'Avg LOS', 'Min LOS', 'Max LOS']
        
        st.dataframe(summary_stats, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Data Summary by Demographics")
    
    col_dem1, col_dem2, col_dem3 = st.columns(3)
    
    with col_dem1:
        st.write("**Gender Distribution**")
        gender_counts = filtered_df['Gender'].value_counts()
        st.bar_chart(gender_counts)
    
    with col_dem2:
        st.write("**Blood Type Distribution**")
        blood_counts = filtered_df['Blood Type'].value_counts()
        st.bar_chart(blood_counts)
    
    with col_dem3:
        st.write("**Test Results Distribution**")
        test_counts = filtered_df['Test Results'].value_counts()
        st.bar_chart(test_counts)
    
    # ============================================
    # ENHANCEMENT 4: AGE GROUP ANALYSIS
    # ============================================
    st.divider()
    st.subheader("👥 Age Group Analysis")
    
    with st.spinner('Loading age group analysis...'):
        age_group_stats = get_age_group_stats(filtered_df)
        
        col_age_chart1, col_age_chart2 = st.columns(2)
        
        with col_age_chart1:
            fig_age_group = px.bar(
                age_group_stats.reset_index(),
                x='Age Group',
                y='Patient Count',
                title='Patient Count by Age Group',
                color='Patient Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_age_group, use_container_width=True)
        
        with col_age_chart2:
            fig_age_billing = px.bar(
                age_group_stats.reset_index(),
                x='Age Group',
                y='Avg Billing',
                title='Average Billing by Age Group',
                color='Avg Billing',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_age_billing, use_container_width=True)
        
        st.dataframe(age_group_stats, use_container_width=True)


# ============================================
# TAB 5: DATA EXPLORER
# ============================================
with tab5:
    st.subheader("🔎 Raw Patient Data")
    
    # ============================================
    # ENHANCEMENT 2: SEARCH BY PATIENT NAME
    # ============================================
    search_name = st.text_input("🔍 Search by Patient Name:", "")
    
    if search_name:
        filtered_df_search = filtered_df[filtered_df['Name'].str.contains(search_name, case=False, na=False)]
        st.info(f"Found {len(filtered_df_search)} patient(s) matching '{search_name}'")
    else:
        filtered_df_search = filtered_df
    
    # OPTIMIZATION: Limit rows displayed
    display_limit = st.slider("Rows to display:", 100, 5000, 1000, 100)
    filtered_df_display = filtered_df_search.head(display_limit)
    
    # Add column selection for better readability
    columns_to_display = st.multiselect(
        "Select columns to display:",
        options=filtered_df_display.columns.tolist(),
        default=['Name', 'Age', 'Gender', 'Medical Condition', 'Hospital', 'Doctor', 
                 'Billing Amount', 'Admission Type', 'LOS', 'Test Results']
    )
    
    # Display table
    st.dataframe(
        filtered_df_display[columns_to_display],
        use_container_width=True,
        height=400
    )
    
    st.divider()
    
    # ============================================
    # ENHANCEMENT 3: SUMMARY REPORT
    # ============================================
    st.subheader("📄 Summary Report")
    
    col_report1, col_report2, col_report3 = st.columns(3)
    
    with col_report1:
        st.metric("Total Records Analyzed", len(filtered_df))
        st.metric("Unique Doctors", filtered_df['Doctor'].nunique())
        st.metric("Unique Medications", filtered_df['Medication'].nunique())
    
    with col_report2:
        st.metric("Min Billing", f"${filtered_df['Billing Amount'].min():,.2f}")
        st.metric("Max Billing", f"${filtered_df['Billing Amount'].max():,.2f}")
        st.metric("Std Dev Billing", f"${filtered_df['Billing Amount'].std():,.2f}")
    
    with col_report3:
        st.metric("Min LOS", f"{filtered_df['LOS'].min():.0f} days")
        st.metric("Max LOS", f"{filtered_df['LOS'].max():.0f} days")
        st.metric("Std Dev LOS", f"{filtered_df['LOS'].std():.2f} days")
    
    st.divider()
    st.subheader("⬇️ Export Data")
    
    # Convert to CSV
    csv_data = filtered_df_search.to_csv(index=False).encode('utf-8')
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"hospital_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Export Excel
    with col_export2:
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df_search.to_excel(writer, sheet_name='Patient Data', index=False)
                hospital_stats.to_excel(writer, sheet_name='Hospital Stats')
                doctor_stats.to_excel(writer, sheet_name='Doctor Stats')
                age_group_stats.to_excel(writer, sheet_name='Age Group Stats')
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=f"hospital_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning(f"⚠️ Excel export not available: {str(e)}")


# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
---
**Dashboard Information:**
- 📊 This dashboard provides descriptive analytics of hospital operations
- 🔍 Use filters in the sidebar to explore specific segments
- 📈 All metrics are calculated from the selected filtered data
- 💾 Download filtered data for further analysis
- ⚡ Optimized with caching for faster performance (55,500+ records)

**Enhanced Features:**
1. 💡 Quick Insights - Gender & age breakdown
2. 🔍 Patient Search - Search by name functionality
3. 📄 Summary Report - Key statistics and metrics
4. 👥 Age Group Analysis - Breakdown by age demographics
5. ⚡ Performance Optimization - Caching & spinners

**Data Columns Used:**
- Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital
- Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date
- Medication, Test Results, LOS (calculated), Age Group (calculated)

**Created for:** Informatika Terapan - Module 7: Data Analytics & Business Intelligence in Healthcare
""")
