import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ==============================
# DATA GENERATION
# ==============================
np.random.seed(42)

data = pd.DataFrame({
    'CustomerID': range(1, 301),
    'SignupDate': pd.date_range(start='2024-01-01', periods=300, freq='D'),
    'Churn': np.random.choice(['Yes', 'No'], 300, p=[0.3, 0.7]),
    'Plan': np.random.choice(['Basic', 'Standard', 'Premium'], 300),
    'Revenue': np.random.randint(50, 500, 300),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 300)
})

data['SignupDate'] = pd.to_datetime(data['SignupDate'])
data['ChurnFlag'] = data['Churn'].map({'Yes':1, 'No':0})

# ==============================
# KPIs
# ==============================
churn_rate = round(data['ChurnFlag'].mean()*100,2)
total_customers = len(data)
avg_revenue = round(data['Revenue'].mean(),2)

# ==============================
# 1. MONTHLY TREND (JAN-FEB)
# ==============================
data['Month'] = data['SignupDate'].dt.month
filtered = data[data['Month'].isin([1,2])]

monthly = filtered.groupby(filtered['SignupDate'].dt.to_period('M')).agg({'ChurnFlag':'mean'}).reset_index()
monthly['SignupDate'] = monthly['SignupDate'].astype(str)

fig1 = px.line(monthly, x='SignupDate', y='ChurnFlag',
               title='Churn Trend (Jan-Feb)', markers=True)
fig1.update_layout(template='plotly_dark', height=350)

# ==============================
# 2. CHURN BY PLAN
# ==============================
fig2 = px.bar(data, x='Plan', y='ChurnFlag', color='Plan',
              title='Churn by Plan')
fig2.update_layout(template='plotly_dark', height=350)

# ==============================
# 3. REVENUE DISTRIBUTION
# ==============================
fig3 = px.histogram(data, x='Revenue', nbins=20,
                    title='Revenue Distribution')
fig3.update_layout(template='plotly_dark', height=350)

# ==============================
# 4. CUSTOMER SEGMENTATION
# ==============================
data['Segment'] = pd.qcut(data['Revenue'], 3,
                          labels=['Low', 'Medium', 'High'])

fig4 = px.bar(data, x='Segment', y='ChurnFlag', color='Segment',
              title='Churn by Customer Segment')
fig4.update_layout(template='plotly_dark', height=350)

# ==============================
# 5. COHORT HEATMAP (ADVANCED)
# ==============================
data['CohortMonth'] = data['SignupDate'].dt.to_period('M')
data['OrderMonth'] = data['SignupDate'].dt.to_period('M')

cohort = data.groupby(['CohortMonth','OrderMonth']).agg({'CustomerID':'count'}).reset_index()
cohort['CohortIndex'] = (cohort['OrderMonth'] - cohort['CohortMonth']).apply(lambda x: x.n)

cohort_pivot = cohort.pivot(index='CohortMonth',
                           columns='CohortIndex',
                           values='CustomerID').fillna(0)

fig5 = go.Figure(data=go.Heatmap(
    z=cohort_pivot.values,
    x=cohort_pivot.columns,
    y=cohort_pivot.index.astype(str),
    colorscale='Viridis'
))
fig5.update_layout(title='Cohort Retention Heatmap',
                   template='plotly_dark', height=350)

# ==============================
# 6. ML CHURN PREDICTION
# ==============================
X = data[['Revenue']]
y = data['ChurnFlag']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

data['ChurnRisk'] = model.predict_proba(X)[:,1]

fig6 = px.scatter(data, x='Revenue', y='ChurnRisk',
                  color='ChurnRisk',
                  title='Churn Prediction (ML)')
fig6.update_layout(template='plotly_dark', height=350)

# ==============================
# ROUTE
# ==============================
@app.route('/')
def dashboard():
    return render_template(
        'index.html',
        churn_rate=churn_rate,
        total_customers=total_customers,
        avg_revenue=avg_revenue,
        plot1=fig1.to_html(full_html=False),
        plot2=fig2.to_html(full_html=False),
        plot3=fig3.to_html(full_html=False),
        plot4=fig4.to_html(full_html=False),
        plot5=fig5.to_html(full_html=False),
        plot6=fig6.to_html(full_html=False)
    )

if __name__ == '__main__':
    app.run(debug=True)