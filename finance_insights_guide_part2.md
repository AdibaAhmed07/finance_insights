# Personal Finance "Behavioral" Insights Engine
## Part 2: ML Implementation & Deployment (Phases 3-4)

---

## 🎯 Phase 3: The "Intelligence" Layer (Weeks 7-9)

### Week 7-8: K-Means Clustering for Financial Personas

#### Person B - Clustering Implementation

**File: backend/ml/clustering.py**
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from models import Transaction, UserPersona, User
from datetime import datetime

def calculate_user_features(db: Session, user_id: int):
    """
    Extract features for clustering from user transactions
    """
    transactions = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'debit'
    ).all()
    
    if not transactions:
        return None
    
    df = pd.DataFrame([{
        'amount': abs(float(t.amount)),
        'category': t.category,
        'is_weekend': t.transaction_date.weekday() >= 5,
        'hour': t.transaction_date.hour
    } for t in transactions])
    
    # Calculate comprehensive features
    features = {
        'avg_transaction_value': df['amount'].mean(),
        'median_transaction_value': df['amount'].median(),
        'std_transaction_value': df['amount'].std(),
        'total_spending': df['amount'].sum(),
        'transaction_count': len(df),
        'spending_frequency': len(df) / 30,  # Approx transactions per month
        'weekend_spending_ratio': (
            df[df['is_weekend']]['amount'].sum() / df['amount'].sum() 
            if len(df) > 0 else 0
        ),
        'max_transaction': df['amount'].max(),
        'categories_used': df['category'].nunique(),
        'evening_spending_ratio': (
            len(df[df['hour'] >= 18]) / len(df)
            if len(df) > 0 else 0
        )
    }
    
    # Top category
    features['top_category'] = df['category'].value_counts().index[0] if len(df) > 0 else 'Unknown'
    
    return features

def cluster_users(db: Session, n_clusters: int = 5):
    """
    Perform K-Means clustering on all users to assign financial personas
    """
    # Get all user IDs
    users = db.query(User).all()
    
    user_features = []
    user_ids = []
    
    print(f"Analyzing {len(users)} users for clustering...")
    
    for user in users:
        features = calculate_user_features(db, user.user_id)
        if features:
            user_features.append(features)
            user_ids.append(user.user_id)
    
    if not user_features:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(user_features)
    df['user_id'] = user_ids
    
    # Select numerical features for clustering
    feature_cols = [
        'avg_transaction_value',
        'spending_frequency',
        'weekend_spending_ratio',
        'categories_used',
        'std_transaction_value',
        'evening_spending_ratio'
    ]
    
    X = df[feature_cols].fillna(0)
    
    # Standardize features (important for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters to assign meaningful labels
    cluster_profiles = df.groupby('cluster')[feature_cols].mean()
    
    print("\nCluster Profiles:")
    print(cluster_profiles)
    
    # Assign personas based on cluster characteristics
    for idx, row in df.iterrows():
        cluster_num = row['cluster']
        user_id = row['user_id']
        
        # Determine persona based on spending characteristics
        if row['avg_transaction_value'] < df['avg_transaction_value'].median() and \
           row['spending_frequency'] < df['spending_frequency'].median():
            persona_label = "The Frugal Saver"
            description = "You're careful with money and make infrequent, small purchases."
            
        elif row['weekend_spending_ratio'] > 0.4:
            persona_label = "The Weekend Warrior"
            description = "Your spending spikes on weekends for fun and entertainment."
            
        elif row['avg_transaction_value'] > df['avg_transaction_value'].quantile(0.75):
            persona_label = "The Big Ticket Buyer"
            description = "You make fewer but larger purchases for high-value items."
            
        elif row['spending_frequency'] > df['spending_frequency'].quantile(0.75):
            persona_label = "The Impulsive Techie"
            description = "You make frequent purchases, especially in tech and subscriptions."
            
        else:
            persona_label = "The Balanced Spender"
            description = "You maintain a healthy balance between spending and saving."
        
        # Delete old persona if exists
        db.query(UserPersona).filter(UserPersona.user_id == user_id).delete()
        
        # Create new persona
        persona = UserPersona(
            user_id=user_id,
            persona_label=persona_label,
            cluster_number=cluster_num,
            avg_transaction_value=row['avg_transaction_value'],
            spending_frequency=row['spending_frequency'],
            top_category=row['top_category'],
            confidence_score=0.85  # Could calculate from silhouette score
        )
        db.add(persona)
        
        if user_id % 20 == 0:
            print(f"Assigned persona to user {user_id}: {persona_label}")
    
    db.commit()
    
    return {
        'cluster_profiles': cluster_profiles.to_dict(),
        'user_assignments': df[['user_id', 'cluster', 'avg_transaction_value']].to_dict('records'),
        'total_users_clustered': len(user_ids)
    }
```

Add API endpoints to `main.py`:
```python
from ml.clustering import cluster_users, calculate_user_features

@app.post("/api/ml/cluster-users")
def run_clustering(db: Session = Depends(get_db)):
    """Run K-Means clustering on all users"""
    result = cluster_users(db)
    if result:
        return {"status": "success", "result": result}
    return {"status": "error", "message": "No users with sufficient data"}

@app.get("/api/ml/persona/{user_id}")
def get_user_persona(user_id: int, db: Session = Depends(get_db)):
    """Get financial persona for a specific user"""
    persona = db.query(models.UserPersona).filter(
        models.UserPersona.user_id == user_id
    ).order_by(models.UserPersona.assigned_at.desc()).first()
    
    if not persona:
        return {"message": "No persona assigned yet. Run clustering first."}
    
    return {
        "persona_label": persona.persona_label,
        "cluster_number": persona.cluster_number,
        "avg_transaction_value": float(persona.avg_transaction_value),
        "spending_frequency": float(persona.spending_frequency),
        "top_category": persona.top_category,
        "confidence_score": float(persona.confidence_score),
        "assigned_at": persona.assigned_at.isoformat()
    }
```

**Testing the clustering:**
```bash
# In a separate terminal or using curl
curl -X POST http://localhost:8000/api/ml/cluster-users

# Check a specific user's persona
curl http://localhost:8000/api/ml/persona/1
```

### Week 8-9: Prophet Forecasting

#### Person B - Balance Forecasting with Prophet

**File: backend/ml/forecasting.py**
```python
import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy.orm import Session
from models import Transaction, Account, BalanceForecast
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def prepare_forecast_data(db: Session, user_id: int):
    """
    Prepare transaction data for Prophet forecasting
    """
    # Get account
    account = db.query(Account).filter(Account.user_id == user_id).first()
    if not account:
        return None, None
    
    # Get all transactions
    transactions = db.query(Transaction).filter(
        Transaction.user_id == user_id
    ).order_by(Transaction.transaction_date).all()
    
    if len(transactions) < 30:  # Need at least 30 transactions
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame([{
        'date': t.transaction_date,
        'amount': float(t.amount)
    } for t in transactions])
    
    # Group by day and sum transactions
    df['date'] = pd.to_datetime(df['date']).dt.date
    daily_changes = df.groupby('date')['amount'].sum().reset_index()
    
    # Calculate cumulative balance
    initial_balance = float(account.initial_balance)
    daily_changes['balance'] = initial_balance + daily_changes['amount'].cumsum()
    
    # Prepare for Prophet (needs 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(daily_changes['date']),
        'y': daily_changes['balance']
    })
    
    return prophet_df, account

def forecast_balance(db: Session, user_id: int, days_ahead: int = 30):
    """
    Forecast account balance using Facebook Prophet
    """
    print(f"Generating forecast for user {user_id}...")
    
    # Prepare data
    df, account = prepare_forecast_data(db, user_id)
    
    if df is None:
        return {"error": "Insufficient data for forecasting (need at least 30 transactions)"}
    
    try:
        # Initialize Prophet model with sensible parameters
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,  # Controls flexibility
            seasonality_prior_scale=10.0,
            interval_width=0.95  # 95% confidence interval
        )
        
        # Fit the model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead, freq='D')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Delete old forecasts for this user
        db.query(BalanceForecast).filter(
            BalanceForecast.user_id == user_id,
            BalanceForecast.account_id == account.account_id
        ).delete()
        
        # Save new forecasts (only future dates)
        future_forecasts = forecast[forecast['ds'] > datetime.now()].head(days_ahead)
        
        for _, row in future_forecasts.iterrows():
            forecast_entry = BalanceForecast(
                user_id=user_id,
                account_id=account.account_id,
                forecast_date=row['ds'],
                predicted_balance=max(0, row['yhat']),  # Don't predict negative balance
                lower_bound=max(0, row['yhat_lower']),
                upper_bound=max(0, row['yhat_upper'])
            )
            db.add(forecast_entry)
        
        db.commit()
        
        print(f"✓ Forecast generated successfully for user {user_id}")
        
        return {
            'forecast_data': future_forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
            'current_balance': float(df['y'].iloc[-1]),
            'predicted_30_day': float(future_forecasts.iloc[-1]['yhat']) if len(future_forecasts) > 0 else None,
            'trend': 'increasing' if future_forecasts.iloc[-1]['yhat'] > df['y'].iloc[-1] else 'decreasing'
        }
    
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        return {"error": str(e)}
```

Add forecasting endpoints to `main.py`:
```python
from ml.forecasting import forecast_balance

@app.post("/api/ml/forecast/{user_id}")
def generate_forecast(user_id: int, days: int = 30, db: Session = Depends(get_db)):
    """Generate balance forecast for a user"""
    result = forecast_balance(db, user_id, days)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.get("/api/ml/forecast/{user_id}")
def get_forecast(user_id: int, db: Session = Depends(get_db)):
    """Retrieve saved forecasts for a user"""
    account = db.query(models.Account).filter(
        models.Account.user_id == user_id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    forecasts = db.query(models.BalanceForecast).filter(
        models.BalanceForecast.user_id == user_id,
        models.BalanceForecast.account_id == account.account_id
    ).order_by(models.BalanceForecast.forecast_date).all()
    
    return {
        "forecasts": [{
            "date": f.forecast_date.isoformat(),
            "predicted_balance": float(f.predicted_balance),
            "lower_bound": float(f.lower_bound),
            "upper_bound": float(f.upper_bound)
        } for f in forecasts],
        "current_balance": float(account.current_balance)
    }
```

#### Person A - Insights Page with Persona & Forecast

**File: app/insights/page.tsx**
```typescript
'use client'

import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { Brain, TrendingUp, AlertTriangle, Target, Award } from 'lucide-react'
import Link from 'next/link'

interface Persona {
  persona_label: string
  avg_transaction_value: number
  spending_frequency: number
  top_category: string
  confidence_score: number
}

interface Forecast {
  date: string
  predicted_balance: number
  lower_bound: number
  upper_bound: number
}

export default function InsightsPage() {
  const [persona, setPersona] = useState<Persona | null>(null)
  const [forecast, setForecast] = useState<Forecast[]>([])
  const [loading, setLoading] = useState(true)
  const [currentBalance, setCurrentBalance] = useState(0)

  useEffect(() => {
    fetchInsights()
  }, [])

  const fetchInsights = async () => {
    try {
      // Fetch persona
      const personaRes = await fetch('http://localhost:8000/api/ml/persona/1')
      const personaData = await personaRes.json()
      setPersona(personaData)

      // Fetch forecast
      const forecastRes = await fetch('http://localhost:8000/api/ml/forecast/1')
      const forecastData = await forecastRes.json()
      setForecast(forecastData.forecasts)
      setCurrentBalance(forecastData.current_balance)
    } catch (error) {
      console.error('Error fetching insights:', error)
    } finally {
      setLoading(false)
    }
  }

  const generateForecast = async () => {
    setLoading(true)
    try {
      await fetch('http://localhost:8000/api/ml/forecast/1', {
        method: 'POST'
      })
      fetchInsights()
    } catch (error) {
      console.error('Error generating forecast:', error)
    }
  }

  const getPersonaColor = (label: string) => {
    const colors: { [key: string]: string } = {
      'The Frugal Saver': 'bg-green-100 text-green-800 border-green-300',
      'The Balanced Spender': 'bg-blue-100 text-blue-800 border-blue-300',
      'The Impulsive Techie': 'bg-purple-100 text-purple-800 border-purple-300',
      'The Weekend Warrior': 'bg-orange-100 text-orange-800 border-orange-300',
      'The Big Ticket Buyer': 'bg-red-100 text-red-800 border-red-300'
    }
    return colors[label] || 'bg-gray-100 text-gray-800 border-gray-300'
  }

  const getPersonaIcon = (label: string) => {
    const icons: { [key: string]: JSX.Element } = {
      'The Frugal Saver': <Award className="w-8 h-8" />,
      'The Balanced Spender': <Target className="w-8 h-8" />,
      'The Impulsive Techie': <Brain className="w-8 h-8" />,
      'The Weekend Warrior': <TrendingUp className="w-8 h-8" />,
      'The Big Ticket Buyer': <AlertTriangle className="w-8 h-8" />
    }
    return icons[label] || <Brain className="w-8 h-8" />
  }

  const getPersonaDescription = (label: string) => {
    const descriptions: { [key: string]: string } = {
      'The Frugal Saver': 'You maintain low spending and prioritize saving. Your transactions are typically small and well-planned. Keep up the excellent financial discipline!',
      'The Balanced Spender': 'You have a healthy balance between spending and saving, with moderate transaction amounts. Your financial habits are sustainable.',
      'The Impulsive Techie': 'You tend to make frequent purchases, especially in tech and entertainment categories. Consider setting purchase limits.',
      'The Weekend Warrior': 'Your spending spikes significantly on weekends, especially for dining and entertainment. Budget for weekend fun in advance!',
      'The Big Ticket Buyer': 'You make fewer but larger purchases, often for high-value items. Ensure you have an emergency fund for unexpected costs.'
    }
    return descriptions[label] || 'Your unique spending pattern has been identified.'
  }

  const getPersonaTips = (label: string) => {
    const tips: { [key: string]: string[] } = {
      'The Frugal Saver': [
        'Consider investing your savings for growth',
        'Review if you can afford small quality-of-life upgrades',
        'Share your budgeting strategies with others'
      ],
      'The Balanced Spender': [
        'Maintain your current balanced approach',
        'Set specific savings goals for motivation',
        'Review subscriptions quarterly'
      ],
      'The Impulsive Techie': [
        'Wait 24 hours before making non-essential purchases',
        'Use a "fun money" budget for tech splurges',
        'Unsubscribe from promotional emails'
      ],
      'The Weekend Warrior': [
        'Plan weekend budgets in advance',
        'Find free or low-cost weekend activities',
        'Cook at home before going out'
      ],
      'The Big Ticket Buyer': [
        'Research purchases thoroughly before buying',
        'Build an emergency fund (3-6 months expenses)',
        'Consider payment plans for large purchases'
      ]
    }
    return tips[label] || []
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-xl text-gray-600">Loading your insights...</div>
      </div>
    )
  }

  const finalBalance = forecast.length > 0 ? forecast[forecast.length - 1].predicted_balance : currentBalance
  const balanceChange = finalBalance - currentBalance
  const isPositiveTrend = balanceChange > 0

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Your Financial Insights</h1>
          <div className="flex gap-4">
            <button
              onClick={generateForecast}
              className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700"
            >
              Regenerate Forecast
            </button>
            <Link href="/dashboard" className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700">
              Back to Dashboard
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-8">
        {/* Persona Card */}
        {persona && persona.persona_label && (
          <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
            <div className="flex items-center mb-6">
              <div className={`p-4 rounded-full mr-4 ${getPersonaColor(persona.persona_label)}`}>
                {getPersonaIcon(persona.persona_label)}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Your Financial Persona</h2>
                <p className="text-gray-600">Based on AI analysis of your spending patterns</p>
              </div>
            </div>

            <div className={`inline-block px-6 py-3 rounded-full text-xl font-bold mb-4 border-2 ${getPersonaColor(persona.persona_label)}`}>
              {persona.persona_label}
            </div>

            <p className="text-lg text-gray-700 mb-6">
              {getPersonaDescription(persona.persona_label)}
            </p>

            {/* Stats Grid */}
            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-6 rounded-lg">
                <p className="text-sm text-indigo-600 font-semibold mb-1">Avg Transaction</p>
                <p className="text-3xl font-bold text-indigo-900">
                  ${persona.avg_transaction_value?.toFixed(2)}
                </p>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg">
                <p className="text-sm text-purple-600 font-semibold mb-1">Monthly Frequency</p>
                <p className="text-3xl font-bold text-purple-900">
                  {persona.spending_frequency?.toFixed(0)} txns
                </p>
              </div>
              <div className="bg-gradient-to-br from-pink-50 to-pink-100 p-6 rounded-lg">
                <p className="text-sm text-pink-600 font-semibold mb-1">Top Category</p>
                <p className="text-3xl font-bold text-pink-900">
                  {persona.top_category}
                </p>
              </div>
            </div>

            {/* Personalized Tips */}
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="font-bold text-lg mb-3 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-indigo-600" />
                Personalized Tips for You
              </h3>
              <ul className="space-y-2">
                {getPersonaTips(persona.persona_label).map((tip, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="text-indigo-600 mr-2">•</span>
                    <span className="text-gray-700">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Forecast Chart */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <TrendingUp className="w-8 h-8 text-indigo-600 mr-3" />
              <div>
                <h2 className="text-2xl font-bold text-gray-900">30-Day Balance Forecast</h2>
                <p className="text-gray-600">AI-powered prediction using historical patterns</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Projected Change</p>
              <p className={`text-2xl font-bold ${isPositiveTrend ? 'text-green-600' : 'text-red-600'}`}>
                {isPositiveTrend ? '+' : ''}{balanceChange >= 0 ? '$' : '-$'}{Math.abs(balanceChange).toFixed(2)}
              </p>
            </div>
          </div>

          {forecast.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={forecast}>
                  <defs>
                    <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#4F46E5" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis tickFormatter={(value) => `$${value.toFixed(0)}`} />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                    formatter={(value: any) => [`$${value.toFixed(2)}`, 'Balance']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="predicted_balance" 
                    stroke="#4F46E5" 
                    fillOpacity={1}
                    fill="url(#colorBalance)"
                  />
                </AreaChart>
              </ResponsiveContainer>

              {/* Forecast Summary */}
              <div className="grid md:grid-cols-2 gap-6 mt-6">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-blue-600 font-semibold mb-1">Current Balance</p>
                  <p className="text-2xl font-bold text-blue-900">${currentBalance.toFixed(2)}</p>
                </div>
                <div className={`p-4 rounded-lg ${isPositiveTrend ? 'bg-green-50' : 'bg-orange-50'}`}>
                  <p className={`text-sm font-semibold mb-1 ${isPositiveTrend ? 'text-green-600' : 'text-orange-600'}`}>
                    Projected Balance (30 days)
                  </p>
                  <p className={`text-2xl font-bold ${isPositiveTrend ? 'text-green-900' : 'text-orange-900'}`}>
                    ${finalBalance.toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Warning if balance goes low */}
              {finalBalance < 500 && (
                <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-6 rounded-r-lg">
                  <div className="flex items-start">
                    <AlertTriangle className="w-6 h-6 text-red-500 mr-3 flex-shrink-0 mt-1" />
                    <div>
                      <p className="font-bold text-red-800 text-lg mb-2">Low Balance Warning</p>
                      <p className="text-red-700 mb-3">
                        Your balance is projected to drop to ${finalBalance.toFixed(2)} in 30 days. 
                        This is below the recommended minimum.
                      </p>
                      <p className="text-red-600 text-sm font-semibold">
                        Recommended actions:
                      </p>
                      <ul className="text-red-700 text-sm mt-2 space-y-1">
                        <li>• Reduce discretionary spending by 20%</li>
                        <li>• Review and cancel unused subscriptions</li>
                        <li>• Consider additional income sources</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-600 mb-4">
                Not enough data for accurate forecasting. Add more transactions!
              </p>
              <button
                onClick={generateForecast}
                className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700"
              >
                Try Generate Forecast
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

---

## 🎯 Phase 4: Nudges & Final Polish (Weeks 10-12)

### Week 10: Nudge Engine Implementation

#### Person B - Smart Nudge Generation System

**File: backend/ml/nudges.py**
```python
from sqlalchemy.orm import Session
from models import Nudge, BalanceForecast, User, Transaction
from datetime import datetime, timedelta

def generate_nudges(db: Session, user_id: int):
    """
    Generate smart nudges based on forecasts and spending patterns
    """
    nudges_generated = []
    
    # 1. Check for low balance warnings
    forecasts = db.query(BalanceForecast).filter(
        BalanceForecast.user_id == user_id
    ).order_by(BalanceForecast.forecast_date).all()
    
    if forecasts:
        for forecast in forecasts:
            predicted = float(forecast.predicted_balance)
            days_until = (forecast.forecast_date - datetime.now()).days
            
            # Critical balance warning
            if predicted < 100 and days_until <= 30:
                nudge = Nudge(
                    user_id=user_id,
                    nudge_type="critical_balance_warning",
                    message=f"⚠️ URGENT: Your balance may drop to ${predicted:.2f} in {days_until} days. Take action now to avoid overdraft!",
                    trigger_condition=f"Forecast shows balance < $100 on {forecast.forecast_date}",
                    is_read=False
                )
                db.add(nudge)
                nudges_generated.append(nudge)
                break  # Only one critical warning
            
            # Low balance warning
            elif predicted < 500 and days_until <= 30:
                nudge = Nudge(
                    user_id=user_id,
                    nudge_type="low_balance_warning",
                    message=f"⚠️ Warning: Your balance may drop to ${predicted:.2f} in {days_until} days. Consider reducing discretionary spending.",
                    trigger_condition=f"Forecast shows balance < $500 on {forecast.forecast_date}",
                    is_read=False
                )
                db.add(nudge)
                nudges_generated.append(nudge)
                break
    
    # 2. Check for overspending patterns
    week_ago = datetime.now() - timedelta(days=7)
    recent_transactions = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_date >= week_ago,
        Transaction.transaction_type == 'debit'
    ).all()
    
    if len(recent_transactions) > 0:
        weekly_spending = sum(abs(float(t.amount)) for t in recent_transactions)
        
        # Get historical average for comparison
        month_ago = datetime.now() - timedelta(days=30)
        two_months_ago = datetime.now() - timedelta(days=60)
        
        historical_transactions = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.transaction_date >= two_months_ago,
            Transaction.transaction_date < month_ago,
            Transaction.transaction_type == 'debit'
        ).all()
        
        if len(historical_transactions) > 0:
            historical_weekly_avg = sum(abs(float(t.amount)) for t in historical_transactions) / 4
            
            # Overspending alert
            if weekly_spending > historical_weekly_avg * 1.5:
                # Find top spending category this week
                category_spending = {}
                for t in recent_transactions:
                    cat = t.category
                    category_spending[cat] = category_spending.get(cat, 0) + abs(float(t.amount))
                
                top_category = max(category_spending.items(), key=lambda x: x[1])
                
                nudge = Nudge(
                    user_id=user_id,
                    nudge_type="overspending_alert",
                    message=f"📊 You're spending 50% more than usual this week (${weekly_spending:.2f}). Top category: {top_category[0]} (${top_category[1]:.2f})",
                    trigger_condition="Weekly spending > 1.5x historical average",
                    is_read=False
                )
                db.add(nudge)
                nudges_generated.append(nudge)
    
    # 3. Savings opportunities
    if forecasts and len(forecasts) > 0:
        final_forecast = forecasts[-1]
        if float(final_forecast.predicted_balance) > 2000:
            surplus = float(final_forecast.predicted_balance) - 1500  # Keep 1500 as buffer
            
            nudge = Nudge(
                user_id=user_id,
                nudge_type="savings_opportunity",
                message=f"💰 Great news! You're projected to have ${final_forecast.predicted_balance:.2f} in 30 days. Consider saving ${surplus:.2f}!",
                trigger_condition="30-day forecast shows surplus > $2000",
                is_read=False
            )
            db.add(nudge)
            nudges_generated.append(nudge)
    
    # 4. Recurring payment reminders
    recurring_txns = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.is_recurring == True
    ).all()
    
    if len(recurring_txns) > 5:
        total_recurring = sum(abs(float(t.amount)) for t in recurring_txns if t.transaction_type == 'debit')
        
        nudge = Nudge(
            user_id=user_id,
            nudge_type="subscription_review",
            message=f"🔄 You have {len(recurring_txns)} recurring payments totaling ${total_recurring:.2f}/month. Review subscriptions to save money!",
            trigger_condition=f"User has {len(recurring_txns)} recurring payments",
            is_read=False
        )
        db.add(nudge)
        nudges_generated.append(nudge)
    
    db.commit()
    return nudges_generated

def check_and_generate_daily_nudges(db: Session):
    """
    Run daily to generate nudges for all active users
    """
    users = db.query(User).all()
    total_nudges = 0
    
    for user in users:
        nudges = generate_nudges(db, user.user_id)
        total_nudges += len(nudges) if nudges else 0
    
    return {"total_users": len(users), "total_nudges": total_nudges}
```

Add nudge endpoints to `main.py`:
```python
from ml.nudges import generate_nudges, check_and_generate_daily_nudges

@app.post("/api/nudges/generate/{user_id}")
def create_nudges(user_id: int, db: Session = Depends(get_db)):
    """Manually generate nudges for a user"""
    nudges = generate_nudges(db, user_id)
    return {
        "status": "success", 
        "nudges_created": len(nudges) if nudges else 0,
        "message": f"Generated {len(nudges) if nudges else 0} nudges"
    }

@app.post("/api/nudges/generate-all")
def generate_all_nudges(db: Session = Depends(get_db)):
    """Generate nudges for all users (admin endpoint)"""
    result = check_and_generate_daily_nudges(db)
    return result

@app.get("/api/nudges/{user_id}")
def get_user_nudges(user_id: int, db: Session = Depends(get_db)):
    """Get unread nudges for a user"""
    nudges = db.query(models.Nudge).filter(
        models.Nudge.user_id == user_id,
        models.Nudge.is_read == False
    ).order_by(models.Nudge.created_at.desc()).all()
    
    return {
        "nudges": [{
            "nudge_id": n.nudge_id,
            "type": n.nudge_type,
            "message": n.message,
            "created_at": n.created_at.isoformat()
        } for n in nudges],
        "count": len(nudges)
    }

@app.patch("/api/nudges/{nudge_id}/read")
def mark_nudge_read(nudge_id: int, db: Session = Depends(get_db)):
    """Mark a nudge as read"""
    nudge = db.query(models.Nudge).filter(
        models.Nudge.nudge_id == nudge_id
    ).first()
    
    if nudge:
        nudge.is_read = True
        db.commit()
        return {"status": "success", "message": "Nudge marked as read"}
    
    raise HTTPException(status_code=404, detail="Nudge not found")
```

**Create scheduled task runner - File: backend/scheduler.py**
```python
from apscheduler.schedulers.background import BackgroundScheduler
from ml.nudges import check_and_generate_daily_nudges
from database import SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_daily_nudge_generation():
    """
    Run daily to generate nudges for all users
    """
    logger.info("Starting daily nudge generation...")
    db = SessionLocal()
    
    try:
        result = check_and_generate_daily_nudges(db)
        logger.info(f"✓ Nudge generation complete: {result}")
    except Exception as e:
        logger.error(f"Error in nudge generation: {str(e)}")
    finally:
        db.close()

def start_scheduler():
    """Start the background scheduler"""
    scheduler = BackgroundScheduler()
    
    # Run every day at 9 AM
    scheduler.add_job(
        run_daily_nudge_generation, 
        'cron', 
        hour=9, 
        minute=0,
        id='daily_nudges',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("✓ Scheduler started - Daily nudges will run at 9:00 AM")
    
    return scheduler
```

Update `main.py` to include scheduler:
```python
from scheduler import start_scheduler

scheduler = None

@app.on_event("startup")
def startup_event():
    global scheduler
    init_db()
    scheduler = start_scheduler()
    print("✓ Application started successfully")

@app.on_event("shutdown")
def shutdown_event():
    if scheduler:
        scheduler.shutdown()
    print("✓ Application shutdown complete")
```

### Week 11: Notification Bell UI Component

#### Person A - Interactive Notification System

**File: app/components/NotificationBell.tsx**
```typescript
'use client'

import { useEffect, useState } from 'react'
import { Bell, X } from 'lucide-react'

interface Nudge {
  nudge_id: number
  type: string
  message: string
  created_at: string
}

export default function NotificationBell() {
  const [nudges, setNudges] = useState<Nudge[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchNudges()
    
    // Poll for new nudges every 2 minutes
    const interval = setInterval(fetchNudges, 120000)
    return () => clearInterval(interval)
  }, [])

  const fetchNudges = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/nudges/1')
      const data = await response.json()
      setNudges(data.nudges || [])
    } catch (error) {
      console.error('Error fetching nudges:', error)
    }
  }

  const markAsRead = async (nudgeId: number) => {
    setLoading(true)
    try {
      await fetch(`http://localhost:8000/api/nudges/${nudgeId}/read`, {
        method: 'PATCH'
      })
      setNudges(nudges.filter(n => n.nudge_id !== nudgeId))
    } catch (error) {
      console.error('Error marking nudge as read:', error)
    } finally {
      setLoading(false)
    }
  }

  const getNudgeColor = (type: string) => {
    switch(type) {
      case 'critical_balance_warning':
        return 'border-l-4 border-red-500 bg-red-50'
      case 'low_balance_warning':
        return 'border-l-4 border-orange-500 bg-orange-50'
      case 'overspending_alert':
        return 'border-l-4 border-yellow-500 bg-yellow-50'
      case 'savings_opportunity':
        return 'border-l-4 border-green-500 bg-green-50'
      case 'subscription_review':
        return 'border-l-4 border-blue-500 bg-blue-50'
      default:
        return 'border-l-4 border-gray-500 bg-gray-50'
    }
  }

  const getNudgeIcon = (type: string) => {
    switch(type) {
      case 'critical_balance_warning':
      case 'low_balance_warning':
        return '⚠️'
      case 'overspending_alert':
        return '📊'
      case 'savings_opportunity':
        return '💰'
      case 'subscription_review':
        return '🔄'
      default:
        return '🔔'
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="relative p-2 rounded-full hover:bg-gray-100 transition"
        aria-label="Notifications"
      >
        <Bell className="w-6 h-6 text-gray-700" />
        {nudges.length > 0 && (
          <span className="absolute top-0 right-0 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold animate-pulse">
            {nudges.length}
          </span>
        )}
      </button>

      {showDropdown && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setShowDropdown(false)}
          />
          
          {/* Dropdown */}
          <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-2xl border border-gray-200 z-50 max-h-[600px] overflow-hidden flex flex-col">
            <div className="p-4 border-b bg-gradient-to-r from-indigo-600 to-purple-600">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-lg text-white">Smart Nudges</h3>
                <button
                  onClick={() => setShowDropdown(false)}
                  className="text-white hover:bg-white/20 p-1 rounded"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <p className="text-indigo-100 text-sm mt-1">
                {nudges.length} active notification{nudges.length !== 1 ? 's' : ''}
              </p>
            </div>
            
            <div className="overflow-y-auto flex-1">
              {nudges.length === 0 ? (
                <div className="p-12 text-center">
                  <Bell className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500 font-medium">All caught up!</p>
                  <p className="text-gray-400 text-sm mt-1">No new notifications</p>
                </div>
              ) : (
                <div className="divide-y">
                  {nudges.map(nudge => (
                    <div
                      key={nudge.nudge_id}
                      className={`p-4 hover:bg-gray-50 cursor-pointer transition ${getNudgeColor(nudge.type)}`}
                      onClick={() => !loading && markAsRead(nudge.nudge_id)}
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-2xl flex-shrink-0">{getNudgeIcon(nudge.type)}</span>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-gray-800 leading-relaxed">
                            {nudge.message}
                          </p>
                          <div className="flex items-center justify-between mt-2">
                            <p className="text-xs text-gray-500">
                              {new Date(nudge.created_at).toLocaleString()}
                            </p>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                markAsRead(nudge.nudge_id)
                              }}
                              className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold"
                            >
                              Dismiss
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
```

**Add notification bell to dashboard - Update app/dashboard/page.tsx:**

Add import:
```typescript
import NotificationBell from '../components/NotificationBell'
```

Update header section:
```typescript
<div className="bg-white shadow">
  <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
    <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
    <div className="flex items-center gap-4">
      <NotificationBell />
      <Link href="/dashboard/add-transaction" className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700">
        Add Transaction
      </Link>
      <Link href="/insights" className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700">
        View Insights
      </Link>
    </div>
  </div>
</div>
```

### Week 12: Deployment

#### Person B - Backend Deployment to Railway/Render

**1. Prepare for deployment:**

Create `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Update `requirements.txt`:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
prophet==1.1.5
APScheduler==3.10.4
python-dotenv==1.0.0
```

**2. Railway Deployment Steps:**

1. Sign up at railway.app
2. Create new project
3. Click "Deploy from GitHub repo"
4. Select your repository
5. Add PostgreSQL database (Railway will provide DATABASE_URL)
6. Set environment variables:
   - `DATABASE_URL` (auto-filled by Railway)
   - `SECRET_KEY` (generate a random string)
7. Deploy!

**Alternative: Render Deployment:**

1. Sign up at render.com
2. Create new Web Service
3. Connect GitHub repository
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add PostgreSQL database
7. Set environment variables
8. Deploy!

#### Person A - Frontend Deployment to Vercel

**1. Update configuration:**

Create `.env.production`:
```
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key
CLERK_SECRET_KEY=your_clerk_secret
```

**2. Vercel Deployment Steps:**

1. Push code to GitHub
2. Go to vercel.com
3. Click "Import Project"
4. Select your repository
5. Vercel auto-detects Next.js
6. Add environment variables from .env.production
7. Deploy!

**3. Update CORS in backend after deployment:**

Update `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app"  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ✅ Final Checklist

### Testing
- [ ] All API endpoints work
- [ ] Frontend pages load correctly
- [ ] Authentication flow works
- [ ] Transactions can be added
- [ ] Charts display data
- [ ] ML clustering assigns personas
- [ ] Forecasting generates predictions
- [ ] Nudges appear correctly
- [ ] Notifications can be dismissed

### Quality
- [ ] Error handling implemented
- [ ] Loading states added
- [ ] Responsive design tested
- [ ] Dark mode works (if implemented)
- [ ] Performance optimized

### Documentation
- [ ] README with setup instructions
- [ ] API documentation
- [ ] Deployment guide
- [ ] User guide

---

## 🎓 Post-Project Enhancements

1. **Advanced ML Features:**
   - Anomaly detection for fraud
   - Goal-based budgeting
   - Investment recommendations

2. **User Experience:**
   - Mobile app version
   - Data export features
   - Custom report builder
   - Email notifications

3. **Integrations:**
   - Plaid API for bank connections
   - Payment processors
   - Calendar integrations

4. **Business Features:**
   - Multi-currency support
   - Team/family accounts
   - Premium tier with advanced features

---

## 📚 Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Next.js**: https://nextjs.org/docs
- **scikit-learn**: https://scikit-learn.org/
- **Prophet**: https://facebook.github.io/prophet/
- **PostgreSQL**: https://www.postgresqltutorial.com/

Congratulations on building this comprehensive project! 🎉
