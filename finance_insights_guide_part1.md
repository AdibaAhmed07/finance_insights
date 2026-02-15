# Personal Finance "Behavioral" Insights Engine
## Complete Step-by-Step Development Guide

---

## 📋 Project Overview

**Goal**: Build an intelligent personal finance tool that analyzes spending patterns, predicts financial health, and provides behavioral insights through ML-powered personas and forecasting.

**Tech Stack**:
- Frontend: Next.js, React, Tailwind CSS
- Backend: FastAPI (Python)
- Database: PostgreSQL
- ML Libraries: scikit-learn (K-Means), Prophet (forecasting)
- Authentication: Clerk or NextAuth
- Deployment: Vercel (Frontend), Railway/Render (Backend)

**Team Structure**: Person A (Frontend-focused), Person B (Backend/ML-focused)

---

## 🎯 Phase 1: Foundation (Weeks 1-3)

### Week 1: Project Setup & Database Design

#### Day 1-2: Environment Setup (Both)

**Person A - Frontend Setup**:
```bash
# Install Node.js (v18+) if not already installed
# Create Next.js project
npx create-next-app@latest finance-insights-frontend
# Choose: TypeScript (Yes), Tailwind CSS (Yes), App Router (Yes)

cd finance-insights-frontend
npm install recharts lucide-react date-fns
npm install @clerk/nextjs  # For authentication
```

**Person B - Backend Setup**:
```bash
# Install Python 3.10+ if not already installed
# Create project directory
mkdir finance-insights-backend
cd finance-insights-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic
pip install pandas numpy scikit-learn prophet
pip install python-jose[cryptography] passlib[bcrypt]
pip install python-multipart alembic python-dotenv APScheduler

# Create requirements.txt
pip freeze > requirements.txt
```

#### Day 3-4: Database Schema Design (Both - Collaborative)

**Create Database Schema Document**:

```sql
-- Users Table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts Table
CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    account_name VARCHAR(255) NOT NULL,
    account_type VARCHAR(50), -- checking, savings, credit
    initial_balance DECIMAL(12, 2) DEFAULT 0.00,
    current_balance DECIMAL(12, 2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions Table
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(account_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    transaction_date TIMESTAMP NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    transaction_type VARCHAR(20), -- debit, credit
    category VARCHAR(100), -- groceries, entertainment, bills, etc.
    description TEXT,
    merchant_name VARCHAR(255),
    is_recurring BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Personas Table (ML-generated)
CREATE TABLE user_personas (
    persona_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    persona_label VARCHAR(100), -- "The Frugal Saver", "The Impulsive Techie"
    cluster_number INTEGER,
    avg_transaction_value DECIMAL(12, 2),
    spending_frequency DECIMAL(5, 2),
    top_category VARCHAR(100),
    confidence_score DECIMAL(5, 4),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Balance Forecasts Table
CREATE TABLE balance_forecasts (
    forecast_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    account_id INTEGER REFERENCES accounts(account_id) ON DELETE CASCADE,
    forecast_date DATE NOT NULL,
    predicted_balance DECIMAL(12, 2),
    lower_bound DECIMAL(12, 2),
    upper_bound DECIMAL(12, 2),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Nudges/Notifications Table
CREATE TABLE nudges (
    nudge_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    nudge_type VARCHAR(50), -- overspend_warning, savings_opportunity
    message TEXT NOT NULL,
    trigger_condition TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spending Patterns Table (Analysis results)
CREATE TABLE spending_patterns (
    pattern_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    pattern_type VARCHAR(100), -- "Friday Night Spike", "Weekend Splurger"
    day_of_week INTEGER, -- 0-6
    hour_of_day INTEGER, -- 0-23
    avg_amount DECIMAL(12, 2),
    frequency INTEGER,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Person B**: Set up PostgreSQL database:
```bash
# Install PostgreSQL locally or use a cloud service (ElephantSQL, Supabase)
# Create database
createdb finance_insights_db

# Create .env file in backend directory
DATABASE_URL=postgresql://username:password@localhost:5432/finance_insights_db
SECRET_KEY=your-secret-key-here-generate-random-string
```

#### Day 5-7: Backend Models Setup

**Person B**: Create the complete backend structure.

**File: backend/models.py**
```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    personas = relationship("UserPersona", back_populates="user", cascade="all, delete-orphan")

class Account(Base):
    __tablename__ = "accounts"
    
    account_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    account_name = Column(String, nullable=False)
    account_type = Column(String)
    initial_balance = Column(Numeric(12, 2), default=0.00)
    current_balance = Column(Numeric(12, 2), default=0.00)
    currency = Column(String, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account")

class Transaction(Base):
    __tablename__ = "transactions"
    
    transaction_id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.account_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    transaction_date = Column(DateTime, nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    transaction_type = Column(String)
    category = Column(String)
    description = Column(Text)
    merchant_name = Column(String)
    is_recurring = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="transactions")
    account = relationship("Account", back_populates="transactions")

class UserPersona(Base):
    __tablename__ = "user_personas"
    
    persona_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    persona_label = Column(String)
    cluster_number = Column(Integer)
    avg_transaction_value = Column(Numeric(12, 2))
    spending_frequency = Column(Numeric(5, 2))
    top_category = Column(String)
    confidence_score = Column(Numeric(5, 4))
    assigned_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="personas")

class BalanceForecast(Base):
    __tablename__ = "balance_forecasts"
    
    forecast_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    account_id = Column(Integer, ForeignKey("accounts.account_id"))
    forecast_date = Column(DateTime, nullable=False)
    predicted_balance = Column(Numeric(12, 2))
    lower_bound = Column(Numeric(12, 2))
    upper_bound = Column(Numeric(12, 2))
    generated_at = Column(DateTime, default=datetime.utcnow)

class Nudge(Base):
    __tablename__ = "nudges"
    
    nudge_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    nudge_type = Column(String)
    message = Column(Text, nullable=False)
    trigger_condition = Column(Text)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SpendingPattern(Base):
    __tablename__ = "spending_patterns"
    
    pattern_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    pattern_type = Column(String)
    day_of_week = Column(Integer)
    hour_of_day = Column(Integer)
    avg_amount = Column(Numeric(12, 2))
    frequency = Column(Integer)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
```

**File: backend/database.py**
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
```

**File: backend/main.py** (Initial version)
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, init_db
import models

app = FastAPI(title="Finance Insights API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def read_root():
    return {"message": "Finance Insights API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Test endpoint
@app.get("/api/test")
def test_endpoint():
    return {"message": "API is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the backend:
```bash
uvicorn main:app --reload
```

**Person A**: Set up authentication with Clerk.

**Installation:**
```bash
npm install @clerk/nextjs
```

**File: app/layout.tsx**
```typescript
import { ClerkProvider } from '@clerk/nextjs'
import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'FinanceIQ - Smart Finance Tracking',
  description: 'AI-powered personal finance insights',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>{children}</body>
      </html>
    </ClerkProvider>
  )
}
```

**File: .env.local** (Create this in your frontend directory)
```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
CLERK_SECRET_KEY=sk_test_your_key_here
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
```

Get your Clerk keys from: https://clerk.com (sign up for free account)

### Week 2: Landing Page & Basic Backend

#### Person A - Landing Page

**File: app/page.tsx**
```typescript
import Link from 'next/link'
import { ArrowRight, TrendingUp, Brain, Bell } from 'lucide-react'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
        <div className="text-2xl font-bold text-indigo-600">FinanceIQ</div>
        <div className="space-x-4">
          <Link href="/sign-in" className="text-gray-600 hover:text-indigo-600">
            Login
          </Link>
          <Link href="/sign-up" className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700">
            Sign Up
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-6">
          Understand Your Money,<br />Transform Your Future
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          AI-powered insights that predict your financial behavior, 
          identify spending patterns, and help you make smarter decisions.
        </p>
        <Link href="/sign-up" className="inline-flex items-center bg-indigo-600 text-white px-8 py-4 rounded-lg text-lg hover:bg-indigo-700">
          Get Started Free
          <ArrowRight className="ml-2" />
        </Link>
      </section>

      {/* Features */}
      <section className="container mx-auto px-6 py-20">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <TrendingUp className="w-12 h-12 text-indigo-600 mb-4" />
            <h3 className="text-2xl font-bold mb-4">Smart Forecasting</h3>
            <p className="text-gray-600">
              Predict your balance 30 days ahead using advanced ML models
            </p>
          </div>
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <Brain className="w-12 h-12 text-indigo-600 mb-4" />
            <h3 className="text-2xl font-bold mb-4">Behavioral Personas</h3>
            <p className="text-gray-600">
              Discover your unique spending personality and patterns
            </p>
          </div>
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <Bell className="w-12 h-12 text-indigo-600 mb-4" />
            <h3 className="text-2xl font-bold mb-4">Smart Nudges</h3>
            <p className="text-gray-600">
              Get warned before overspending based on your patterns
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-8 mt-20 border-t">
        <p className="text-center text-gray-600">
          © 2024 FinanceIQ. Built with Next.js and FastAPI.
        </p>
      </footer>
    </div>
  )
}
```

**Create sign-in page: app/sign-in/[[...sign-in]]/page.tsx**
```typescript
import { SignIn } from "@clerk/nextjs"

export default function Page() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <SignIn />
    </div>
  )
}
```

**Create sign-up page: app/sign-up/[[...sign-up]]/page.tsx**
```typescript
import { SignUp } from "@clerk/nextjs"

export default function Page() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <SignUp />
    </div>
  )
}
```

### Week 3: Synthetic Data Generation

#### Person B - Generate Synthetic Dataset

This is critical - you need realistic test data!

**File: backend/data_generator.py**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_transactions(num_users=100, months=12):
    """
    Generate synthetic transaction data for testing
    """
    categories = [
        'Groceries', 'Dining', 'Entertainment', 'Transportation',
        'Shopping', 'Bills', 'Healthcare', 'Travel', 'Technology',
        'Subscriptions', 'Fitness', 'Education'
    ]
    
    merchants = {
        'Groceries': ['Whole Foods', 'Trader Joes', 'Safeway', 'Local Market'],
        'Dining': ['Chipotle', 'Starbucks', 'Local Cafe', 'Pizza Place', 'Sushi Bar'],
        'Entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Concert Venue'],
        'Transportation': ['Uber', 'Lyft', 'Gas Station', 'Parking'],
        'Shopping': ['Amazon', 'Target', 'Nike', 'Best Buy'],
        'Bills': ['Electric Company', 'Internet Provider', 'Water Utility'],
        'Technology': ['Apple Store', 'Electronics Shop', 'Software Subscription'],
        'Healthcare': ['Pharmacy', 'Doctor Office', 'Gym'],
        'Travel': ['Airbnb', 'Hotel', 'Airline'],
        'Subscriptions': ['Netflix', 'Spotify', 'Adobe', 'Cloud Storage'],
        'Fitness': ['Gym Membership', 'Yoga Studio', 'Sports Equipment'],
        'Education': ['Online Course', 'Books', 'Training Program']
    }
    
    transactions = []
    user_personas = {
        'frugal': {'avg_spend': 30, 'frequency': 'low', 'multiplier': 0.7},
        'moderate': {'avg_spend': 75, 'frequency': 'medium', 'multiplier': 1.0},
        'impulsive': {'avg_spend': 150, 'frequency': 'high', 'multiplier': 1.5}
    }
    
    for user_id in range(1, num_users + 1):
        # Assign persona
        persona = random.choice(list(user_personas.keys()))
        persona_data = user_personas[persona]
        
        # Generate transactions for this user
        start_date = datetime.now() - timedelta(days=months * 30)
        
        if persona == 'impulsive':
            num_transactions = random.randint(200, 500)
        elif persona == 'moderate':
            num_transactions = random.randint(100, 250)
        else:
            num_transactions = random.randint(50, 150)
        
        for _ in range(num_transactions):
            category = random.choice(categories)
            merchant = random.choice(merchants.get(category, ['Generic Merchant']))
            
            # Create date with patterns
            days_offset = random.randint(0, months * 30)
            transaction_date = start_date + timedelta(days=days_offset)
            
            # Add realistic time
            hour = random.randint(6, 23)
            minute = random.randint(0, 59)
            transaction_date = transaction_date.replace(hour=hour, minute=minute)
            
            # Weekend spending spike for some personas
            amount_multiplier = 1.0
            if persona == 'impulsive' and transaction_date.weekday() >= 5:
                amount_multiplier = 1.5
            
            # Friday night spike
            if transaction_date.weekday() == 4 and transaction_date.hour >= 18:
                amount_multiplier *= 1.3
            
            base_amount = persona_data['avg_spend']
            amount = round(base_amount * amount_multiplier * random.uniform(0.5, 2.5), 2)
            
            transactions.append({
                'user_id': user_id,
                'transaction_date': transaction_date,
                'amount': -amount,  # Negative for expenses
                'transaction_type': 'debit',
                'category': category,
                'merchant_name': merchant,
                'description': f'{category} purchase at {merchant}',
                'is_recurring': category in ['Bills', 'Subscriptions']
            })
        
        # Add income transactions (monthly salary)
        for month in range(months):
            payday = start_date + timedelta(days=month * 30 + 15)
            payday = payday.replace(hour=9, minute=0)
            salary = random.uniform(3000, 6000)
            
            transactions.append({
                'user_id': user_id,
                'transaction_date': payday,
                'amount': salary,
                'transaction_type': 'credit',
                'category': 'Income',
                'merchant_name': 'Employer',
                'description': 'Monthly salary',
                'is_recurring': True
            })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values(['user_id', 'transaction_date'])
    df.to_csv('synthetic_transactions.csv', index=False)
    print(f"Generated {len(df)} transactions for {num_users} users")
    print(f"\nSample statistics:")
    print(df.groupby('user_id')['amount'].describe())
    return df

if __name__ == "__main__":
    df = generate_synthetic_transactions(num_users=100, months=12)
    print("\nFirst 10 transactions:")
    print(df.head(10))
```

Run the generator:
```bash
python data_generator.py
```

**File: backend/load_data.py**
```python
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
from models import User, Account, Transaction
import pandas as pd
from datetime import datetime
import hashlib

def load_synthetic_data():
    """Load synthetic data into database"""
    init_db()
    db = SessionLocal()
    
    # Read CSV
    df = pd.read_csv('synthetic_transactions.csv')
    
    # Get unique user IDs
    user_ids = df['user_id'].unique()
    
    print(f"Loading data for {len(user_ids)} users...")
    
    for user_id in user_ids:
        # Create user
        user = User(
            email=f"user{user_id}@financeiq.com",
            password_hash=hashlib.sha256(f"password{user_id}".encode()).hexdigest(),
            full_name=f"Test User {user_id}"
        )
        db.add(user)
        db.flush()
        
        # Create account
        account = Account(
            user_id=user.user_id,
            account_name="Main Checking Account",
            account_type="checking",
            initial_balance=5000.00,
            current_balance=5000.00
        )
        db.add(account)
        db.flush()
        
        # Load transactions for this user
        user_transactions = df[df['user_id'] == user_id]
        
        running_balance = 5000.00
        for _, row in user_transactions.iterrows():
            transaction = Transaction(
                account_id=account.account_id,
                user_id=user.user_id,
                transaction_date=pd.to_datetime(row['transaction_date']),
                amount=row['amount'],
                transaction_type=row['transaction_type'],
                category=row['category'],
                merchant_name=row['merchant_name'],
                description=row['description'],
                is_recurring=row['is_recurring']
            )
            db.add(transaction)
            
            # Update running balance
            running_balance += float(row['amount'])
        
        # Update final account balance
        account.current_balance = running_balance
        
        if user_id % 10 == 0:
            print(f"Loaded data for user {user_id}...")
    
    db.commit()
    db.close()
    print("✓ Data loading complete!")

if __name__ == "__main__":
    load_synthetic_data()
```

Run the data loader:
```bash
python load_data.py
```

---

## 🎯 Phase 2: Core Features & Data Analysis (Weeks 4-6)

### Week 4: Transaction Dashboard

#### Person A - Dashboard UI

**File: app/dashboard/page.tsx**
```typescript
'use client'

import { useEffect, useState } from 'react'
import { useUser } from '@clerk/nextjs'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts'
import { DollarSign, TrendingDown, TrendingUp, Activity } from 'lucide-react'
import Link from 'next/link'

interface Transaction {
  transaction_id: number
  transaction_date: string
  amount: number
  category: string
  merchant_name: string
  transaction_type: string
}

export default function Dashboard() {
  const { user } = useUser()
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [balance, setBalance] = useState(0)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({
    monthlyIncome: 0,
    monthlyExpenses: 0,
    topCategory: '',
  })

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/dashboard/1')
      const data = await response.json()
      
      setTransactions(data.transactions)
      setBalance(data.balance)
      
      // Calculate stats
      const income = data.transactions
        .filter((t: Transaction) => t.transaction_type === 'credit')
        .reduce((sum: number, t: Transaction) => sum + Math.abs(t.amount), 0)
      
      const expenses = data.transactions
        .filter((t: Transaction) => t.transaction_type === 'debit')
        .reduce((sum: number, t: Transaction) => sum + Math.abs(t.amount), 0)
      
      // Get top category
      const categoryTotals: { [key: string]: number } = {}
      data.transactions
        .filter((t: Transaction) => t.transaction_type === 'debit')
        .forEach((t: Transaction) => {
          categoryTotals[t.category] = (categoryTotals[t.category] || 0) + Math.abs(t.amount)
        })
      
      const topCat = Object.keys(categoryTotals).reduce((a, b) => 
        categoryTotals[a] > categoryTotals[b] ? a : b, ''
      )
      
      setStats({
        monthlyIncome: income,
        monthlyExpenses: expenses,
        topCategory: topCat
      })
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Prepare chart data
  const prepareCategoryData = () => {
    const categoryTotals: { [key: string]: number } = {}
    transactions
      .filter(t => t.transaction_type === 'debit')
      .forEach(t => {
        categoryTotals[t.category] = (categoryTotals[t.category] || 0) + Math.abs(t.amount)
      })
    
    return Object.keys(categoryTotals).map(category => ({
      name: category,
      value: categoryTotals[category]
    })).sort((a, b) => b.value - a.value).slice(0, 6)
  }

  const COLORS = ['#4F46E5', '#7C3AED', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-xl text-gray-600">Loading your dashboard...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <div className="flex gap-4">
            <Link href="/dashboard/add-transaction" className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700">
              Add Transaction
            </Link>
            <Link href="/insights" className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700">
              View Insights
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-8">
        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm text-gray-600">Current Balance</h3>
              <DollarSign className="w-5 h-5 text-indigo-600" />
            </div>
            <p className="text-3xl font-bold text-gray-900">
              ${balance.toFixed(2)}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm text-gray-600">Monthly Income</h3>
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <p className="text-3xl font-bold text-green-600">
              ${stats.monthlyIncome.toFixed(2)}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm text-gray-600">Monthly Expenses</h3>
              <TrendingDown className="w-5 h-5 text-red-600" />
            </div>
            <p className="text-3xl font-bold text-red-600">
              ${stats.monthlyExpenses.toFixed(2)}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm text-gray-600">Top Category</h3>
              <Activity className="w-5 h-5 text-purple-600" />
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {stats.topCategory}
            </p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* Spending by Category - Pie Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">Spending by Category</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={prepareCategoryData()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {prepareCategoryData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: any) => `$${value.toFixed(2)}`} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Spending Trend - Bar Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold mb-4">Top Categories</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={prepareCategoryData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip formatter={(value: any) => `$${value.toFixed(2)}`} />
                <Bar dataKey="value" fill="#4F46E5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Transactions */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold mb-4">Recent Transactions</h3>
          <div className="space-y-3">
            {transactions.slice(0, 15).map((txn) => (
              <div key={txn.transaction_id} className="flex justify-between items-center border-b pb-3">
                <div>
                  <p className="font-semibold text-gray-900">{txn.merchant_name}</p>
                  <p className="text-sm text-gray-600">{txn.category} • {new Date(txn.transaction_date).toLocaleDateString()}</p>
                </div>
                <p className={`font-bold text-lg ${txn.amount < 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {txn.amount < 0 ? '-' : '+'}${Math.abs(txn.amount).toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
```

#### Person B - Dashboard API Endpoint

Add to `backend/main.py`:
```python
from typing import List
from pydantic import BaseModel

class TransactionResponse(BaseModel):
    transaction_id: int
    transaction_date: str
    amount: float
    category: str
    merchant_name: str
    transaction_type: str
    
    class Config:
        from_attributes = True

@app.get("/api/dashboard/{user_id}")
def get_dashboard(user_id: int, db: Session = Depends(get_db)):
    # Get user's transactions (last 90 days)
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=90)
    
    transactions = db.query(models.Transaction).filter(
        models.Transaction.user_id == user_id,
        models.Transaction.transaction_date >= cutoff_date
    ).order_by(models.Transaction.transaction_date.desc()).all()
    
    # Get current balance
    account = db.query(models.Account).filter(
        models.Account.user_id == user_id
    ).first()
    
    # Convert to dict
    transaction_list = [{
        "transaction_id": t.transaction_id,
        "transaction_date": t.transaction_date.isoformat(),
        "amount": float(t.amount),
        "category": t.category,
        "merchant_name": t.merchant_name,
        "transaction_type": t.transaction_type
    } for t in transactions]
    
    return {
        "transactions": transaction_list,
        "balance": float(account.current_balance) if account else 0.0
    }
```

### Week 5: Manual Transaction Entry

#### Person A - Transaction Form

**File: app/dashboard/add-transaction/page.tsx**
```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

export default function AddTransaction() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    amount: '',
    category: '',
    merchant: '',
    date: new Date().toISOString().split('T')[0],
    type: 'debit',
    description: ''
  })
  const [submitting, setSubmitting] = useState(false)

  const categories = [
    'Groceries', 'Dining', 'Entertainment', 'Transportation',
    'Shopping', 'Bills', 'Healthcare', 'Travel', 'Technology',
    'Subscriptions', 'Fitness', 'Education'
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    
    try {
      const response = await fetch('http://localhost:8000/api/transactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 1, // In production, get from auth
          amount: formData.type === 'debit' ? -parseFloat(formData.amount) : parseFloat(formData.amount),
          category: formData.category,
          merchant_name: formData.merchant,
          transaction_date: formData.date,
          transaction_type: formData.type,
          description: formData.description
        })
      })
      
      if (response.ok) {
        router.push('/dashboard')
      } else {
        alert('Error adding transaction')
      }
    } catch (error) {
      console.error('Error adding transaction:', error)
      alert('Error adding transaction')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <Link href="/dashboard" className="inline-flex items-center text-gray-600 hover:text-gray-900">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Link>
        </div>
      </div>

      <div className="max-w-2xl mx-auto p-8">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold mb-8">Add Transaction</h1>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Transaction Type
              </label>
              <div className="grid grid-cols-2 gap-4">
                <button
                  type="button"
                  onClick={() => setFormData({...formData, type: 'debit'})}
                  className={`p-4 border-2 rounded-lg text-center font-semibold transition ${
                    formData.type === 'debit'
                      ? 'border-red-500 bg-red-50 text-red-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  Expense
                </button>
                <button
                  type="button"
                  onClick={() => setFormData({...formData, type: 'credit'})}
                  className={`p-4 border-2 rounded-lg text-center font-semibold transition ${
                    formData.type === 'credit'
                      ? 'border-green-500 bg-green-50 text-green-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  Income
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Amount
              </label>
              <div className="relative">
                <span className="absolute left-3 top-3 text-gray-500">$</span>
                <input
                  type="number"
                  step="0.01"
                  value={formData.amount}
                  onChange={(e) => setFormData({...formData, amount: e.target.value})}
                  className="w-full pl-8 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="0.00"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Category
              </label>
              <select
                value={formData.category}
                onChange={(e) => setFormData({...formData, category: e.target.value})}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                required
              >
                <option value="">Select a category</option>
                {categories.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Merchant / Source
              </label>
              <input
                type="text"
                value={formData.merchant}
                onChange={(e) => setFormData({...formData, merchant: e.target.value})}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="e.g., Starbucks, Salary"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Date
              </label>
              <input
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({...formData, date: e.target.value})}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                rows={3}
                placeholder="Additional notes..."
              />
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full bg-indigo-600 text-white py-4 rounded-lg text-lg font-semibold hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {submitting ? 'Adding...' : 'Add Transaction'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
```

#### Person B - Transaction Creation Endpoint

Add to `backend/main.py`:
```python
from pydantic import BaseModel
from datetime import datetime

class TransactionCreate(BaseModel):
    user_id: int
    amount: float
    category: str
    merchant_name: str
    transaction_date: str
    transaction_type: str
    description: str = ""

@app.post("/api/transactions")
def create_transaction(transaction: TransactionCreate, db: Session = Depends(get_db)):
    # Get user's account
    account = db.query(models.Account).filter(
        models.Account.user_id == transaction.user_id
    ).first()
    
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Create transaction
    new_transaction = models.Transaction(
        account_id=account.account_id,
        user_id=transaction.user_id,
        transaction_date=datetime.fromisoformat(transaction.transaction_date),
        amount=transaction.amount,
        transaction_type=transaction.transaction_type,
        category=transaction.category,
        merchant_name=transaction.merchant_name,
        description=transaction.description,
        is_recurring=False
    )
    
    db.add(new_transaction)
    
    # Update account balance
    account.current_balance = float(account.current_balance) + transaction.amount
    
    db.commit()
    db.refresh(new_transaction)
    
    return {"message": "Transaction created successfully", "transaction_id": new_transaction.transaction_id}
```

### Week 6: Time-Series Analysis

#### Person B - Spending Pattern Analysis

**File: backend/analysis/time_series.py**

```python
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from models import Transaction, SpendingPattern
from datetime import datetime, timedelta

def analyze_spending_cycles(db: Session, user_id: int):
    """
    Identify spending patterns based on day of week and time
    """
    # Get all debit transactions for user
    transactions = db.query(Transaction).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'debit'
    ).all()
    
    if len(transactions) < 10:  # Need minimum data
        return {"error": "Insufficient data"}
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': t.transaction_date,
        'amount': abs(float(t.amount)),
        'category': t.category,
        'day_of_week': t.transaction_date.weekday(),
        'hour': t.transaction_date.hour
    } for t in transactions])
    
    patterns_found = []
    
    # 1. Analyze by day of week
    daily_patterns = df.groupby('day_of_week').agg({
        'amount': ['mean', 'sum', 'count']
    }).round(2)
    
    # 2. Identify "Friday Night Spike"
    friday_evening = df[
        (df['day_of_week'] == 4) &  # Friday
        (df['hour'] >= 18)  # After 6 PM
    ]
    
    if len(friday_evening) > 5:  # Need minimum occurrences
        avg_friday_evening = friday_evening['amount'].mean()
        overall_avg = df['amount'].mean()
        
        if avg_friday_evening > overall_avg * 1.3:  # 30% higher
            # Delete old pattern if exists
            db.query(SpendingPattern).filter(
                SpendingPattern.user_id == user_id,
                SpendingPattern.pattern_type == "Friday Night Spike"
            ).delete()
            
            pattern = SpendingPattern(
                user_id=user_id,
                pattern_type="Friday Night Spike",
                day_of_week=4,
                hour_of_day=18,
                avg_amount=avg_friday_evening,
                frequency=len(friday_evening),
                analyzed_at=datetime.utcnow()
            )
            db.add(pattern)
            patterns_found.append("Friday Night Spike")
    
    # 3. Weekend vs Weekday comparison
    weekend_spending = df[df['day_of_week'].isin([5, 6])]['amount'].mean()
    weekday_spending = df[~df['day_of_week'].isin([5, 6])]['amount'].mean()
    
    if weekend_spending > weekday_spending * 1.5:
        # Delete old pattern
        db.query(SpendingPattern).filter(
            SpendingPattern.user_id == user_id,
            SpendingPattern.pattern_type == "Weekend Splurger"
        ).delete()
        
        pattern = SpendingPattern(
            user_id=user_id,
            pattern_type="Weekend Splurger",
            day_of_week=None,
            hour_of_day=None,
            avg_amount=weekend_spending,
            frequency=len(df[df['day_of_week'].isin([5, 6])]),
            analyzed_at=datetime.utcnow()
        )
        db.add(pattern)
        patterns_found.append("Weekend Splurger")
    
    db.commit()
    
    return {
        'patterns_found': patterns_found,
        'daily_averages': daily_patterns.to_dict(),
        'friday_night_avg': float(avg_friday_evening) if len(friday_evening) > 0 else 0,
        'weekend_avg': float(weekend_spending),
        'weekday_avg': float(weekday_spending)
    }
```

Add API endpoint to `main.py`:
```python
from analysis.time_series import analyze_spending_cycles

@app.get("/api/analysis/patterns/{user_id}")
def get_spending_patterns(user_id: int, db: Session = Depends(get_db)):
    analysis = analyze_spending_cycles(db, user_id)
    
    # Get saved patterns from DB
    saved_patterns = db.query(models.SpendingPattern).filter(
        models.SpendingPattern.user_id == user_id
    ).all()
    
    pattern_list = [{
        "pattern_type": p.pattern_type,
        "avg_amount": float(p.avg_amount) if p.avg_amount else 0,
        "frequency": p.frequency
    } for p in saved_patterns]
    
    return {
        "analysis": analysis,
        "patterns": pattern_list
    }
```

---

Due to length limits, I need to continue this in multiple parts. This covers Phases 1-2. Would you like me to continue with:

1. **Phase 3** (ML Clustering & Forecasting)
2. **Phase 4** (Nudges & Deployment)
3. Create additional helper files and scripts

Let me know and I'll continue building out the complete guide!
