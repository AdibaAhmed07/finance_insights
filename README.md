# Personal Finance Behavioral Insights Engine 💰🤖

A full-stack AI-powered personal finance application that analyzes spending patterns, predicts financial health, and provides behavioral insights through machine learning.

## 🌟 Features

### Core Functionality
- **Transaction Management**: Manual entry and automatic tracking
- **Smart Dashboard**: Real-time balance and spending visualization
- **AI-Powered Personas**: ML clustering to identify spending personalities
- **30-Day Forecasting**: Prophet-based balance predictions
- **Smart Nudges**: Proactive alerts for overspending and savings opportunities
- **Time-Series Analysis**: Identify spending cycles and patterns

### Technical Highlights
- **Frontend**: Next.js 14, React, Tailwind CSS, TypeScript
- **Backend**: FastAPI, Python, SQLAlchemy
- **Database**: PostgreSQL
- **ML**: scikit-learn (K-Means), Facebook Prophet
- **Authentication**: Clerk
- **Deployment**: Vercel (Frontend), Railway/Render (Backend)

## 📊 Project Architecture

```
finance-insights/
├── frontend/                 # Next.js application
│   ├── app/
│   │   ├── page.tsx         # Landing page
│   │   ├── dashboard/       # Main dashboard
│   │   ├── insights/        # AI insights page
│   │   └── components/      # React components
│   └── public/
│
└── backend/                  # FastAPI application
    ├── main.py              # API endpoints
    ├── models.py            # Database models
    ├── database.py          # DB connection
    ├── ml/
    │   ├── clustering.py    # K-Means personas
    │   ├── forecasting.py   # Prophet forecasting
    │   └── nudges.py        # Smart alerts
    ├── analysis/
    │   └── time_series.py   # Pattern analysis
    ├── data_generator.py    # Synthetic data
    └── scheduler.py         # Background tasks
```

## 🚀 Quick Start Guide

### Prerequisites
- Node.js 18+
- Python 3.10+
- PostgreSQL 14+
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-insights.git
cd finance-insights

# Frontend setup
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local with your Clerk keys

# Backend setup
cd ../backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb finance_insights_db

# Create .env file in backend directory
echo "DATABASE_URL=postgresql://username:password@localhost:5432/finance_insights_db" > .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

### 3. Generate Test Data

```bash
cd backend
python data_generator.py
python load_data.py
```

This creates 100 test users with realistic transaction data.

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```
Backend runs at: http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend runs at: http://localhost:3000

### 5. Initial ML Setup

Once both servers are running:

```bash
# Generate personas (one-time)
curl -X POST http://localhost:8000/api/ml/cluster-users

# Generate forecasts (can run anytime)
curl -X POST http://localhost:8000/api/ml/forecast/1

# Generate nudges (runs automatically daily at 9 AM)
curl -X POST http://localhost:8000/api/nudges/generate/1
```

## 📖 Usage Guide

### For Person A (Frontend Developer)

**Week 1-2: Setup & Landing**
- Create Next.js app
- Implement Clerk authentication
- Build landing page with features

**Week 4-5: Dashboard & Forms**
- Build transaction dashboard
- Implement charts (Recharts)
- Create transaction entry form

**Week 8-9: Insights Page**
- Display persona information
- Show forecast charts
- Add personalized tips

**Week 11: Notifications**
- Build notification bell component
- Implement real-time nudge display
- Add dismiss functionality

### For Person B (Backend/ML Developer)

**Week 1-2: Database & API**
- Set up PostgreSQL schema
- Create SQLAlchemy models
- Build basic CRUD endpoints

**Week 3: Data Generation**
- Generate synthetic transaction data
- Create realistic spending patterns
- Load data into database

**Week 6: Time-Series Analysis**
- Identify spending cycles
- Detect Friday night spikes
- Analyze weekend vs weekday patterns

**Week 7-8: ML Clustering**
- Extract user features
- Implement K-Means clustering
- Assign financial personas

**Week 8-9: Forecasting**
- Prepare data for Prophet
- Train forecasting models
- Generate 30-day predictions

**Week 10: Nudge Engine**
- Implement nudge logic
- Create background scheduler
- Set up automated alerts

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing Checklist
- [ ] User can sign up and login
- [ ] Dashboard displays transactions
- [ ] New transactions can be added
- [ ] Persona is correctly assigned
- [ ] Forecast displays 30-day prediction
- [ ] Nudges appear when triggered
- [ ] Notifications can be dismissed

## 🎯 Financial Personas

The ML model assigns users to one of these personas:

1. **The Frugal Saver** 💚
   - Low spending, infrequent purchases
   - High savings potential
   - Very disciplined

2. **The Balanced Spender** 💙
   - Moderate spending and saving
   - Sustainable financial habits
   - Well-rounded approach

3. **The Impulsive Techie** 💜
   - Frequent purchases
   - Tech & subscription heavy
   - Needs spending awareness

4. **The Weekend Warrior** 🧡
   - Weekend spending spikes
   - Entertainment focused
   - Needs weekend budgeting

5. **The Big Ticket Buyer** ❤️
   - Fewer, larger purchases
   - High-value items
   - Needs emergency fund

## 🔔 Smart Nudge Types

1. **Critical Balance Warning**: Balance < $100
2. **Low Balance Warning**: Balance < $500
3. **Overspending Alert**: 50%+ increase vs historical
4. **Savings Opportunity**: Surplus > $2000
5. **Subscription Review**: 5+ recurring payments

## 📈 ML Models Explained

### K-Means Clustering (Personas)
- **Features Used**: avg transaction, frequency, weekend ratio, categories
- **Algorithm**: K-Means with 5 clusters
- **Preprocessing**: StandardScaler normalization
- **Output**: Financial persona label + confidence score

### Prophet Forecasting (Balance)
- **Input**: Daily balance time series
- **Model**: Facebook Prophet with weekly seasonality
- **Horizon**: 30 days ahead
- **Output**: Predicted balance + confidence intervals

## 🚢 Deployment

### Backend (Railway)
1. Create Railway account
2. Connect GitHub repository
3. Add PostgreSQL database
4. Set environment variables
5. Deploy automatically on push

### Frontend (Vercel)
1. Create Vercel account
2. Import GitHub repository
3. Auto-detects Next.js config
4. Add environment variables
5. Deploy automatically on push

## 🔧 Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)
```
DATABASE_URL=postgresql://user:pass@localhost:5432/finance_insights_db
SECRET_KEY=your-secret-key-here
```

## 📊 Database Schema

### Core Tables
- **users**: User accounts and auth
- **accounts**: Bank accounts
- **transactions**: All financial transactions
- **user_personas**: ML-assigned personas
- **balance_forecasts**: Prophet predictions
- **nudges**: Smart notifications
- **spending_patterns**: Identified patterns

## 🎓 Learning Outcomes

By completing this project, you'll learn:

1. **Full-Stack Development**
   - Next.js with TypeScript
   - FastAPI with Python
   - PostgreSQL database design

2. **Machine Learning**
   - K-Means clustering
   - Time-series forecasting
   - Feature engineering

3. **Data Visualization**
   - Recharts library
   - Interactive dashboards
   - Real-time updates

4. **DevOps**
   - Docker containers
   - Cloud deployment
   - CI/CD pipelines

5. **Best Practices**
   - RESTful API design
   - Component architecture
   - Error handling
   - Security (auth, CORS)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - feel free to use this project for learning!

## 🙋 Support

- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Documentation**: See `docs/` folder

## 🎉 Acknowledgments

- FastAPI documentation
- Next.js team
- Facebook Prophet team
- Clerk authentication
- scikit-learn community

---

**Built with ❤️ as a comprehensive full-stack ML project**

Happy coding! 🚀
