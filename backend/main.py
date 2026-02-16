from fastapi import FastAPI
from database import engine
import models
from fastapi import Depends
from sqlalchemy.orm import Session
from database import SessionLocal
import schemas

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend is running"}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create a new user
@app.post("/users", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    POST /users

    This endpoint creates a new user in the database.

    - `user`: Request body validated using UserCreate schema
    - `db`: Database session provided by dependency injection
    """

    # Create a new SQLAlchemy User object using the email from request
    new_user = models.User(email=user.email)

    # Add the new user object to the database session
    db.add(new_user)

    # Commit the transaction (actually saves to database)
    db.commit()

    # Refresh the object to get updated values (like auto-generated ID)
    db.refresh(new_user)

    # Return the newly created user
    # FastAPI automatically converts it using UserResponse schema
    return new_user


# Get all users
@app.get("/users", response_model=list[schemas.UserResponse])
def get_users(db: Session = Depends(get_db)):
    """
    GET /users

    This endpoint retrieves all users from the database.

    - `db`: Database session provided by dependency injection
    """

    # Query the database to fetch all users
    users = db.query(models.User).all()

    # Return list of users
    # FastAPI converts each user using UserResponse schema
    return users

# Create a new transaction
@app.post("/transactions", response_model=schemas.TransactionResponse)
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    """
    POST /transactions

    This endpoint creates a new transaction for a specific user.

    - `transaction`: Request body validated using TransactionCreate schema
    - `db`: Database session provided using dependency injection
    """

    # Create a new SQLAlchemy Transaction object
    # Values are taken from the validated request body
    new_transaction = models.Transaction(
        user_id=transaction.user_id,     # Foreign key referencing User table
        amount=transaction.amount,       # Transaction amount
        category=transaction.category,   # Category (e.g., Food, Rent, Salary)
        date=transaction.date            # Optional date (can be None)
    )

    # Add transaction object to database session
    db.add(new_transaction)

    # Commit the transaction (saves to database permanently)
    db.commit()

    # Refresh object to retrieve auto-generated values (like ID)
    db.refresh(new_transaction)

    # Return the created transaction
    # FastAPI converts it to TransactionResponse schema automatically
    return new_transaction


# Get all transactions for a specific user
@app.get("/transactions/{user_id}", response_model=list[schemas.TransactionResponse])
def get_transactions(user_id: int, db: Session = Depends(get_db)):
    """
    GET /transactions/{user_id}

    This endpoint retrieves all transactions belonging to a specific user.

    - `user_id`: Path parameter (taken from URL)
    - `db`: Database session provided using dependency injection
    """

    # Query database to filter transactions by user_id
    transactions = db.query(models.Transaction) \
                     .filter(models.Transaction.user_id == user_id) \
                     .all()

    # Return list of transactions
    # FastAPI automatically serializes them using TransactionResponse schema
    return transactions


