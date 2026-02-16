# Used for generating random numbers and choices
import random

# Used to handle date calculations
from datetime import datetime, timedelta

# Import database session creator
from database import SessionLocal

# Import SQLAlchemy models (User, Transaction)
import models


# =========================
# CONSTANTS
# =========================

# Different spending categories for transactions
CATEGORIES = [
    "groceries",
    "entertainment",
    "tech",
    "rent",
    "subscriptions",
    "dining",
    "travel"
]

# Different behavioral personas for realistic spending patterns
PERSONA_TYPES = [
    "frugal",
    "balanced",
    "impulsive",
    "weekend",
    "big_spender"
]


# =========================
# USER GENERATION
# =========================

def generate_users(db, n=50):
    """
    Generates 'n' users and saves them to the database.

    Parameters:
    - db: active database session
    - n: number of users to generate (default 50)
    """
    users = []

    for i in range(n):
        # Create user with unique email
        user = models.User(email=f"user{i}@test.com")

        # Add user to session
        db.add(user)

        # Keep reference for later transaction generation
        users.append(user)

    # Commit all users to database
    db.commit()

    return users


# =========================
# TRANSACTION GENERATION
# =========================

def generate_transactions_for_user(db, user, persona):
    """
    Generates 6 months (180 days) of transactions for a given user,
    based on their spending persona.
    """

    # Start from 180 days ago
    start_date = datetime.now() - timedelta(days=180)

    # Loop through each day
    for day in range(180):
        date = start_date + timedelta(days=day)

        # =========================
        # Persona-based logic
        # =========================

        if persona == "frugal":
            # Very few transactions, low spending
            daily_transactions = random.randint(0, 1)
            amount_range = (5, 30)

        elif persona == "balanced":
            # Moderate spending
            daily_transactions = random.randint(1, 2)
            amount_range = (10, 70)

        elif persona == "impulsive":
            # Frequent and high spending
            daily_transactions = random.randint(1, 4)
            amount_range = (20, 200)

        elif persona == "weekend":
            # Spends more on weekends
            if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                daily_transactions = random.randint(2, 4)
                amount_range = (30, 150)
            else:
                daily_transactions = random.randint(0, 1)
                amount_range = (10, 40)

        else:  # big_spender
            # Large but less frequent spending
            daily_transactions = random.randint(0, 2)
            amount_range = (50, 500)

        # =========================
        # Create daily transactions
        # =========================

        for _ in range(daily_transactions):
            transaction = models.Transaction(
                user_id=user.id,  # Link transaction to user
                amount=random.uniform(*amount_range),  # Random amount in range
                category=random.choice(CATEGORIES),   # Random category
                date=date  # Transaction date
            )

            db.add(transaction)

    # =========================
    # Add Monthly Subscriptions
    # =========================

    # Add one subscription every month for 6 months
    for month in range(6):
        sub_date = start_date + timedelta(days=month * 30)

        subscription = models.Transaction(
            user_id=user.id,
            amount=random.uniform(10, 50),
            category="subscriptions",
            date=sub_date
        )

        db.add(subscription)

    # Save all transactions to database
    db.commit()


# =========================
# MAIN EXECUTION
# =========================

def main():
    """
    Entry point of the script.
    Creates users and generates transactions for each.
    """

    # Create database session
    db = SessionLocal()

    print("Generating users...")
    users = generate_users(db, n=50)

    print("Generating transactions...")

    # Assign each user a random spending persona
    for user in users:
        persona = random.choice(PERSONA_TYPES)
        generate_transactions_for_user(db, user, persona)

    # Close database connection
    db.close()

    print("Done generating data!")


# Run script only if executed directly
if __name__ == "__main__":
    main()
