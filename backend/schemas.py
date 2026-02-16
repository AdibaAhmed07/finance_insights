# Import BaseModel to create request/response schemas
from pydantic import BaseModel

# Used for transaction date field
from datetime import datetime

# Used to make fields optional
from typing import Optional


# =========================
# USER SCHEMAS
# =========================

class UserCreate(BaseModel):
    """
    Schema used when creating a new user.
    This validates incoming request data.
    """
    email: str  # User's email address (required field)


class UserResponse(BaseModel):
    """
    Schema used when returning user data in API responses.
    """
    id: int      # Unique user ID (comes from database)
    email: str   # User's email

    class Config:
        # Allows FastAPI to convert SQLAlchemy model objects
        # into Pydantic response objects automatically
        orm_mode = True


# =========================
# TRANSACTION SCHEMAS
# =========================

class TransactionCreate(BaseModel):
    """
    Schema used when creating a new transaction.
    """
    user_id: int           # ID of the user who owns this transaction
    amount: float          # Transaction amount (positive/negative)
    category: str          # Category like 'Food', 'Rent', etc.
    date: Optional[datetime] = None  
    # Date is optional.
    # If not provided, you can set default in database.


class TransactionResponse(BaseModel):
    """
    Schema used when returning transaction data in responses.
    """
    id: int                # Unique transaction ID
    user_id: int           # Associated user ID
    amount: float          # Transaction amount
    category: str          # Transaction category
    date: datetime         # Date of transaction

    class Config:
        # Enables compatibility with SQLAlchemy ORM models
        orm_mode = True
