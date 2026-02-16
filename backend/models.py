from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


# =========================
# USER TABLE
# =========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)

    # Relationship to transactions
    transactions = relationship("Transaction", back_populates="user")


# =========================
# TRANSACTION TABLE
# =========================
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    category = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", back_populates="transactions")
