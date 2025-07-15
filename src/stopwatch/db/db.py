from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:////db/stopwatch.db")
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()
