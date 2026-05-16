from sqlalchemy import create_engine, text

engine = create_engine(
    "mysql+pymysql://root:root@localhost/toxicity_db"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT word FROM abuse_words"))

    for row in result:
        print(row[0])