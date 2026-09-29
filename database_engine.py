from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def create_database_engine(database_url):
    url = make_url(database_url)
    connect_args = {}
    # Transaction poolers may reuse server sessions across client connections.
    # Psycopg 3's automatically named prepared statements can collide there.
    if url.get_dialect().driver == "psycopg":
        connect_args["prepare_threshold"] = None
    return create_engine(url, connect_args=connect_args)
