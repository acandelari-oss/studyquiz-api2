import unittest

from sqlalchemy import event, text

from database_engine import create_database_engine


class DatabaseEngineTests(unittest.TestCase):
    def connection_arguments(self, url):
        engine = create_database_engine(url)
        captured = {}

        class ConnectionIntercepted(Exception):
            pass

        @event.listens_for(engine, "do_connect")
        def intercept(dialect, connection_record, args, kwargs):
            captured.update(kwargs)
            raise ConnectionIntercepted()

        try:
            with self.assertRaises(ConnectionIntercepted):
                engine.connect()
        finally:
            engine.dispose()
        return captured

    def test_psycopg_disables_preparation_before_connecting(self):
        args = self.connection_arguments(
            "postgresql+psycopg://test:test@localhost/test"
        )
        self.assertIn("prepare_threshold", args)
        self.assertIsNone(args["prepare_threshold"])

    def test_psycopg2_does_not_receive_psycopg3_option(self):
        args = self.connection_arguments(
            "postgresql+psycopg2://test:test@localhost/test"
        )
        self.assertNotIn("prepare_threshold", args)

    def test_default_postgres_driver_uses_appropriate_options(self):
        from sqlalchemy.engine import make_url

        url = "postgresql://test:test@localhost/test"
        args = self.connection_arguments(url)
        self.assertEqual(
            "prepare_threshold" in args,
            make_url(url).get_dialect().driver == "psycopg",
        )

    def test_sqlite_still_connects(self):
        engine = create_database_engine("sqlite://")
        try:
            with engine.connect() as connection:
                self.assertEqual(connection.execute(text("select 1")).scalar(), 1)
        finally:
            engine.dispose()
