import unittest
from unittest.mock import Mock
from sqlalchemy.exc import OperationalError
from taxonomy_assignment import initial_source_read


def disconnected(invalidated=True):
    return OperationalError('select', {}, Exception('closed'), connection_invalidated=invalidated)


class InitialSourceReadTests(unittest.TestCase):
    def session(self):
        db = Mock()
        db.in_transaction.return_value = False
        db.get_bind.return_value.dialect.name = 'postgresql'
        return db

    def test_success_is_read_only_and_releases_transaction(self):
        db = self.session()
        result = object()
        self.assertIs(initial_source_read(db, lambda log: result), result)
        self.assertEqual(str(db.execute.call_args.args[0]), 'SET TRANSACTION READ ONLY')
        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_snapshot_disconnect_reloads_all_reads_without_partial_results(self):
        db = self.session()
        counts = {'chunks': 0, 'coverage': 0, 'snapshot': 0}
        def load(log):
            for stage in counts:
                counts[stage] += 1
                log(stage)
            if counts['snapshot'] == 1:
                raise disconnected()
            return ['fresh rows'], {'fresh snapshot'}
        self.assertEqual(initial_source_read(db, load), (['fresh rows'], {'fresh snapshot'}))
        self.assertEqual(counts, dict.fromkeys(counts, 2))
        self.assertEqual(db.rollback.call_count, 2)
        self.assertEqual(db.execute.call_count, 2)

    def test_second_disconnect_propagates_without_third_attempt(self):
        db = self.session()
        load = Mock(side_effect=disconnected())
        with self.assertRaises(OperationalError):
            initial_source_read(db, load)
        self.assertEqual(load.call_count, 2)

    def test_non_disconnect_and_coverage_failures_are_not_retried(self):
        for error in (disconnected(False), ValueError('incomplete coverage')):
            db = self.session()
            load = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                initial_source_read(db, load)
            load.assert_called_once()
            db.rollback.assert_called_once()

    def test_pending_transaction_cannot_be_retried_or_rolled_back(self):
        db = self.session()
        db.in_transaction.return_value = True
        load = Mock()
        with self.assertRaises(ValueError):
            initial_source_read(db, load)
        load.assert_not_called()
        db.rollback.assert_not_called()
