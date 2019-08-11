import pytest

from naive_bayes import (
    EmailObject,
    SpamTrainer,
)

@pytest.fixture(scope='module')
def training_set():
    return (
        ['spam', './datasets/bayes/plain.eml'],
        ['ham', './datasets/bayes/small.eml'],
        ['scram', './datasets/bayes/plain.eml']
    )


@pytest.fixture(scope='module')
def trainer(training_set):
    return SpamTrainer(training_files=training_set)


@pytest.fixture(scope='module')
def test_email():
    with open('./datasets/bayes/plain.eml') as f:
        return EmailObject(content=f)


class TestSpamTrainer:

    def test_multiple_categories(self, trainer, training_set):
        categories = trainer.categories
        expected = {cat for cat, path in training_set}
        assert categories == expected

    @pytest.mark.parametrize('category', ['_all', 'spam', 'ham', 'scram'])
    def test_counts_all_at_zero(self, trainer, category):
        assert trainer.total_for(category) == 0
