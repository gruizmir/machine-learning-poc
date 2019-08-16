import pytest

from naive_bayes import (
    EmailObject,
    SpamTrainer,
)


@pytest.fixture(scope='module')
def training_set():
    return (
        ['spam', './naive_bayes/datasets/tests/plain.eml'],
        ['ham', './naive_bayes/datasets/tests/small.eml'],
        ['scram', './naive_bayes/datasets/tests/plain.eml'],
    )


@pytest.fixture(scope='module')
def trainer(training_set):
    return SpamTrainer(training_files=training_set)


@pytest.fixture(scope='module')
def spam_email():
    with open('./naive_bayes/datasets/tests/plain.eml') as f:
        return EmailObject(content=f.read())


class TestSpamTrainer:
    # TODO: Repasar la sección de Clasificador Bayesiano
    # y añadir los comentarios necesarios a cada test y funcion

    def test_multiple_categories(self, trainer, training_set):
        categories = trainer.categories
        expected = {cat for cat, path in training_set}
        assert categories == expected

    @pytest.mark.parametrize('category', ['_all', 'spam', 'ham', 'scram'])
    def test_counts_all_at_zero(self, trainer, category):
        assert trainer.total_for(category) == 0

    def test_probability_being_1_over_n(self, trainer, spam_email):
        scores = list(trainer.score(email=spam_email).values())
        for i in range(len(scores) - 1):
            assert round(scores[i], 1) == round(scores[i + 1], 1)

    def test_adds_up_to_one(self, trainer, spam_email):
        scores = list(trainer.normalized_score(email=spam_email).values())
        assert 0.99 <= sum(scores) <= 1.01
        assert 0.49 <= round(scores[0], 2) <= 0.51

    def test_preference_category(self, trainer):
        expected_result = sorted(
            trainer.categories,
            key=lambda cat: trainer.total_for(cat),
        )
        assert trainer.preference == expected_result

    def test_give_preference_to_whatever_has_the_most(self, trainer, spam_email):
        score = trainer.score(email=spam_email)
        preference = trainer.preference[-1]  # Works because we have only two possible results (spam & ham)  # NOQA
        preference_score = score[preference]
        SpamTrainer.classification = (preference, preference_score)

        assert trainer.classify(email=spam_email) == SpamTrainer.classification
