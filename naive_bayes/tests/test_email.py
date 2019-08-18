import pytest
import re

from bs4 import BeautifulSoup

from naive_bayes import EmailObject


CLRF = '\n\n'


@pytest.fixture(scope='function')
def plain_email():
    plain_file = './datasets/tests/plain.eml'
    with open(plain_file, 'r') as f:
        text = f.read()
        email_obj = EmailObject(content=text)
        yield text, email_obj


@pytest.fixture(scope='function')
def html_email():
    plain_file = './datasets/tests/html.eml'
    with open(plain_file, 'r') as f:
        text = f.read()
        email_obj = EmailObject(content=text)
        yield text, email_obj


class TestPlainEmail:

    def test_parse_plain_body(self, plain_email):
        email_content, email_object = plain_email
        body = CLRF.join(email_content.split(CLRF)[1:])
        assert email_object.body == body

    def test_parses_the_subject(self, plain_email):
        email_content, email_object = plain_email
        subject = re.search("Subject: (.*)", email_content).group(1)
        assert email_object.subject == subject


class TestHtmlEmail:

    def test_parses_stores_inner_text_html(self, html_email):
        email_content, email_object = html_email
        body = CLRF.join(email_content.split(CLRF)[1:])
        expected = BeautifulSoup(body).text
        assert email_object.body == expected

    def test_parses_stores_subject(self, html_email):
        email_content, email_object = html_email
        subject = re.search("Subject: (.*)", email_content).group(1)
        assert email_object.subject == subject
