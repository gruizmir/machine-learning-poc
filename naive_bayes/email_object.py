import email

from bs4 import BeautifulSoup


class EmailObject:

    def __init__(self, content=None, category=None):
        self.content = content
        self.category = category
        self._body = None
        self._subject = None
        self._email = None

    @property
    def mail(self):
        self._mail = getattr(self, '_mail', None) or email.message_from_string(self.content)
        return self._mail

    @property
    def subject(self):
        return self.mail.get('Subject')

    @property
    def body(self):
        self._body = getattr(self, '_body', None)
        if not self._body:
            payload = self.mail.get_payload()
            content_type = self.mail.get_content_type()
            if content_type == 'text/html':
                self._body = BeautifulSoup(payload).text
            elif content_type == 'text/plain':
                self._body = payload
            else:
                raise AttributeError('Only Html or Plain text are allowed as email body')

        return self._body
