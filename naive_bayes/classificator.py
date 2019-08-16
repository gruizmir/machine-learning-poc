import os
import fire

from naive_bayes.spam_trainer import SpamTrainer
from naive_bayes.email_object import EmailObject



class Classificator:
    correct = 0
    false_positives = 0.0
    false_negatives = 0.0
    confidence = 0.0
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def load_data(self, fold):
        with open(fold) as f:
            training_data = [
                line.rstrip().split(' ')
                for line in f.readlines()
            ]
        # print(training_data)
        return SpamTrainer(training_files=training_data)

    def parse_emails(self, keyfile):
        print(f'Parsing emails for {keyfile}')
        emails = []
        with open(keyfile) as kf:
            for line in kf.readlines():
                label, filename = line.strip().split(' ')
                path = os.path.join(self.BASE_DIR, 'datasets', 'bayes', filename)
                with open(path) as email_file:
                    email_obj = EmailObject(content=email_file.read(), category=label)
                    emails.append(email_obj)

        print(f'Done parsing emails for {keyfile}')
        print(f'Parsed {len(emails)} emails')
        return emails

    def run(self, fold, keyfile):
        '''
        Execution example:

        ./classificator.py run fold1.label
        '''
        fold = os.path.join(self.BASE_DIR, 'datasets', 'bayes', fold)
        keyfile = os.path.join(self.BASE_DIR, 'datasets', 'bayes', keyfile)

        trainer = self.load_data(fold=fold)
        emails = self.parse_emails(keyfile=keyfile)
        for email in emails:
            classification = trainer.classify(email=email)
            self.confidence += classification.score

            if classification.guess == 'spam' and email.category == 'ham':
                self.false_positives += 1
            elif classification.guess == 'ham' and email.category == 'spam':
                self.false_negatives += 1
            else:
                self.correct += 1
        self.print_results()

    def print_results(self):
        total = self.false_negatives + self.false_positives + self.correct
        false_positive_rate = self.false_positives / total
        false_negative_rate = self.false_negatives / total
        error_rate = (self.false_positives + self.false_negatives) / total
        print(
            f'''
                False Positives: {false_positive_rate}
                False Negatives: {false_negative_rate}
                Error Rate: {error_rate}
            '''
        )


if __name__ == '__main__':
    fire.Fire(Classificator)
