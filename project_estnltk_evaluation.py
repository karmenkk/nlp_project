from estnltk.ner import NerTagger, NerTrainer
from estnltk.corpus import read_json_corpus
from seqeval.metrics import f1_score, classification_report
import settings_for_eval as settings


# Read the training corpus
corpus = read_json_corpus('data/project_estner/project_train.json')

# Directory to save the model
model_dir = 'for_project_evaluation'

# Train and save the model
trainer = NerTrainer(settings)
trainer.train(corpus, model_dir)

# Load the model and settings
tagger = NerTagger('for_project_evaluation')

# ne-tag the test corpus (json-format, labels removed)
test_corpus = read_json_corpus('data/project_estner/project_test_no_labels.json')
tagged_docs = tagger.tag_documents(test_corpus)

# Create list of lists of predicted labels
predicted_labels = []
for elem in tagged_docs:
    predicted_labels.extend(elem.labels)

# Read in the true labels from corrected dataset
with open('data/project_estner/project_test_manual_corrs.txt', encoding='utf-8') as f:
    test_labels = []
    doc_labels = []
    for line in f:
        line = line.strip()
        array = line.split('\t')
        if len(array) > 1:
            label = array[1]
            doc_labels.append(label)
        else:
            test_labels.extend(doc_labels)
            doc_labels = []

print(len(predicted_labels), len(test_labels))
print(f1_score(test_labels, predicted_labels))
print(classification_report(test_labels, predicted_labels, digits=3))