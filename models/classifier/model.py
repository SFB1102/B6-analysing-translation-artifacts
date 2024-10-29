import torch
import transformers
from tqdm import tqdm
import os
import json


def get_data(path):
    with open(path, "r", encoding="utf-8") as input_file:
        data = []
        for k, line in enumerate(input_file):
            json_line = json.loads(line)
            sentence = json_line["sentence"]
            label = json_line["label"]
            data.append((sentence, label, k))
    return data


def data_prefix_path(*path):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", *path
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


class XLMRBinaryClassifier:
    def __init__(self, dataset_name, model_name="xlm-roberta-base", num_labels=2):
        global g_storage
        self.dataset_name = dataset_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, from_tf=False, output_hidden_states=True
        ).to(device)

    def train(
        self,
        train_texts,
        train_labels,
        batch_size=16,
        num_epochs=10,
        learning_rate=2e-5,
    ):
        # Tokenize training texts
        train_encodings = self.tokenizer(train_texts, max_length=512, padding=True)

        # Convert encoded inputs and labels into PyTorch tensors
        train_inputs = torch.tensor(train_encodings["input_ids"]).to(device)
        train_masks = torch.tensor(train_encodings["attention_mask"]).to(device)
        train_labels = torch.tensor(train_labels).to(device)

        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_inputs) * num_epochs,
        )

        # Define loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train model
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for i in tqdm(range(0, len(train_inputs), batch_size)):
                optimizer.zero_grad()
                input_ids = train_inputs[i : i + batch_size]
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=train_masks[i : i + batch_size],
                    labels=None,
                )

                logits = outputs.logits
                loss = loss_fn(logits.view(-1, 2), train_labels[i : i + batch_size])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / (len(train_inputs) / batch_size)
            print(f"Epoch {epoch + 1} loss: {avg_loss:.3f}")

    def predict(self, test_text):
        # Tokenize human-eval text
        test_encoding = self.tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Predict label and probability
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**test_encoding)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()
            proba = probs[0][pred_label].item()

        return pred_label, proba

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))
