"""
dbert.py
Provides the dBert class that implements Reader using BERT contextual embeddings to disambiguate heteronyms.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from speach.ttlig import RubyFrag, RubyToken
from transformers import (
    AutoModelForTokenClassification,
    BertJapaneseTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from yomikata import utils
from yomikata.config import config, logger
from yomikata.reader import Reader
from yomikata.utils import LabelEncoder

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


class dBert(Reader):
    def __init__(
        self,
        artifacts_dir: Path = config.DBERT_DIR,
        reinitialize: bool = False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        # Set the device
        self.device = device
        logger.info(f"Running on {self.device}")
        if self.device.type == "cuda":
            logger.info(torch.cuda.get_device_name(0))

        # Hardcoded parameters
        self.max_length = 128

        # Load the model
        self.artifacts_dir = artifacts_dir
        if reinitialize:
            # load tokenizer from upstream huggingface repository
            default_model = "cl-tohoku/bert-base-japanese-v2"
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(default_model)
            logger.info(f"Using {default_model} tokenizer")

            # load the heteronyms list
            self.heteronyms = config.HETERONYMS

            # make the label encoder
            label_list = ["<OTHER>"]
            for i, heteronym in enumerate(self.heteronyms.keys()):
                for j, reading in enumerate(self.heteronyms[heteronym]):
                    label_list.append(heteronym + ":" + reading)

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label_list)

            logger.info("Made label encoder with default heteronyms")

            # add surface forms to tokenizer vocab
            surfaces = list(
                set([x.split(":")[0] for x in self.label_encoder.classes if x != "<OTHER>"])
            )

            new_tokens = [
                surface
                for surface in surfaces
                if surface
                not in (list(self.tokenizer.vocab.keys()) + list(self.tokenizer.get_added_vocab()))
            ]

            self.tokenizer.add_tokens(new_tokens)
            if len(new_tokens) > 0:
                logger.info(f"Added {len(new_tokens)} surface forms to tokenizer vocab")

            # check that new tokens were added properly
            assert [
                self.tokenizer.decode(
                    self.tokenizer.encode(
                        [surface],
                        add_special_tokens=False,
                    )
                )
                for surface in surfaces
            ] == surfaces

            self.surfaceIDs = self.tokenizer.encode(
                list(set([x.split(":")[0] for x in self.label_encoder.classes if x != "<OTHER>"])),
                add_special_tokens=False,
            )
            assert len(self.surfaceIDs) == len(surfaces)

            # Load model from upstream huggingface repository
            self.model = AutoModelForTokenClassification.from_pretrained(
                default_model, num_labels=len(self.label_encoder.classes)
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Using model {default_model}")

            self.save(artifacts_dir)
        else:
            self.load(artifacts_dir)

    def load(self, directory):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(directory)
        self.model = AutoModelForTokenClassification.from_pretrained(directory).to(self.device)
        self.label_encoder = LabelEncoder.load(Path(directory, "label_encoder.json"))
        self.heteronyms = utils.load_dict(Path(directory, "heteronyms.json"))

        self.surfaceIDs = self.tokenizer.encode(
            list(set([x.split(":")[0] for x in self.label_encoder.classes if x != "<OTHER>"])),
            add_special_tokens=False,
        )
        logger.info(f"Loaded model from directory {directory}")

    def save(self, directory):
        self.tokenizer.save_pretrained(directory)
        self.model.save_pretrained(directory)
        self.label_encoder.save(Path(directory, "label_encoder.json"))
        utils.save_dict(self.heteronyms, Path(directory, "heteronyms.json"))
        logger.info(f"Saved model to directory {directory}")

    def batch_preprocess_function(self, entries, pad=False):
        inputs = [entry for entry in entries["sentence"]]
        furiganas = [entry for entry in entries["furigana"]]
        if pad:
            tokenized_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                # return_tensors="np",
            )
        else:
            tokenized_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                truncation=True,
            )

        labels = []
        for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
            furigana_temp = furiganas[i]
            label_ids = []
            assert inputs[i] == utils.remove_furigana(furiganas[i])
            for j, input_id in enumerate(input_ids):
                if input_id not in self.surfaceIDs:
                    label = -100
                else:
                    surface = self.tokenizer.decode([input_id])
                    try:
                        reading_start_idx = furigana_temp.index(surface) + len(surface)
                        furigana_temp = furigana_temp[reading_start_idx + 1 :]
                        reading_end_idx = furigana_temp.index("}")
                        reading = furigana_temp[:reading_end_idx]
                        furigana_temp = furigana_temp[reading_end_idx + 1 :]
                        label = self.label_encoder.class_to_index[surface + ":" + reading]
                    except KeyError:
                        # this means there's an unknown reading
                        label = 0
                    except ValueError:
                        # this means that the surface form is not present in the furigana
                        # probably it got split between two different words
                        label = 0
                label_ids.append(label)
            assert len(label_ids) == len(input_ids)
            labels.append(label_ids)

        assert len(labels) == len(tokenized_inputs["input_ids"])

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
        }

    def train(self, dataset, training_args={}) -> dict:
        dataset = dataset.map(
            self.batch_preprocess_function, batched=True, fn_kwargs={"pad": False}
        )
        dataset = dataset.filter(
            lambda entry: any(x in entry["input_ids"] for x in list(self.surfaceIDs))
        )

        # put the model in training mode
        self.model.train()

        default_training_args = {
            "output_dir": self.artifacts_dir,
            "num_train_epochs": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 10,
            "logging_strategy": "steps",
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 10,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 128,
            "per_device_eval_batch_size": 128,
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss",
            "weight_decay": 0.01,
            "save_total_limit": 3,
            "fp16": True,
            "report_to": "tensorboard",
        }

        default_training_args.update(training_args)
        training_args = default_training_args

        # Not padding in batch_preprocess_function so need data_collator for trainer
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)

        if "val" in list(dataset):
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=dataset["train"],
                eval_dataset=dataset["val"],
                tokenizer=self.tokenizer,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=5),
                ],
                data_collator=data_collator,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=dataset["train"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        result = trainer.train()

        # Output some training information
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])
        utils.print_gpu_utilization(gpu_index)

        # Get metrics for each train/val/split
        self.model.eval()
        full_performance = {}
        for key in dataset.keys():
            max_evals = min(100000, len(dataset[key]))
            # max_evals = len(dataset[key])
            logger.info(f"getting predictions for {key}")
            subset = dataset[key].shuffle().select(range(max_evals))
            prediction_output = trainer.predict(subset)
            logger.info(f"processing predictions for {key}")
            metrics = prediction_output[2]
            labels = prediction_output[1]
            predictions = np.argmax(prediction_output[0], axis=2)

            true_inputs = [
                self.tokenizer.decode([input_id])
                for row in subset["input_ids"]
                for input_id in row
                if input_id in self.surfaceIDs
            ]

            true_predictions = [
                str(self.label_encoder.index_to_class[p])
                for prediction, label in zip(predictions, labels)
                for (p, l) in zip(prediction, label)
                if l != -100
            ]

            true_labels = [
                str(self.label_encoder.index_to_class[l])
                for prediction, label in zip(predictions, labels)
                for (p, l) in zip(prediction, label)
                if l != -100
            ]

            logger.info("processing performance")
            performance = {
                heteronym: {
                    "n": 0,
                    "readings": {
                        reading: {
                            "n": 0,
                            "found": {
                                readingprime: 0
                                for readingprime in list(self.heteronyms[heteronym].keys())
                                + ["<OTHER>"]
                            },
                        }
                        for reading in list(self.heteronyms[heteronym].keys()) + ["<OTHER>"]
                    },
                }
                for heteronym in self.heteronyms.keys()
            }

            for i, surface in enumerate(true_inputs):
                performance[surface]["n"] += 1

                true_reading = true_labels[i].split(":")[-1]

                performance[surface]["readings"][true_reading]["n"] += 1

                if true_predictions[i] != "<OTHER>":
                    if true_predictions[i].split(":")[0] != surface:
                        logger.warning(f"big failure at {surface} {true_predictions[i]}")
                        found_reading = "<OTHER>"
                    else:
                        found_reading = true_predictions[i].split(":")[1]
                else:
                    found_reading = "<OTHER>"

                performance[surface]["readings"][true_reading]["found"][found_reading] += 1

                # if found_reading != true_reading:
                #     # pass
                #     logger.info(
                #         f"Predicted {found_reading} instead of {true_reading} in {subset["furigana"][furi_rows[i]]}"
                #     )

            n = 0
            correct = 0
            for surface in performance.keys():
                for true_reading in performance[surface]["readings"].keys():
                    performance[surface]["readings"][true_reading]["accuracy"] = np.round(
                        performance[surface]["readings"][true_reading]["found"][true_reading]
                        / np.array(performance[surface]["readings"][true_reading]["n"]),
                        3,
                    )

                performance[surface]["accuracy"] = np.round(
                    sum(
                        performance[surface]["readings"][true_reading]["found"][true_reading]
                        for true_reading in performance[surface]["readings"].keys()
                    )
                    / np.array(performance[surface]["n"]),
                    3,
                )

                correct += sum(
                    performance[surface]["readings"][true_reading]["found"][true_reading]
                    for true_reading in performance[surface]["readings"].keys()
                )
                n += performance[surface]["n"]

            performance = {
                "metrics": metrics,
                "accuracy": round(correct / n, 3),
                "heteronym_performance": performance,
            }

            full_performance[key] = performance

        trainer.save_model()

        return full_performance

    def furigana(self, text: str) -> str:
        text = utils.standardize_text(text)
        text = utils.remove_furigana(text)
        text = text.replace("{", "").replace("}", "")

        self.model.eval()

        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_encoded["input_ids"].to(self.device)
        input_mask = text_encoded["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=input_mask).logits

        predictions = torch.argmax(logits, dim=2)

        output_ruby = []
        for i, p in enumerate(predictions[0]):
            text = self.tokenizer.decode([input_ids[0][i]])
            if text in ["[CLS]", "[SEP]"]:
                continue
            if text[:2] == "##":
                text = text[2:]
            if input_ids[0][i].item() in self.surfaceIDs:
                furi = self.label_encoder.index_to_class[p.item()]

                if furi == "<OTHER>":
                    output_ruby.append(f"{{{text}}}")
                elif furi.split(":")[0] != text:
                    output_ruby.append(f"{{{text}}}")
                else:
                    output_ruby.append(RubyFrag(text=text, furi=furi.split(":")[1]))
            else:
                output_ruby.append(text)

        return RubyToken(groups=output_ruby).to_code()
