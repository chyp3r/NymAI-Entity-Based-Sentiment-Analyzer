import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np
import tensorflow as tf
import transformers

from . import alignment
from . import utils
from .data_types import TokenizedExample, Example, LabeledExample, PredictedExample, SubTask, CompletedSubTask, Task, CompletedTask, InputBatch, OutputBatch, Sentiment
from .models import BertABSClassifier
from .training import classifier_loss
from .professors import Professor

# Logger ayarları
logger = logging.getLogger('absa.pipeline')

@dataclass
class _Pipeline(ABC):
    @abstractmethod
    def __call__(self, text: str, aspects: List[str]) -> CompletedTask:
        """
        İş hattının ana işlemi.
        """

    @abstractmethod
    def preprocess(self, text: str, aspects: List[str]) -> Task:
        """
        Ön işleme adımı.
        """

    @abstractmethod
    def tokenize(self, examples: Iterable[Example]) -> Iterable[TokenizedExample]:
        """
        Tokenize etme adımı.
        """

    @abstractmethod
    def encode(self, examples: Iterable[TokenizedExample]) -> InputBatch:
        """
        Kodlama adımı.
        """

    @abstractmethod
    def predict(self, input_batch: InputBatch) -> OutputBatch:
        """
        Tahmin adımı.
        """

    @staticmethod
    def postprocess(task: Task, batch_examples: Iterable[PredictedExample]) -> CompletedTask:
        """
        Son işlem adımı.
        """

    @abstractmethod
    def evaluate(self, examples: Iterable[LabeledExample], metric: tf.metrics.Metric, batch_size: int) -> tf.Tensor:
        """
        Değerlendirme adımı.
        """

@dataclass
class Pipeline(_Pipeline):
    model: BertABSClassifier
    tokenizer: transformers.BertTokenizer
    professor: Professor
    text_splitter: Callable[[str], List[str]] = None

    def __call__(self, text: str, aspects: List[str]) -> CompletedTask:
        """
        İş hattının ana çağrı metodu. Tüm işlemleri sırasıyla gerçekleştirir.
        """
        task = self.preprocess(text, aspects)
        predictions = self.transform(task.examples)
        completed_task = self.postprocess(task, predictions)
        return completed_task

    def preprocess(self, text: str, aspects: List[str]) -> Task:
        """
        Metni ve yanıtları ön işler.
        """
        spans = self.text_splitter(text) if self.text_splitter else [text]
        subtasks = OrderedDict()
        for aspect in aspects:
            examples = [Example(span, aspect) for span in spans]
            subtasks[aspect] = SubTask(text, aspect, examples)
        task = Task(text, aspects, subtasks)
        return task

    def transform(self, examples: Iterable[Example]) -> Iterable[PredictedExample]:
        """
        Tokenize etme, kodlama ve tahmin adımlarını gerçekleştirir.
        """
        tokenized_examples = self.tokenize(examples)
        input_batch = self.encode(tokenized_examples)
        output_batch = self.predict(input_batch)
        predictions = self.review(tokenized_examples, output_batch)
        return predictions

    def tokenize(self, examples: Iterable[Example]) -> List[TokenizedExample]:
        """
        Örnekleri tokenize eder.
        """
        return [alignment.tokenize(self.tokenizer, e.text, e.aspect) for e in examples]

    def encode(self, examples: Iterable[TokenizedExample]) -> InputBatch:
        """
        Tokenize edilmiş örnekleri kodlar.
        """
        token_pairs = [(e.text_subtokens, e.aspect_subtokens) for e in examples]
        encoded = self.tokenizer.batch_encode_plus(
            token_pairs,
            add_special_tokens=True,
            padding=True,
            return_tensors='tf',
            return_attention_masks=True,
            max_length=512
        )
        batch = InputBatch(
            token_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            token_type_ids=encoded['token_type_ids']
        )
        return batch

    def predict(self, input_batch: InputBatch) -> OutputBatch:
        """
        Kodlanmış girdilerle modelden tahminler yapar.
        """
        with tf.GradientTape() as tape:
            logits, hidden_states, attentions = self.model.call(
                input_ids=input_batch.token_ids,
                attention_mask=input_batch.attention_mask,
                token_type_ids=input_batch.token_type_ids
            )

            predictions = tf.argmax(logits, axis=-1)
            labels = tf.one_hot(predictions, depth=3)
            loss_value = classifier_loss(labels, logits)
        attention_grads = tape.gradient(loss_value, attentions)

        scores = tf.nn.softmax(logits, axis=1)

        stack = lambda x, order: tf.transpose(tf.stack(x), order)
        hidden_states = stack(hidden_states, [1, 0, 2, 3])
        attentions = stack(attentions, [1, 0, 2, 3, 4])
        attention_grads = stack(attention_grads, [1, 0, 2, 3, 4])
        output_batch = OutputBatch(
            scores=scores,
            hidden_states=hidden_states,
            attentions=attentions,
            attention_grads=attention_grads
        )
        return output_batch

    def review(self, examples: Iterable[TokenizedExample], output_batch: OutputBatch) -> Iterable[PredictedExample]:
        """
        Tahminleri gözden geçirir ve etiketler.
        """
        return (self.professor.review(e, o) for e, o in zip(examples, output_batch))

    @staticmethod
    def postprocess(task: Task, batch_examples: Iterable[PredictedExample]) -> CompletedTask:
        """
        Son işleme adımı. Görevleri tamamlama.
        """
        batch_examples = list(batch_examples)  # Örnekleri materialize eder.
        subtasks = OrderedDict()
        for start, end in task.indices:
            examples = batch_examples[start:end]
            # Örnekler aynı yanıta sahip olmalıdır (implicit bir kontrol).
            aspect, = {e.aspect for e in examples}
            scores = np.max([e.scores for e in examples], axis=0)
            scores /= np.linalg.norm(scores, ord=1)
            sentiment_id = np.argmax(scores).astype(int)
            aspect_document = CompletedSubTask(
                text=task.text,
                aspect=aspect,
                examples=examples,
                sentiment=Sentiment(sentiment_id),
                scores=list(scores)
            )
            subtasks[aspect] = aspect_document
        task = CompletedTask(task.text, task.aspects, subtasks)
        return task

    def evaluate(self, examples: Iterable[LabeledExample], metric: tf.metrics.Metric, batch_size: int) -> tf.Tensor:
        """
        Modeli değerlendirir.
        """
        batches = utils.batches(examples, batch_size)
        for batch in batches:
            predictions = self.transform(batch)
            y_pred = [e.sentiment.value for e in predictions]
            y_true = [e.sentiment.value for e in batch]
            metric.update_state(y_true, y_pred)
        result = metric.result()
        return result
